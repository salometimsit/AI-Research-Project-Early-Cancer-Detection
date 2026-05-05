"""TCAV scoring core: hooks, CAV training, directional derivatives.

Pipeline stage: post-K interpretability of the SwinViT track.

This module owns the actual TCAV mechanics:

1. Resolve a target submodule by dotted name (with sensible fallbacks).
2. Register a forward hook to extract pooled per-patient activations.
3. Train one linear CAV per concept via ``LogisticRegression`` (delegated
   to :class:`TCAV3D`'s static helper).
4. Compute directional derivatives ``<grad_h logit_target, cav>`` per
   sample using ``torch.autograd.grad`` and aggregate them into the
   TCAV score.
5. Estimate two-sided permutation significance against random concepts.

Failure modes
-------------
* **Layer not found** -> ``LookupError``; the helper logs known module
  names so you can pick a valid one.
* **No grad flow** -> ``RuntimeError``; ensure the hooked layer is on
  the forward path between input and the target logit.
* **Degenerate CAV** -> ``ValueError`` raised by :func:`TCAV3D.train_cav`
  when activations are linearly inseparable up to numerical noise.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "TCAV core requires torch, numpy and scikit-learn. "
        "Install them to enable TCAV scoring."
    ) from exc

from src.utils.logger import get_logger
from src.utils.tcav.types import CAVRecord, ConceptSets

logger = get_logger(__name__)


class TCAV3D:
    """Compute TCAV scores against a 3D SwinViT model.

    Parameters
    ----------
    model:
        A trained ``SwinViT3D`` (or any ``nn.Module``) whose forward
        pass returns ``(logits, deep_features)``.
    target_layer_name:
        Dotted submodule path under ``model`` to hook (e.g.
        ``"encoder.layers3"``).
    target_class:
        Index of the class whose logit we differentiate.
    device:
        Torch device (defaults to CUDA if available).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str,
        target_class: int = 1,
        device: torch.device | str | None = None,
        cav_c: float = 0.1,
    ) -> None:
        self.model = model
        self.target_class = int(target_class)
        self.cav_c = float(cav_c)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()
        self.target_layer_name = self._resolve_layer_name(target_layer_name)
        self.target_layer = self._get_submodule(self.target_layer_name)

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------
    def _list_module_names(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

    def _resolve_layer_name(self, requested: str) -> str:
        names = self._list_module_names()
        if requested in names:
            return requested
        candidates = [n for n in names if requested in n]
        if candidates:
            chosen = max(candidates, key=lambda n: n.count("."))
            logger.warning(
                "Exact target layer '%s' not found; falling back to '%s'.",
                requested,
                chosen,
            )
            return chosen
        encoder_layers = sorted(
            n for n in names if n.startswith("encoder.layers")
        )
        if encoder_layers:
            logger.warning(
                "Target layer '%s' missing; using deepest encoder layer '%s'.",
                requested,
                encoder_layers[-1],
            )
            return encoder_layers[-1]
        raise LookupError(
            f"Could not resolve target_layer='{requested}'. Known modules: "
            + ", ".join(names[:50])
            + ("..." if len(names) > 50 else "")
        )

    def _get_submodule(self, dotted_name: str) -> nn.Module:
        module: nn.Module = self.model
        for part in dotted_name.split("."):
            module = getattr(module, part)
        return module

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_pool(tensor: torch.Tensor) -> torch.Tensor:
        """Pool a feature map of arbitrary spatial dim into ``(B, C)``.

        Uses global max pooling so a small focal lesion in any corner of the
        liver volume is not diluted by the surrounding healthy tissue average.
        """
        if tensor.dim() <= 2:
            return tensor
        return tensor.amax(dim=tuple(range(2, tensor.dim())))

    @torch.no_grad()
    def extract_activations(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> tuple[np.ndarray, list[str]]:
        """Run inference once and return per-patient activations.

        Returns
        -------
        activations:
            ``(N, D)`` array (global-max-pooled over spatial dims).
        patient_ids:
            ordered patient IDs aligned with ``activations`` rows.
        """
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            captured.append(self._spatial_pool(tensor.detach()).cpu())

        handle = self.target_layer.register_forward_hook(_hook)
        all_acts: list[np.ndarray] = []
        all_pids: list[str] = []
        try:
            for batch in loader:
                image = batch["image"].float().to(self.device)
                _ = self.model(image)
                act = captured.pop(0)
                all_acts.append(act.numpy())
                pids = batch["patient_id"]
                if isinstance(pids, (list, tuple)):
                    all_pids.extend(str(p) for p in pids)
                else:
                    all_pids.append(str(pids))
        finally:
            handle.remove()
        if not all_acts:
            raise RuntimeError("No activations captured.")
        return np.concatenate(all_acts, axis=0), all_pids

    # ------------------------------------------------------------------
    # CAV training
    # ------------------------------------------------------------------
    @staticmethod
    def train_cav(
        positive_acts: np.ndarray,
        negative_acts: np.ndarray,
        random_state: int = 0,
        c: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """Train a linear classifier; return ``(unit_cav, accuracy)``."""
        if positive_acts.ndim != 2 or negative_acts.ndim != 2:
            raise ValueError("activations must be 2D arrays (N, D).")
        x = np.concatenate([positive_acts, negative_acts], axis=0)
        y = np.concatenate(
            [
                np.ones(len(positive_acts), dtype=np.int64),
                np.zeros(len(negative_acts), dtype=np.int64),
            ]
        )
        clf = LogisticRegression(
            C=float(c),
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
        clf.fit(x, y)
        accuracy = float(clf.score(x, y))
        cav = clf.coef_[0]
        norm = np.linalg.norm(cav)
        if norm < 1e-12:
            raise ValueError("Degenerate CAV (zero norm); concept too noisy.")
        return cav / norm, accuracy

    def build_cavs(
        self,
        activations: np.ndarray,
        pid_to_index: dict[str, int],
        concepts: ConceptSets,
    ) -> dict[str, CAVRecord]:
        """Train one CAV per concept, dropping concepts with empty arms."""
        cavs: dict[str, CAVRecord] = {}
        for concept, (pos_pids, neg_pids) in concepts.items():
            pos_idx = [pid_to_index[p] for p in pos_pids if p in pid_to_index]
            neg_idx = [pid_to_index[p] for p in neg_pids if p in pid_to_index]
            if len(pos_idx) < 2 or len(neg_idx) < 2:
                logger.warning(
                    "Skipping CAV '%s' (pos=%d, neg=%d).",
                    concept,
                    len(pos_idx),
                    len(neg_idx),
                )
                continue
            try:
                vec, acc = self.train_cav(
                    activations[pos_idx],
                    activations[neg_idx],
                    c=self.cav_c,
                )
            except ValueError as exc:
                logger.warning("CAV training failed for '%s': %s", concept, exc)
                continue
            cavs[concept] = CAVRecord(
                concept=concept,
                vector=vec,
                train_accuracy=acc,
                n_positive=len(pos_idx),
                n_negative=len(neg_idx),
            )
        return cavs

    # ------------------------------------------------------------------
    # Directional derivatives + scores
    # ------------------------------------------------------------------
    def _forward_with_grad_capture(
        self,
        image: torch.Tensor,
        radiomics_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass capturing the pooled, grad-enabled activation.

        When ``radiomics_features`` is provided the joint-embedding model
        (SwinViT + Cross-Attention fusion) receives both modalities, so the
        gradient is computed at the point where both tracks meet.
        """
        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            captured.append(tensor)

        handle = self.target_layer.register_forward_hook(_hook)
        try:
            if radiomics_features is not None:
                logits, _ = self.model(image, radiomics_features)
            else:
                logits, _ = self.model(image)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Hook did not capture any activation.")
        activation = captured[0]
        pooled = self._spatial_pool(activation)
        return logits, pooled

    def directional_derivative(
        self,
        image: torch.Tensor,
        cav: np.ndarray,
        radiomics_features: torch.Tensor | None = None,
    ) -> float:
        """Compute ``<grad_h logit_target, cav>`` for a single sample.

        Parameters
        ----------
        image:
            Single CT volume tensor ``(1, C, D, H, W)`` or ``(C, D, H, W)``.
        cav:
            Unit CAV vector ``(D,)`` for the concept to probe.
        radiomics_features:
            Optional radiomics feature vector ``(1, F)``; required when the
            model is a joint-embedding fusion model (SwinViT + Cross-Attention).
        """
        image = image.to(self.device)
        if radiomics_features is not None:
            radiomics_features = radiomics_features.to(self.device)
        image.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        logits, pooled = self._forward_with_grad_capture(image, radiomics_features)
        target_logit = logits[:, self.target_class].sum()
        grads = torch.autograd.grad(
            outputs=target_logit,
            inputs=pooled,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        if grads is None:
            raise RuntimeError(
                "No gradient flowed through the hooked layer; "
                "verify target_layer is on the forward path."
            )
        cav_t = torch.from_numpy(cav.astype(np.float32)).to(self.device)
        grad_vec = grads.mean(dim=0).flatten()
        return float(torch.dot(grad_vec, cav_t).item())

    def compute_tcav_score(
        self,
        dataset: Dataset,
        cav: np.ndarray,
        max_samples: int | None = None,
    ) -> float:
        """Return the fraction of samples with positive directional derivative."""
        positive_count = 0
        total = 0
        n = len(dataset)  # type: ignore[arg-type]
        limit = n if max_samples is None else min(n, max_samples)
        for idx in range(limit):
            sample = dataset[idx]
            image = sample["image"]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            if image.dim() == 4:
                image = image.unsqueeze(0)

            # Extract radiomics features from sample when available (joint model).
            radiomics_features: torch.Tensor | None = None
            if "radiomics_features" in sample:
                rf = sample["radiomics_features"]
                if isinstance(rf, np.ndarray):
                    rf = torch.from_numpy(rf).float()
                if rf.dim() == 1:
                    rf = rf.unsqueeze(0)
                radiomics_features = rf

            try:
                deriv = self.directional_derivative(image, cav, radiomics_features)
            except RuntimeError as exc:
                logger.warning("Sample %d: %s", idx, exc)
                continue
            if deriv > 0:
                positive_count += 1
            total += 1
        if total == 0:
            return float("nan")
        return positive_count / total

    @staticmethod
    def compute_significance(
        real_score: float, random_scores: list[float]
    ) -> float:
        """Two-sided permutation p-value of ``real_score`` against the null."""
        if not random_scores:
            return float("nan")
        random_arr = np.asarray(random_scores)
        random_arr = random_arr[~np.isnan(random_arr)]
        if random_arr.size == 0:
            return float("nan")
        observed = abs(real_score - 0.5)
        nulls = np.abs(random_arr - 0.5)
        return float((nulls >= observed).mean())
