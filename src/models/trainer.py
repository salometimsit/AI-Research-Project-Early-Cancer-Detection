"""Training loop, early stopping, checkpointing, and metric history.

Pipeline stage: drives Phase 4 (SwinViT) and Phase 5 (fusion) training.

Inputs
------
* PyTorch ``DataLoader`` instances for training and validation.
* The config dict for ``training.*`` (epochs, lr, weight decay, focal
  loss alpha/gamma, early stopping patience, weighted sampler).
* ``model.early_stopping`` — ``min_delta`` (val-loss criterion) and
  ``min_delta_auc`` (``-val_auc`` criterion when AUC is finite).
* ``model.trainer`` — ``grad_clip_max_norm`` (``null`` disables clipping)
  and ``cuda_empty_cache_between_train_val``.

Outputs
-------
* ``models/saved/best_*.pth`` -- best checkpoint with ``model_state``
  and the original ``cfg`` snapshot.
* ``results/training_history_*.json`` -- per-epoch training/val loss
  and AUC.

Side effects
------------
* Logs every epoch's metrics at INFO level.
* ``EarlyStopping`` halts when the monitored score (``-val_auc`` when AUC
  is defined, else ``val_loss``) does not improve for ``patience`` epochs.

Failure modes
-------------
* Single-class validation set -> AUC reported as ``nan`` (best-checkpoint
  selection still functional via val loss).
* Out-of-memory during a forward pass -> reduce ``training.batch_size``
  in the config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when a *lower-is-better* score stops improving for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta_loss: float, min_delta_auc: float) -> None:
        self.patience = int(patience)
        self.min_delta_loss = float(min_delta_loss)
        self.min_delta_auc = float(min_delta_auc)
        self.best: float = float("inf")
        self.bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, score: float, *, uses_neg_auc: bool) -> bool:
        """Update the patience counter with this epoch's monitored score.

        Args:
            score: ``-val_auc`` when ``uses_neg_auc`` is True, else ``val_loss``.
            uses_neg_auc: If True, ``min_delta_auc`` applies to ``-val_auc``;
                otherwise ``min_delta_loss`` applies to ``val_loss``.

        Returns:
            True when training should stop (i.e. ``bad_epochs >= patience``).
        """
        min_delta = self.min_delta_auc if uses_neg_auc else self.min_delta_loss
        if score < self.best - min_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
        return self.should_stop


class Trainer:
    """Generic SwinViT / fusion trainer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        cfg: dict[str, Any],
        forward_fn: Callable[[nn.Module, dict[str, Any]], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg
        self.forward_fn = forward_fn or _default_forward
        train_cfg = cfg["training"]
        model_cfg = cfg["model"]
        early_cfg = model_cfg["early_stopping"]
        trainer_cfg = model_cfg["trainer"]
        self.history: list[dict[str, float]] = []
        min_delta_auc = float(early_cfg.get("min_delta_auc", early_cfg["min_delta"]))
        self.early_stopping = EarlyStopping(
            patience=int(train_cfg["early_stopping_patience"]),
            min_delta_loss=float(early_cfg["min_delta"]),
            min_delta_auc=min_delta_auc,
        )
        raw_clip = trainer_cfg.get("grad_clip_max_norm")
        self._grad_clip_max_norm: float | None = (
            float(raw_clip) if raw_clip is not None else None
        )
        self._cuda_empty_cache_between_train_val: bool = bool(
            trainer_cfg["cuda_empty_cache_between_train_val"]
        )
        self.best_val_auc: float = -float("inf")
        self.best_val_loss: float = float("inf")
        self._early_stop_uses_neg_auc: bool | None = None

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training pass; return per-epoch loss / AUC dict."""
        self.model.train()
        losses: list[float] = []
        preds: list[float] = []
        targets: list[int] = []
        for batch in loader:
            self.optimizer.zero_grad()
            logits, y = self.forward_fn(self.model, _to_device(batch, self.device))
            loss = self.loss_fn(logits, y)
            loss.backward()
            if self._grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self._grad_clip_max_norm,
                )
            self.optimizer.step()
            losses.append(float(loss.item()))
            probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
            preds.extend(probs.tolist())
            targets.extend(y.detach().cpu().numpy().tolist())
        return _compute_metrics(losses, preds, targets, prefix="train")

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on ``loader``; return ``val_loss`` / ``val_auc``."""
        self.model.eval()
        losses: list[float] = []
        preds: list[float] = []
        targets: list[int] = []
        for batch in loader:
            logits, y = self.forward_fn(self.model, _to_device(batch, self.device))
            loss = self.loss_fn(logits, y)
            losses.append(float(loss.item()))
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend(probs.tolist())
            targets.extend(y.cpu().numpy().tolist())
        return _compute_metrics(losses, preds, targets, prefix="val")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Path | str,
        history_path: Path | str | None = None,
    ) -> dict[str, float]:
        """Run the training loop with early stopping and checkpointing.

        Best-checkpoint selection is nan-safe:

        * Primary criterion is ``val_auc`` (higher is better).
        * If ``val_auc`` is ``nan`` (e.g. single-class validation
          fold) the criterion silently falls back to ``val_loss``
          (lower is better) so a checkpoint is *always* written
          before early stopping fires.
        * ``best_val_loss`` is always taken from the same epoch as the
          best ``val_auc`` when the latter improves (no historical ``min``
          mixing epochs).
        * Early stopping monitors ``-val_auc`` when AUC is finite, else
          ``val_loss``, using the corresponding ``min_delta`` from config.

        Args:
            train_loader: DataLoader over the training fold.
            val_loader: DataLoader over the validation fold.
            checkpoint_path: Where to write the best checkpoint.
            history_path: Optional JSON path; per-epoch metrics are
                streamed there so partial runs leave a trail.

        Returns:
            Dict with ``best_val_auc``, ``best_val_loss`` and
            ``epochs_run``.
        """
        epochs = int(self.cfg["training"]["epochs"])
        logger.debug("Trainer.fit starting for %d max epochs.", epochs)
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            if (
                self._cuda_empty_cache_between_train_val
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            val_metrics = self.validate(val_loader)
            entry = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(entry)
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f val_auc=%.3f",
                epoch,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics["val_auc"],
            )

            val_auc = val_metrics["val_auc"]
            val_loss = val_metrics["val_loss"]
            uses_neg_auc = not math.isnan(val_auc)
            if (
                self._early_stop_uses_neg_auc is not None
                and self._early_stop_uses_neg_auc != uses_neg_auc
            ):
                logger.warning(
                    "val_auc finiteness changed mid-run; resetting early-stopping baseline."
                )
                self.early_stopping.best = float("inf")
                self.early_stopping.bad_epochs = 0
            self._early_stop_uses_neg_auc = uses_neg_auc

            improved = False
            if not math.isnan(val_auc) and val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                improved = True
            elif math.isnan(val_auc) and val_loss < self.best_val_loss:
                # AUC is undefined (single-class fold) — fall back to val_loss.
                self.best_val_loss = val_loss
                improved = True
                logger.warning(
                    "val_auc=nan at epoch %d; falling back to val_loss for "
                    "best-checkpoint selection.",
                    epoch,
                )
            if improved:
                self.save_checkpoint(checkpoint_path)

            if history_path is not None:
                self.save_training_history(history_path)
            current_score = -val_auc if uses_neg_auc else val_loss
            if self.early_stopping.step(current_score, uses_neg_auc=uses_neg_auc):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break
        logger.debug(
            "Trainer.fit finished: epochs_run=%d best_val_auc=%s best_val_loss=%s",
            len(self.history),
            self.best_val_auc,
            self.best_val_loss,
        )
        return {
            "best_val_auc": self.best_val_auc,
            "best_val_loss": self.best_val_loss,
            "epochs_run": len(self.history),
        }

    def save_checkpoint(self, path: Path | str) -> None:
        """Write the current model state and config to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": self.cfg},
            path,
        )
        logger.info("Saved checkpoint -> %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore weights from a previously :meth:`save_checkpoint`-d file."""
        path = Path(path)
        bundle = torch.load(path, map_location=self.device)
        self.model.load_state_dict(bundle["model_state"])
        logger.info("Loaded checkpoint <- %s", path)

    def save_training_history(self, path: Path | str) -> None:
        """Persist the per-epoch metrics list as pretty-printed JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        logger.info("Saved training history -> %s", path.resolve())


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _default_forward(model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Default SwinViT forward: ``model(image) -> (logits, _)``."""
    image = batch["image"].float()
    label = batch["label"].long()
    out = model(image)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    return logits, label


def _compute_metrics(
    losses: list[float],
    preds: list[float],
    targets: list[int],
    prefix: str,
) -> dict[str, float]:
    loss = float(np.mean(losses)) if losses else float("nan")
    pred_arr = np.asarray(preds, dtype=np.float64)
    if (
        not losses
        or len(set(targets)) < 2
        or pred_arr.size == 0
        or not np.isfinite(pred_arr).all()
    ):
        return {f"{prefix}_loss": loss, f"{prefix}_auc": float("nan")}
    try:
        auc = float(roc_auc_score(targets, preds))
    except (ValueError, RuntimeError):
        auc = float("nan")
    return {f"{prefix}_loss": loss, f"{prefix}_auc": auc}
