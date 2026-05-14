# Medical Imaging Rules

Domain rules for DICOM/NIfTI handling. Always apply.

- Always validate NIfTI orientation and voxel spacing immediately after loading.
- Preserve the patient ID as the primary key in every output file name and
  every CSV column.
- Never modify files inside `data/raw/`. All transformations must write to
  `data/processed/`.
- Log every DICOM phase-filtering decision (accepted/rejected) per series.
- When saving NIfTI, preserve the original affine and header metadata from the
  source volume.
- Volume operations must explicitly handle both `(H, W, D)` and `(D, H, W)`
  axis orders — never assume either silently.
- All intermediate per-patient artifacts must be saved with patient ID
  traceability (file name and metadata).
