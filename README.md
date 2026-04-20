# HCC Early Prediction from CT Scans

Predict future Hepatocellular Carcinoma (HCC) development from pre-diagnosis CT scans using a dual-track approach combining quantitative Radiomics features with deep learning.

## Approach

1. **Data Preparation** — DICOM → NIfTI conversion, HU windowing, liver segmentation via TotalSegmentator
2. **Registration & Labeling** — Deformable co-registration of Before/After scans; project tumor masks onto "Before" scans to generate ground truth
3. **Radiomics Track** — PyRadiomics feature extraction (histogram, GLCM, GLRLM, GLSZM, shape) with VaRFS stability-based feature selection
4. **Deep Learning Track** — 3D ResNet-18 (MedicalNet pre-trained) on liver patches
5. **Combined Classifier** — Fused radiomics + deep features for final cancer-risk prediction

## Project Structure

```
proj/
  README.md
  requirements.txt
  notebooks/               # Step-by-step exploration notebooks
    01_data_exploration.ipynb
    02_preprocessing.ipynb
    03_registration.ipynb
    04_radiomics.ipynb
    05_model_training.ipynb
  src/
    data/
      dicom_loader.py      # DICOM reading and NIfTI conversion
      preprocessing.py     # HU windowing, normalization
      liver_segmentation.py# TotalSegmentator wrapper
      registration.py      # Multi-stage co-registration
      patch_extraction.py  # 3D patch extraction
    features/
      radiomics_extractor.py  # PyRadiomics pipeline
      varfs_selection.py      # VaRFS feature selection
    models/
      resnet3d.py          # 3D ResNet with MedicalNet weights
      trainer.py           # Training loop, metrics, Focal Loss
    utils/
      visualization.py     # Slice viewer, mask overlay
      config.py            # Config loading utilities
  configs/
    default.yaml           # All hyperparameters
  scripts/
    run_preprocessing.py
    run_training.py
```

## Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# TotalSegmentator downloads model weights on first run (~1.5 GB)
```

## Usage

### Preprocessing

```bash
python scripts/run_preprocessing.py --config configs/default.yaml
```

### Training

```bash
python scripts/run_training.py --config configs/default.yaml
```

### Notebooks

Notebooks in `notebooks/` walk through each pipeline stage interactively. Start with `01_data_exploration.ipynb`.

## Configuration

All hyperparameters live in `configs/default.yaml`. Override any value via CLI flags or by creating a custom YAML file.
