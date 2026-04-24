# COMP9517 Group Project Code README

## 1. Repository Overview
This folder contains our code for wheat crop segmentation on the EWS dataset, covering:
- Traditional image segmentation methods
- Machine learning methods (Random Forest / XGBoost)
- Deep learning methods (U-Net / DeepLabV3)

Main files:
- `ML.ipynb`: machine learning pipeline (feature engineering, model comparison, thresholding, robustness, statistics)
- `random_forest_v1.ipynb`: Random Forest baseline notebook
- `deeplearning.ipynb`: deep learning pipeline (U-Net and DeepLabV3, ablations, robustness)
- `test_z5568377.ipynb`: notebook version of traditional-method experiments
- `test_z5568377_run.py`: script version of traditional-method experiments (batch runnable)

## 2. Environment
Recommended:
- Python 3.10+
- Jupyter Notebook / JupyterLab

Core libraries used in this project:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `opencv-python` (`cv2`), `scikit-image`
- `scikit-learn`, `xgboost`, `scipy`
- `torch`, `torchvision`, `tqdm`, `Pillow`

Example installation:
```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-image scikit-learn xgboost scipy torch torchvision tqdm pillow
```

## 3. Dataset Preparation
Download and unzip `EWS-Dataset.zip`.

Expected structure:
```text
EWS-Dataset/
  train/
  validation/
  test/
```
Each split should contain paired files:
- image: `xxx.png`
- mask: `xxx_mask.png`

## 4. How to Run
### 4.1 Traditional methods (script)
Run:
```bash
python test_z5568377_run.py
```

Before running, update these paths in `test_z5568377_run.py` if needed:
- `DATASET_ROOT`
- `OUTPUT_ROOT`

### 4.2 Traditional methods (notebook)
Open and run `test_z5568377.ipynb` from top to bottom.
Also check/update:
- `DATASET_ROOT`
- `OUTPUT_ROOT`

### 4.3 Machine learning pipeline
Open and run `ML.ipynb` from top to bottom.
This notebook uses a relative dataset path (`./EWS-Dataset`) by default.

### 4.4 Random Forest baseline
Open and run `random_forest_v1.ipynb` from top to bottom.
This notebook also expects `./EWS-Dataset`.

### 4.5 Deep learning pipeline
Open and run `deeplearning.ipynb` from top to bottom.

Important:
- The notebook currently contains an absolute path for `DATA_ROOT`.
- Please change `DATA_ROOT` to your local dataset path before training/evaluation.

## 5. Outputs
Typical outputs include:
- quantitative tables (`.csv` / `.txt`)
- prediction masks and overlay images (`.png`)
- robustness plots
- model weights (`.pth`, generated during deep learning runs)

Output locations depend on the notebook/script settings, commonly under:
- `./results/` (for notebooks)
- `OUTPUT_ROOT` (for `test_z5568377_run.py` and `test_z5568377.ipynb`)

## 6. External Code / Libraries Declaration
This submission mainly contains our own implementation code.

Third-party dependencies are standard open-source Python libraries (installed via `pip`), including but not limited to:
- PyTorch / Torchvision
- Scikit-learn
- Scikit-image
- OpenCV
- XGBoost
- NumPy / Pandas / Matplotlib / SciPy

Pretrained weights usage:
- In `deeplearning.ipynb`, DeepLabV3 may use official ImageNet-pretrained backbone weights from Torchvision.
- If pretrained weight download fails (offline environment), the code falls back to random initialization.

No third-party project source code is directly copied into this folder.

## 7. Notes for Submission Packaging
When preparing the final ZIP upload:
- Keep source code and notebooks.
- Do not include large generated artifacts if size is an issue (e.g., trained models, dataset images, result images), in line with project specification constraints.
- Ensure paths are portable (prefer relative paths) for marker reproducibility.
