If you find any part of this implementation helpful and please cite the corresponding paper: https://iopscience.iop.org/article/10.1088/1361-6560/adf9b5

# Enhancing B-mode-based Breast Cancer Diagnosis via Cross-attention Fusion of H-scan and Nakagami Imaging

This repository contains a PyTorch/Kaggle notebook implementation of a tri-modal breast ultrasound classification pipeline using:

- B-mode or Machine B-mode ultrasound images
- H-scan images
- Nakagami images
- Segmentation-mask-guided CAM supervision
- Quantitative ultrasound (QUS)-guided auxiliary losses
- Cross-attention-based tri-modal feature fusion

The pipeline is designed for binary breast lesion classification: **benign vs malignant**.

---

## Overview

The model uses three parallel PTL-CNN branches for B-mode/Machine B-mode, H-scan, and Nakagami modalities. Each branch extracts modality-specific deep features. These features are then fused using self-attention and cross-modal attention, followed by an ADCE-based fusion block and a final classification head.

The training pipeline includes:

1. **Cross-entropy classification loss**
2. **Multi-CAM loss** using lesion segmentation masks
3. **F-QUS loss** to regress handcrafted QUS descriptors from B-mode features
4. **P-QUS loss** using a QUS-SVM teacher probability and a differentiable prediction head

The notebook is written in a flexible way so that users can switch between `B-mode` and `Machine_B-mode` input branches without modifying the rest of the model pipeline.

---

## Expected Dataset Format

The notebook expects the dataset to be organized as follows:

```text
Binary/
├── Machine_B-mode/
│   ├── Fold1/
│   │   ├── Benign/
│   │   └── Malignant/
│   ├── Fold2/
│   └── ...
├── B-mode/
│   ├── Fold1/
│   │   ├── Benign/
│   │   └── Malignant/
│   ├── Fold2/
│   └── ...
├── H-scan/
│   ├── Fold1/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── ...
├── Nakagami/
│   ├── Fold1/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── ...
└── Mask/
    ├── Fold1/
    │   ├── Benign/
    │   └── Malignant/
    └── ...
```

The code supports common image formats such as `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

---

## Dataset Directory and Modality Selection

To run the notebook on a new dataset, users mainly need to change the dataset directory path.

In the uploaded notebook:

- Change the dataset root directory in **Cell 10, line 4**
- Choose whether to use `B-mode` or `Machine_B-mode` in **Cell 10, line 14**

This applies to BUET-BUSD, ATL, OASBUD, or any similarly structured dataset.

No other code changes are required for switching between `B-mode` and `Machine_B-mode`, because the pipeline is built to handle this flexibly.

For the public **OASBUD** dataset, the same model can be run without additional parameter tuning or architectural modification, assuming the dataset is arranged according to the expected folder structure.

For private ATL-style datasets, dataset-specific parameter tuning and higher training epochs are recommended for stronger performance.

---

## QUS Features

The following eight handcrafted QUS features are extracted from B-mode images and lesion masks:

1. Edge diffusivity
2. Shape asymmetry factor
3. Tissue heterogeneity
4. Aspect ratio
5. Convexity
6. Solidity
7. Co-occurrence contrast
8. Lesion boundary descriptor

These features are used for the F-QUS and P-QUS auxiliary objectives.

---

## Training and Evaluation

The notebook supports 5-fold cross-validation. Each fold is used once as the validation fold while the remaining folds are used for training.
The notebook reports:

- Accuracy
- Sensitivity
- Specificity
- Balanced accuracy
- F1 score
- Confusion matrix

---

## Representative Result

A representative 5-fold cross-validation run on a private BUET-BUSD dataset achieved:

| Dataset | Modalities | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|
| BUET-BUSD Private | B-mode + H-scan + Nakagami | 93.38 ± 3.89% | 87.27 ± 13.79% | 95.74 ± 3.00% | 0.88 ± 0.08 |

Experiments were also conducted on OASBUD and ATL-style tri-modal datasets.

---

## Data Availability

This repository shares the notebook code for research and reproducibility purposes. Private datasets, segmentation masks, trained weights, QUS CSV files, and saved checkpoints are **not included** in this repository.

Users should prepare their own dataset following the expected folder structure. The OASBUD dataset used in this work is publicly available from its original source and can be utilized with this notebook after arranging it in the required folder format.

---

## Requirements

The notebook was developed and tested in a Kaggle GPU environment.

Main Python packages:

```text
torch
torchvision
numpy
pandas
opencv-python
scikit-learn
scikit-image
scipy
matplotlib
seaborn
tqdm
Pillow
joblib
```

Most of these packages are already available in Kaggle notebooks.

---

## Acknowledgement

This work was developed and tested using the free Kaggle GPU environment, especially the NVIDIA T4 GPU runtime. The availability of free GPU resources through Kaggle was very helpful for running the experiments and 5-fold cross-validation studies.

---

## License

This repository is intended for academic and research use. Please check dataset-specific licenses before using or redistributing any medical imaging data.
