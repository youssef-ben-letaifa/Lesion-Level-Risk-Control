# Pretrained nnU-Net v2 Models

This directory holds trained neural network model weights for KiTS23 renal tumor segmentation.

Due to GitHub's file size limits (checkpoints are ~240 MB each), model checkpoints (`.pth`) are distributed externally rather than tracked directly in git.

## Expected Directory Layout

Place downloaded nnU-Net checkpoints in the following structure:

```
models/
└── nnUNetTrainer_KiTS23_100ep_s101__nnUNetPlans__3d_fullres/
    ├── fold_0/
    │   ├── checkpoint_best.pth
    │   └── checkpoint_final.pth
    └── fold_all/
        ├── checkpoint_best.pth
        └── checkpoint_final.pth
```

## Downloading Checkpoints

You can train new models using the provided Google Colab notebooks (`colab_notebook.ipynb` or `colabfull_notebook.ipynb`), or download pre-trained weights from the project's external storage / release page:

- **GitHub Releases / Zenodo**: [KiTS23 Model Weights Archive](https://github.com/youssef-ben-letaifa/Lesion-Level-Risk-Control/releases)
- **Interactive Interface**: The web viewer (`python run_interface.py`) works immediately with precomputed predictions in `predictions_s101/` even before downloading full raw weights.

