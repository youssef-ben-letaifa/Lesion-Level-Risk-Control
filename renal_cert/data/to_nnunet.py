"""Converts KiTS23 Train split to standard nnU-Net v2 dataset structure.

Cites:
    PDF §4.3 (Splits: Train split only converted into training dataset)
    PDF §5.2 (nnU-Net v2 data conversion requirements and channel setup)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class NNUnetConverter:
    """Converts Train split to Dataset501_KiTS23_Train in nnUNet_raw (PDF §5.2)."""

    def __init__(
        self,
        dataset_dir: Path,
        nnunet_raw_dir: Path,
        dataset_id: int = 501,
        dataset_name: str = "Dataset501_KiTS23_Train"
    ):
        self.dataset_dir = Path(dataset_dir)
        self.nnunet_raw_dir = Path(nnunet_raw_dir)
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.target_dir = self.nnunet_raw_dir / self.dataset_name

    def convert(self, train_cases: List[str], cal_cases: List[str], test_cases: List[str], work_dir: Path) -> Path:
        """Sets up nnUNet_raw folders and links/copies training cases."""
        images_tr = self.target_dir / "imagesTr"
        labels_tr = self.target_dir / "labelsTr"
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)

        work_dir = Path(work_dir)
        cal_img_dir = work_dir / "cal_images"
        test_img_dir = work_dir / "test_images"
        cal_img_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting {len(train_cases)} cases to {self.target_dir}...", flush=True)

        # Process Train cases (use symlinks to avoid duplicating gigabytes of NIfTI volumes)
        for cid in train_cases:
            src_img = self.dataset_dir / cid / "imaging.nii.gz"
            src_seg = self.dataset_dir / cid / "segmentation.nii.gz"
            dst_img = images_tr / f"{cid}_0000.nii.gz"
            dst_seg = labels_tr / f"{cid}.nii.gz"

            if src_img.exists() and not dst_img.exists():
                try:
                    os.symlink(src_img, dst_img)
                except (OSError, AttributeError):
                    shutil.copyfile(src_img, dst_img)

            if src_seg.exists() and not dst_seg.exists():
                try:
                    os.symlink(src_seg, dst_seg)
                except (OSError, AttributeError):
                    shutil.copyfile(src_seg, dst_seg)

        # Generate dataset.json
        dataset_json = {
            "channel_names": {
                "0": "CT"
            },
            "labels": {
                "background": 0,
                "kidney": 1,
                "tumor": 2,
                "cyst": 3
            },
            "numTraining": len(train_cases),
            "file_ending": ".nii.gz"
        }
        json_path = self.target_dir / "dataset.json"
        with open(json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"Created {json_path} with {len(train_cases)} training cases.", flush=True)
        return self.target_dir
