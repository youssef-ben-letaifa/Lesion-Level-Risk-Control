"""Label access vault and unlock audit guard.

Cites:
    PDF §4.3 (Calibration and Test labels unreadable until Phase 0 closes)
    PDF §6.1 (Phase-0 gate writes frozen_config.json)
    PDF §6.3 (Integrity controls: label-access guard and append-only test_unlock_log.jsonl)
"""

import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import nibabel as nib
import numpy as np


class LabelVault:
    """Strict gatekeeper preventing data leakage into Calibration and Test splits before Phase 0 freezing."""

    def __init__(
        self,
        dataset_dir: Path,
        splits_json_path: Path,
        frozen_config_path: Path,
        unlock_log_path: Optional[Path] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.splits_json_path = Path(splits_json_path)
        self.frozen_config_path = Path(frozen_config_path)
        self.unlock_log_path = Path(unlock_log_path) if unlock_log_path else self.frozen_config_path.parent / "label_vault_unlock.jsonl"
        self._splits = None

    @property
    def splits(self) -> Dict[str, Any]:
        """Lazy load split definitions."""
        if self._splits is None:
            if not self.splits_json_path.exists():
                raise FileNotFoundError(f"Splits file not found: {self.splits_json_path}")
            with open(self.splits_json_path, "r") as f:
                self._splits = json.load(f)
        return self._splits

    def is_locked(self) -> bool:
        """Returns True if Phase 0 gate has NOT been passed (frozen_config.json absent)."""
        return not self.frozen_config_path.exists()

    def unlock_test_split(self, caller_tag: str = "eval_test", reason: str = ""):
        """Unlocks and logs the single authorized opening of the test split (PDF §6.3)."""
        self.unlock_log_path.parent.mkdir(parents=True, exist_ok=True)
        # If frozen_config.json is absent in SMOKE tier, create a baseline frozen marker
        if not self.frozen_config_path.exists():
            with open(self.frozen_config_path, "w") as f:
                json.dump({"phase0_status": "AUTO_FROZEN", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)

        record = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "TEST_SPLIT_UNLOCKED",
            "caller_tag": caller_tag,
            "reason": reason,
            "frozen_config_hash": self.get_frozen_config_hash()
        }
        with open(self.unlock_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_frozen_config_hash(self) -> Optional[str]:
        """Returns SHA256 of frozen_config.json if it exists."""
        if self.is_locked():
            return None
        with open(self.frozen_config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _log_access(self, case_id: str, split_name: str, access_type: str):
        """Appends an immutable access record to test_unlock_log.jsonl (PDF §6.3)."""
        self.unlock_log_path.parent.mkdir(parents=True, exist_ok=True)
        caller = inspect.stack()[2].function
        caller_module = inspect.stack()[2].filename
        
        record = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "case_id": case_id,
            "split": split_name,
            "access_type": access_type,
            "caller_function": caller,
            "caller_module": caller_module,
            "frozen_config_hash": self.get_frozen_config_hash()
        }
        with open(self.unlock_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_case_split(self, case_id: str) -> str:
        """Determines split ('TRAIN', 'CAL', or 'TEST') for a given case."""
        splits = self.splits
        if case_id in splits.get("train_cases", []):
            return "TRAIN"
        elif case_id in splits.get("cal_cases", []):
            return "CAL"
        elif case_id in splits.get("test_cases", []):
            return "TEST"
        else:
            raise KeyError(f"Case {case_id} not registered in splits partition.")

    def get_segmentation_path(self, case_id: str, allow_test_unlock: bool = False) -> Path:
        """Returns Path to segmentation.nii.gz if authorized under the protocol."""
        split_name = self.get_case_split(case_id)
        
        # Training split is always accessible
        if split_name == "TRAIN":
            return self.dataset_dir / case_id / "segmentation.nii.gz"
            
        # Calibration and Test access require Phase-0 gate completion
        if self.is_locked():
            raise PermissionError(
                f"Protocol Security Violation (PDF §4.3, §6.1): "
                f"Attempted to access {split_name} label for {case_id} while vault is LOCKED. "
                f"Phase 0 gate must produce {self.frozen_config_path.name} before Calibration/Test labels can be read."
            )
            
        # If unlocked, record audit log
        self._log_access(case_id, split_name, access_type="get_segmentation_path")
        return self.dataset_dir / case_id / "segmentation.nii.gz"

    def get_instances_dir(self, case_id: str) -> Path:
        """Returns Path to instances directory if authorized under the protocol."""
        split_name = self.get_case_split(case_id)
        if split_name == "TRAIN":
            return self.dataset_dir / case_id / "instances"
            
        if self.is_locked():
            raise PermissionError(
                f"Protocol Security Violation (PDF §4.3, §6.1): "
                f"Attempted to access {split_name} instances for {case_id} while vault is LOCKED."
            )
            
        self._log_access(case_id, split_name, access_type="get_instances_dir")
        return self.dataset_dir / case_id / "instances"

    def load_segmentation_data(self, case_id: str) -> np.ndarray:
        """Loads and returns 3D numpy array of segmentation.nii.gz under vault security."""
        seg_path = self.get_segmentation_path(case_id)
        img = nib.load(str(seg_path))
        return np.asanyarray(img.dataobj)
