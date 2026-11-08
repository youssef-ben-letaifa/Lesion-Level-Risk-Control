"""Training orchestration, idempotence checks, and resume logic for nnU-Net v2.

Cites:
    PDF §5.2 (Model pool training: 5x ResEnc-M, 3x original, 2x ResEnc-L on fold 'all')
    PDF §6.1 (Phase-0 5-fold CV inside Train split with --npz)
    PDF §6.2 (Compute, timeline, and resume controls)
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from renal_cert.config import ModelMember


class TrainingRunner:
    """Manages training execution with per-member marker files and resume flags."""

    def __init__(self, results_dir: Path, drive_results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self.drive_results_dir = Path(drive_results_dir) if drive_results_dir else None

    def get_member_output_dir(self, member: ModelMember, dataset_name: str, configuration: str) -> Path:
        """Determines expected output directory for a model member."""
        return self.results_dir / dataset_name / f"{member.trainer_name}__{member.preset}" / f"fold_{member.fold}"

    def is_member_finished(self, member: ModelMember, dataset_name: str, configuration: str) -> bool:
        """Checks if training completed marker exists locally or on Drive."""
        out_dir = self.get_member_output_dir(member, dataset_name, configuration)
        local_marker = out_dir / "training_finished.marker"
        if local_marker.exists():
            return True
        if self.drive_results_dir:
            drive_out = self.drive_results_dir / dataset_name / f"{member.trainer_name}__{member.preset}" / f"fold_{member.fold}"
            if (drive_out / "training_finished.marker").exists() or (drive_out / "checkpoint_final.pth").exists():
                return True
        return False

    def build_train_command(
        self,
        dataset_id: int,
        configuration: str,
        member: ModelMember,
        plans_name: Optional[str] = None
    ) -> List[str]:
        """Constructs command line arguments for nnUNetv2_train."""
        plans = plans_name if plans_name else f"nnUNetPlans"
        cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(member.fold),
            "-tr", member.trainer_name,
            "-p", plans,
            "--c"  # Automatically continue from checkpoint_latest.pth if interrupted
        ]
        return cmd
