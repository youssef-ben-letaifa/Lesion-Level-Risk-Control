"""Configuration and parameters for the KiTS23 lesion-level certification study.

Cites:
    PDF §1.4 (Key terms and definitions)
    PDF §4.3 (Splits: Train 60%, Calibration 20%, Test 20%)
    PDF §5.2 (nnU-Net model pool and architecture)
    PDF §5.3 (Prediction families F0–F5)
    PDF §5.4 (Certification layer: split-conformal, CRC, LTT)
    PDF §6.1 (Phase-0 gate)
    PDF §6.2 (Compute, epochs, and timeline)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml


@dataclass
class PathConfig:
    """Directory paths for local inspection and Colab execution (PDF §6.2)."""
    drive_root: Path = Path("/content/drive/MyDrive/KiTS23_Cert")
    data_dir: Path = Path("./dataset")
    dataset_zip: Path = Path("./dataset and kits23.zip")
    work_dir: Path = Path("./work")
    results_dir: Path = Path("./results")
    outputs_dir: Path = Path("./outputs")
    logs_dir: Path = Path("./logs")
    
    # nnU-Net v2 specific environment paths
    nnunet_raw: Path = Path("./work/nnUNet_raw")
    nnunet_preprocessed: Path = Path("./work/nnUNet_preprocessed")
    nnunet_results: Path = Path("./results/nnUNet_results")


@dataclass
class SplitConfig:
    """Random patient partition settings (PDF §4.3)."""
    train_ratio: float = 0.60
    cal_ratio: float = 0.20
    test_ratio: float = 0.20
    random_seed: int = 20261026
    max_cases: Optional[int] = 200  # Exactly 200 cases in the study as requested
    dataset_fraction: Optional[float] = None  # Optional fraction override
    # Balance tolerance bounds for redraw checks
    max_cohort_imbalance: float = 0.08
    max_size_imbalance: float = 0.08
    max_thickness_imbalance: float = 0.08


@dataclass
class ModelMember:
    """Specification of an ensemble pool member (PDF §5.2)."""
    name: str
    preset: str  # 'ResEnc_M', 'original_3d_fullres', 'ResEnc_L'
    seed: int
    trainer_name: str
    fold: str = "all"


@dataclass
class ModelPoolConfig:
    """Model training configuration (PDF §5.2, §6.2)."""
    dataset_name: str = "Dataset501_KiTS23_Train"
    dataset_id: int = 501
    configuration: str = "3d_fullres"
    num_epochs: int = 100  # User constraint: all training epochs are in the number of 100
    batch_size: Optional[int] = None
    members: List[ModelMember] = field(default_factory=list)


@dataclass
class GridConfig:
    """Grid definitions for prediction families (PDF §5.3)."""
    # Threshold grid for probability families: T = 1 - lambda
    # log-spaced thresholds from 0.5 down to 1e-6
    # lambda values: lambda = 1 - T, so lambda increases from 0.5 to 1.0 - 1e-6
    num_lambda_points: int = 60
    min_threshold: float = 1e-6
    max_threshold: float = 0.5
    
    # Dilation family (F4) radius grid in mm
    min_radius_mm: float = 0.0
    max_radius_mm: float = 20.0
    num_radius_points: int = 41
    
    # F5 lesion-adaptive component seed threshold
    tau_min: float = 0.35
    
    # F3 disagreement scale beta
    beta_f3: float = 1.0

    def get_threshold_grid(self) -> np.ndarray:
        """Returns score thresholds T in descending order (permissive to strict)."""
        # Log-spaced thresholds from max_threshold (0.5) down to min_threshold (1e-6)
        return np.geomspace(self.max_threshold, self.min_threshold, self.num_lambda_points)

    def get_lambda_grid(self) -> np.ndarray:
        """Returns lambda values in ascending order (strict to permissive).
        
        Since flag condition is score >= 1 - lambda (score >= T),
        lambda = 1.0 - T. Larger lambda flags more voxels.
        """
        thresholds = self.get_threshold_grid()
        lambdas = 1.0 - thresholds
        return np.sort(lambdas)

    def get_radius_grid(self) -> np.ndarray:
        """Returns radius in mm for F4 dilation family."""
        return np.linspace(self.min_radius_mm, self.max_radius_mm, self.num_radius_points)


@dataclass
class CertifyConfig:
    """Statistical risk control parameters (PDF §5.4)."""
    alpha_levels: List[float] = field(default_factory=lambda: [0.05, 0.10])
    gamma_primary: float = 0.50  # Primary lesion overlap fraction
    gamma_sensitivity: List[float] = field(default_factory=lambda: [0.10, 0.90])
    fp_budget: float = 2.0  # Max allowable mean false-positive lesions per patient
    delta_ltt: float = 0.10  # Learn-then-test error budget
    num_resplits: int = 1000  # Calibration U Test re-split evaluation count
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class StudyConfig:
    """Top-level master configuration (PDF §5.1, §6.1)."""
    tier: str = "TIER1"  # 'TIER1', 'SMOKE', 'FULL'
    paths: PathConfig = field(default_factory=PathConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model_pool: ModelPoolConfig = field(default_factory=ModelPoolConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    cert: CertifyConfig = field(default_factory=CertifyConfig)
    
    # Phase 0 gate status
    frozen_config_file: Path = Path("./frozen_config.json")
    is_frozen: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "StudyConfig":
        """Loads configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        cfg = cls()
        if "tier" in data:
            cfg.tier = data["tier"].upper()
        
        if "paths" in data:
            root = Path(__file__).resolve().parent.parent
            for k, v in data["paths"].items():
                if hasattr(cfg.paths, k):
                    p = Path(v)
                    # If /content path does not exist on local system, resolve relative to project root
                    if not p.exists() and str(p).startswith(("/content", "\\content")):
                        rel_parts = [part for part in p.parts if part not in ("/", "\\", "content", "drive", "MyDrive", "KiTS23_Cert")]
                        fallback = root.joinpath(*rel_parts)
                        if fallback.exists() or k in ("work_dir", "results_dir", "outputs_dir", "logs_dir", "nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
                            p = fallback
                    setattr(cfg.paths, k, p)
        
        if "splits" in data:
            for k, v in data["splits"].items():
                if hasattr(cfg.splits, k):
                    setattr(cfg.splits, k, v)
                    
        if "model_pool" in data:
            mp = data["model_pool"]
            if "dataset_name" in mp:
                cfg.model_pool.dataset_name = mp["dataset_name"]
            if "dataset_id" in mp:
                cfg.model_pool.dataset_id = mp["dataset_id"]
            if "num_epochs" in mp:
                cfg.model_pool.num_epochs = mp["num_epochs"]
            if "configuration" in mp:
                cfg.model_pool.configuration = mp["configuration"]
            if "members" in mp:
                cfg.model_pool.members = [
                    ModelMember(**m) for m in mp["members"]
                ]
                
        if "grid" in data:
            for k, v in data["grid"].items():
                if hasattr(cfg.grid, k):
                    setattr(cfg.grid, k, v)
                    
        if "cert" in data:
            for k, v in data["cert"].items():
                if hasattr(cfg.cert, k):
                    setattr(cfg.cert, k, v)
                    
        return cfg
