"""Lesion extraction, size stratification, and coverage calculation.

Cites:
    PDF §1.4 (Key terms: lesion definition, detected lesion coverage fraction gamma)
    PDF §4.2 (Tumour and cyst definitions, TNM size strata)
    PDF §4.4 (Released instances vs 26-connected components)
    PDF §5.4 (Lesion-level coverage and critical threshold calculation)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure


@dataclass
class LesionInstance:
    """Represents a single anatomical mass lesion (tumour or cyst)."""
    case_id: str
    lesion_id: str
    class_type: str  # 'tumor', 'cyst', or 'mass'
    voxel_count: int
    volume_ml: float
    equiv_diameter_cm: float
    max_extent_cm: float
    size_stratum: str  # '<=4cm', '4-7cm', '>7cm'
    voxel_coords: np.ndarray  # Shape (N, 3) integer coordinates

    def compute_coverage(self, flagged_mask: np.ndarray) -> float:
        """Computes fraction of lesion voxels flagged by F_lambda (PDF §1.4).
        
        Coverage c = |lesion ∩ F_lambda| / |lesion|.
        """
        if self.voxel_count == 0:
            return 1.0
        
        # Fast indexing with coordinate arrays
        z = self.voxel_coords[:, 0]
        y = self.voxel_coords[:, 1]
        x = self.voxel_coords[:, 2]
        flagged_hits = np.count_nonzero(flagged_mask[z, y, x])
        return float(flagged_hits) / float(self.voxel_count)

    def is_detected(self, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
        """Evaluates detection at threshold gamma (PDF §1.4)."""
        return self.compute_coverage(flagged_mask) >= gamma


class LesionExtractor:
    """Extracts ground-truth lesion instances from official annotations or 26-connected components."""

    @staticmethod
    def classify_size_stratum(diameter_cm: float) -> str:
        """Maps diameter to TNM size strata (PDF §4.2)."""
        if diameter_cm <= 4.0:
            return "<=4cm"
        elif diameter_cm <= 7.0:
            return "4-7cm"
        else:
            return ">7cm"

    @classmethod
    def extract_from_instances(
        cls,
        case_id: str,
        instances_dir: Path,
        spacing: Tuple[float, float, float],
        annotation_index: int = 1
    ) -> List[LesionInstance]:
        """Extracts lesions from released instance masks for a specific annotation repeat (PDF §4.2, §5.6)."""
        instances_dir = Path(instances_dir)
        if not instances_dir.exists():
            return []

        # Find matching instance files: e.g. tumor_instance-1_annotation-1.nii.gz
        pattern_tum = f"tumor_instance-*_annotation-{annotation_index}.nii.gz"
        pattern_cys = f"cyst_instance-*_annotation-{annotation_index}.nii.gz"
        
        tum_files = sorted(list(instances_dir.glob(pattern_tum)))
        cys_files = sorted(list(instances_dir.glob(pattern_cys)))
        all_files = [(f, "tumor") for f in tum_files] + [(f, "cyst") for f in cys_files]
        
        voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        lesions = []
        
        for fpath, class_type in all_files:
            img = nib.load(str(fpath))
            arr = np.asanyarray(img.dataobj) > 0
            count = int(np.count_nonzero(arr))
            if count == 0:
                continue
                
            coords = np.argwhere(arr)
            vol_ml = count * voxel_vol_ml
            
            # Equivalent spherical diameter: V = 4/3 * pi * r^3 => d = 2 * (3V / (4pi))^(1/3)
            # Volume in mL = cm^3, so equiv diameter is directly in cm
            equiv_diam_cm = 2.0 * ((3.0 * vol_ml) / (4.0 * np.pi)) ** (1.0 / 3.0)
            
            # Spatial bounding box max extent in cm
            min_c = coords.min(axis=0) * np.array(spacing)
            max_c = coords.max(axis=0) * np.array(spacing)
            extent_mm = np.linalg.norm(max_c - min_c)
            extent_cm = float(extent_mm / 10.0)
            
            # Lesion ID: e.g. case_00000_tumor_inst1
            inst_str = fpath.stem.split("_")[1]
            lesion_id = f"{case_id}_{class_type}_{inst_str}"
            
            lesions.append(
                LesionInstance(
                    case_id=case_id,
                    lesion_id=lesion_id,
                    class_type=class_type,
                    voxel_count=count,
                    volume_ml=round(vol_ml, 4),
                    equiv_diameter_cm=round(equiv_diam_cm, 3),
                    max_extent_cm=round(extent_cm, 3),
                    size_stratum=cls.classify_size_stratum(equiv_diam_cm),
                    voxel_coords=coords
                )
            )
        return lesions

    @classmethod
    def extract_from_segmentation_26cc(
        cls,
        case_id: str,
        seg_data: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> List[LesionInstance]:
        """Fallback lesion extraction using 26-connected components of (tumour U cyst) (PDF §1.4)."""
        s26 = generate_binary_structure(3, 3)
        mass_mask = (seg_data == 2) | (seg_data == 3)
        labeled_mask, num_features = label(mass_mask, structure=s26)
        
        voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        lesions = []
        
        for feat_id in range(1, num_features + 1):
            feat_voxels = (labeled_mask == feat_id)
            count = int(np.count_nonzero(feat_voxels))
            if count == 0:
                continue
                
            coords = np.argwhere(feat_voxels)
            vol_ml = count * voxel_vol_ml
            equiv_diam_cm = 2.0 * ((3.0 * vol_ml) / (4.0 * np.pi)) ** (1.0 / 3.0)
            
            min_c = coords.min(axis=0) * np.array(spacing)
            max_c = coords.max(axis=0) * np.array(spacing)
            extent_cm = float(np.linalg.norm(max_c - min_c) / 10.0)
            
            # Determine predominant class
            tumor_count = np.count_nonzero(seg_data[feat_voxels] == 2)
            cyst_count = np.count_nonzero(seg_data[feat_voxels] == 3)
            class_type = "tumor" if tumor_count >= cyst_count else "cyst"
            
            lesions.append(
                LesionInstance(
                    case_id=case_id,
                    lesion_id=f"{case_id}_26cc_feat{feat_id}",
                    class_type=class_type,
                    voxel_count=count,
                    volume_ml=round(vol_ml, 4),
                    equiv_diameter_cm=round(equiv_diam_cm, 3),
                    max_extent_cm=round(extent_cm, 3),
                    size_stratum=cls.classify_size_stratum(equiv_diam_cm),
                    voxel_coords=coords
                )
            )
        return lesions

    @classmethod
    def extract_case_lesions(
        cls,
        case_id: str,
        case_dir: Path,
        seg_data: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        prefer_instances: bool = True
    ) -> List[LesionInstance]:
        """Main lesion extraction dispatcher respecting protocol preference (PDF §1.4)."""
        case_dir = Path(case_dir)
        seg_path = case_dir / "segmentation.nii.gz"
        
        if spacing is None:
            hdr = nib.load(str(seg_path)).header
            zooms = hdr.get_zooms()
            spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
            
        instances_dir = case_dir / "instances"
        if prefer_instances and instances_dir.exists() and any(instances_dir.glob("*.nii.gz")):
            lesions = cls.extract_from_instances(case_id, instances_dir, spacing=spacing)
            if lesions:
                return lesions
                
        # Fallback to 26-CC
        if seg_data is None:
            seg_data = np.asanyarray(nib.load(str(seg_path)).dataobj)
        return cls.extract_from_segmentation_26cc(case_id, seg_data, spacing=spacing)
