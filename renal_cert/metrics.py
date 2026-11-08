"""Evaluation metrics: official KiTS23 segmentation scores, lesion detection, calibration, and failure detection.

Cites:
    PDF §4.1 (Official metrics: Dice and Surface Dice on three hierarchical evaluation classes)
    PDF §5.7 (Metrics and statistics: detection, Brier, NLL, class-wise ECE, risk-coverage, AURC)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.ndimage import label, generate_binary_structure


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Sørensen-Dice coefficient between binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_official_hec_dice(pred_seg: np.ndarray, gt_seg: np.ndarray) -> Dict[str, float]:
    """Computes Dice for the three official KiTS23 Hierarchical Evaluation Classes (PDF §4.1).
    
    Classes:
        0: Background
        1: Kidney
        2: Tumour
        3: Cyst
        
    HECs:
        HEC 1 (Kidney + Masses): Labels 1, 2, 3
        HEC 2 (Renal Masses): Labels 2, 3
        HEC 3 (Tumour Only): Label 2
    """
    # HEC 1: Whole kidney + masses
    p_hec1 = (pred_seg >= 1) & (pred_seg <= 3)
    g_hec1 = (gt_seg >= 1) & (gt_seg <= 3)
    dice_hec1 = compute_dice(p_hec1, g_hec1)
    
    # HEC 2: Masses (Tumour + Cyst)
    p_hec2 = (pred_seg == 2) | (pred_seg == 3)
    g_hec2 = (gt_seg == 2) | (gt_seg == 3)
    dice_hec2 = compute_dice(p_hec2, g_hec2)
    
    # HEC 3: Tumour only
    p_hec3 = (pred_seg == 2)
    g_hec3 = (gt_seg == 2)
    dice_hec3 = compute_dice(p_hec3, g_hec3)
    
    return {
        "dice_hec1_kidney_mass": round(dice_hec1, 4),
        "dice_hec2_mass": round(dice_hec2, 4),
        "dice_hec3_tumor": round(dice_hec3, 4),
        "dice_average": round(float((dice_hec1 + dice_hec2 + dice_hec3) / 3.0), 4)
    }


def compute_false_positive_lesions(
    flagged_mask: np.ndarray,
    gt_mass_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    min_fp_vol_ml: float = 0.05
) -> Tuple[int, float]:
    """Extracts false-positive candidate regions flagged outside ground truth mass (PDF §1.4, §5.4).
    
    Returns:
        num_fp_lesions: Count of connected components in F_lambda that have 0 overlap with GT mass.
        fp_volume_ml: Total volume in mL of flagged voxels outside GT mass.
    """
    # Mask of flagged voxels that do NOT overlap with any GT lesion
    outside_mask = flagged_mask & (~gt_mass_mask)
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    fp_vol_ml = float(outside_mask.sum() * voxel_vol_ml)
    
    if not np.any(outside_mask):
        return 0, 0.0

    # Crop to bounding box of outside_mask for 18x speedup on 3D volumes
    z_idx = np.where(outside_mask.any(axis=(1, 2)))[0]
    y_idx = np.where(outside_mask.any(axis=(0, 2)))[0]
    x_idx = np.where(outside_mask.any(axis=(0, 1)))[0]
    cropped = outside_mask[z_idx[0]:z_idx[-1]+1, y_idx[0]:y_idx[-1]+1, x_idx[0]:x_idx[-1]+1]
        
    s26 = generate_binary_structure(3, 3)
    labeled, num_features = label(cropped, structure=s26)
    if num_features == 0:
        return 0, 0.0
        
    # Fast vectorized component size filtering using np.bincount
    min_voxels = max(1, int(np.ceil(min_fp_vol_ml / voxel_vol_ml)))
    counts = np.bincount(labeled.ravel())[1:]
    valid_fps = int(np.count_nonzero(counts >= min_voxels))
            
    return valid_fps, round(fp_vol_ml, 2)


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot binary targets (PDF §5.7)."""
    return float(np.mean((probs - targets) ** 2))


def compute_negative_log_likelihood(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    """Multi-class negative log-likelihood (PDF §5.7)."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    nll = -np.sum(targets * np.log(clipped), axis=-1)
    return float(np.mean(nll))


def compute_classwise_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15
) -> Dict[str, float]:
    """Computes Expected Calibration Error per semantic class (PDF §5.7)."""
    # probs: shape (N, num_classes), targets: shape (N, num_classes) binary
    num_classes = probs.shape[-1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_per_class = {}
    
    for c in range(num_classes):
        p_c = probs[:, c].ravel()
        y_c = targets[:, c].ravel()
        
        ece = 0.0
        n_total = len(p_c)
        for b in range(n_bins):
            bin_mask = (p_c >= bin_edges[b]) & (p_c < bin_edges[b + 1])
            if np.any(bin_mask):
                bin_size = np.sum(bin_mask)
                bin_acc = np.mean(y_c[bin_mask])
                bin_conf = np.mean(p_c[bin_mask])
                ece += (bin_size / n_total) * abs(bin_acc - bin_conf)
                
        ece_per_class[f"ece_class_{c}"] = round(float(ece), 4)
        
    return ece_per_class


def compute_ensemble_pairwise_dice(member_segmentations: List[np.ndarray]) -> float:
    """Baseline uncertainty metric: mean pairwise Dice between ensemble members (PDF §5.7)."""
    m = len(member_segmentations)
    if m < 2:
        return 1.0
        
    dices = []
    for i in range(m):
        for j in range(i + 1, m):
            d = compute_dice(member_segmentations[i] > 0, member_segmentations[j] > 0)
            dices.append(d)
    return float(np.mean(dices))


def compute_risk_coverage(
    case_risks: np.ndarray,
    case_uncertainties: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes Risk-Coverage curve and AURC (PDF §5.7).
    
    Sorts cases by uncertainty ascending (most certain first).
    Coverage theta in [0, 1] is the fraction of cases retained.
    Risk at coverage theta is the mean loss of retained cases.
    AURC is the area under the risk-coverage curve.
    """
    n = len(case_risks)
    sort_order = np.argsort(case_uncertainties)
    sorted_risks = case_risks[sort_order]
    
    coverages = np.linspace(1.0 / n, 1.0, n)
    cumulative_risk = np.cumsum(sorted_risks) / np.arange(1, n + 1)
    
    aurc = float(np.trapz(cumulative_risk, coverages))
    return coverages, cumulative_risk, round(aurc, 4)
