"""Streaming inference reduction: evaluates families F0–F5 on the fly and saves compact tables.

Cites:
    PDF §5.1 (Streaming reduction architecture: raw member probability maps are never stored)
    PDF §5.2 (Per-voxel mass probability p_mass = p_tumour + p_cyst; ensemble mean, max, SD)
    PDF §5.3 (Evaluation along pre-registered lambda and radius grids)
    PDF §5.4 (Compact reduction tables: lesion coverage, patient burden, case overlap)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from renal_cert.families import PredictionFamilies
from renal_cert.lesions import LesionInstance
from renal_cert.metrics import compute_official_hec_dice, compute_false_positive_lesions, compute_dice


class StreamingReducer:
    """Reduces raw member probability maps to compact tabular summaries without disk bloat."""

    def __init__(
        self,
        lambda_grid: np.ndarray,
        radius_grid: np.ndarray,
        beta_f3: float = 1.0,
        tau_min: float = 0.35,
        gamma_primary: float = 0.50
    ):
        self.lambda_grid = np.array(lambda_grid, dtype=np.float64)
        self.radius_grid = np.array(radius_grid, dtype=np.float64)
        self.beta_f3 = beta_f3
        self.tau_min = tau_min
        self.gamma_primary = gamma_primary

    def reduce_case(
        self,
        case_id: str,
        member_mass_probs: np.ndarray,  # Shape (M, Z, Y, X), mass probabilities
        gt_lesions: List[LesionInstance],
        gt_seg: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> Dict[str, pd.DataFrame]:
        """Processes a single case through all prediction families and generates reduction records."""
        M, Z, Y, X = member_mass_probs.shape
        gt_mass_mask = (gt_seg == 2) | (gt_seg == 3)
        
        # 1. Build all 6 family score representations
        score_reprs = PredictionFamilies.get_all_score_representations(
            member_mass_probs,
            spacing=spacing,
            beta_f3=self.beta_f3,
            tau_min=self.tau_min
        )
        p_ens = score_reprs["p_ens"]
        p_std = score_reprs["p_std"]

        # 2. Case-level segmentation overlap
        # Binary prediction at standard 0.5 threshold on ensemble mean
        pred_seg_bin = (p_ens >= 0.5).astype(np.uint8)
        # Synthesize multi-class for official HEC metrics (kidney=1, tumor=2, cyst=3)
        pred_multi = np.zeros_like(pred_seg_bin, dtype=np.uint8)
        pred_multi[pred_seg_bin > 0] = 2  # Treat detected mass as class 2
        hec_dices = compute_official_hec_dice(pred_multi, gt_seg)
        
        # Pairwise Dice between members
        pair_dices = []
        for i in range(M):
            for j in range(i + 1, M):
                d = compute_dice(member_mass_probs[i] >= 0.5, member_mass_probs[j] >= 0.5)
                pair_dices.append(d)
        avg_pairwise_dice = float(np.mean(pair_dices)) if pair_dices else 1.0

        df_case = pd.DataFrame([{
            "case_id": case_id,
            "num_members": M,
            "num_gt_lesions": len(gt_lesions),
            "dice_hec1": hec_dices["dice_hec1_kidney_mass"],
            "dice_hec2": hec_dices["dice_hec2_mass"],
            "dice_hec3": hec_dices["dice_hec3_tumor"],
            "dice_avg": hec_dices["dice_average"],
            "pairwise_member_dice": round(avg_pairwise_dice, 4)
        }])

        # 3. Lesion-level ensemble stats
        lesion_stats = []
        for les in gt_lesions:
            zc = les.voxel_coords[:, 0]
            yc = les.voxel_coords[:, 1]
            xc = les.voxel_coords[:, 2]
            les_p_mean = float(np.mean(p_ens[zc, yc, xc]))
            les_std_mean = float(np.mean(p_std[zc, yc, xc]))
            lesion_stats.append({
                "case_id": case_id,
                "lesion_id": les.lesion_id,
                "class_type": les.class_type,
                "volume_ml": les.volume_ml,
                "equiv_diam_cm": les.equiv_diameter_cm,
                "size_stratum": les.size_stratum,
                "mean_ens_prob": round(les_p_mean, 4),
                "mean_ens_std": round(les_std_mean, 4)
            })
        df_lesion_stats = pd.DataFrame(lesion_stats)

        # 4. Family and grid evaluation
        coverage_records = []
        burden_records = []
        
        families = ["F0", "F1", "F2", "F3", "F4", "F5"]
        for fam in families:
            param_grid = self.radius_grid if fam == "F4" else self.lambda_grid
            fam_repr = score_reprs["F4_dist_mm"] if fam == "F4" else score_reprs[fam]
            
            for l_idx, param_val in enumerate(param_grid):
                flagged = PredictionFamilies.get_flagged_mask(fam, fam_repr, param_val)
                
                # False positive burden
                fp_count, fp_vol = compute_false_positive_lesions(flagged, gt_mass_mask, spacing=spacing)
                
                # Coverage per lesion
                lesion_misses = []
                for les in gt_lesions:
                    cov = les.compute_coverage(flagged)
                    is_det_05 = bool(cov >= 0.50)
                    is_det_01 = bool(cov >= 0.10)
                    is_det_09 = bool(cov >= 0.90)
                    if not is_det_05:
                        lesion_misses.append(1)
                    else:
                        lesion_misses.append(0)
                        
                    coverage_records.append({
                        "case_id": case_id,
                        "lesion_id": les.lesion_id,
                        "family": fam,
                        "param_idx": l_idx,
                        "param_value": round(float(param_val), 6),
                        "coverage": round(float(cov), 4),
                        "detected_gamma_05": is_det_05,
                        "detected_gamma_01": is_det_01,
                        "detected_gamma_09": is_det_09
                    })
                    
                # Patient-level risk indicators
                num_les = len(gt_lesions)
                is_any_missed = bool(any(lesion_misses)) if num_les > 0 else False
                lesion_miss_rate = float(np.mean(lesion_misses)) if num_les > 0 else 0.0
                
                burden_records.append({
                    "case_id": case_id,
                    "family": fam,
                    "param_idx": l_idx,
                    "param_value": round(float(param_val), 6),
                    "fp_lesions": fp_count,
                    "fp_vol_ml": fp_vol,
                    "patient_miss_05": int(is_any_missed),
                    "lesion_miss_rate_05": round(lesion_miss_rate, 4)
                })

        df_coverage = pd.DataFrame(coverage_records)
        df_burden = pd.DataFrame(burden_records)

        return {
            "case_metrics": df_case,
            "lesion_stats": df_lesion_stats,
            "lesion_coverage": df_coverage,
            "patient_burden": df_burden
        }
