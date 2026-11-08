"""Nested prediction families F0–F5.

Cites:
    PDF §1.4 (Prediction family definition: nested set F_lambda, larger lambda flags more)
    PDF §5.3 (Prediction families F0–F5: single-model, ensemble mean, member maximum,
             disagreement-aware, dilation, component selection)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure, maximum


class PredictionFamilies:
    """Implements and evaluates the six nested prediction families F0–F5 (PDF §5.3)."""

    @staticmethod
    def compute_ensemble_stats(member_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes summary statistics across ensemble member mass probability maps.
        
        Args:
            member_probs: Shape (M, Z, Y, X), where M is number of members, values in [0, 1].
            
        Returns:
            p_ens: Mean probability across members, shape (Z, Y, X).
            p_max: Maximum probability across members, shape (Z, Y, X).
            p_std: Standard deviation across members, shape (Z, Y, X).
        """
        p_ens = np.mean(member_probs, axis=0).astype(np.float32)
        p_max = np.max(member_probs, axis=0).astype(np.float32)
        p_std = np.std(member_probs, axis=0, ddof=1 if member_probs.shape[0] > 1 else 0).astype(np.float32)
        return p_ens, p_max, p_std

    @staticmethod
    def get_f0_score(member_probs: np.ndarray, member_idx: int = 0) -> np.ndarray:
        """F0: Single-model mass probability score (PDF §5.3)."""
        return member_probs[member_idx]

    @staticmethod
    def get_f1_score(p_ens: np.ndarray) -> np.ndarray:
        """F1: Ensemble mean mass probability score (PDF §5.3)."""
        return p_ens

    @staticmethod
    def get_f2_score(p_max: np.ndarray) -> np.ndarray:
        """F2: Maximum over members mass probability score (PDF §5.3)."""
        return p_max

    @staticmethod
    def get_f3_score(p_ens: np.ndarray, p_std: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """F3: Disagreement-aware score (p_ens + beta * SD) (PDF §5.3)."""
        score = p_ens + beta * p_std
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def get_f4_distance_map(p_ens: np.ndarray, spacing: Tuple[float, float, float], seed_threshold: float = 0.50) -> np.ndarray:
        """F4: Distance map (in mm) from the seed mask {p_ens >= 0.5} (PDF §5.3).
        
        Voxels inside the seed mask have distance 0.0 mm.
        Voxels outside have Euclidean distance to the nearest seed voxel.
        Flagged region at radius r: {distance_mm <= r}.
        """
        seed_mask = p_ens >= seed_threshold
        if not np.any(seed_mask):
            # No seed voxels: distance is infinite
            return np.full_like(p_ens, fill_value=1e6, dtype=np.float32)
            
        # distance_transform_edt computes distance to nearest zero voxel.
        # We invert seed_mask so seed voxels are 0.
        dist_mm = distance_transform_edt(~seed_mask, sampling=spacing).astype(np.float32)
        return dist_mm

    @staticmethod
    def get_f5_score(p_ens: np.ndarray, tau_min: float = 0.35) -> np.ndarray:
        """F5: Lesion-adaptive component selection score map (PDF §5.3).
        
        Connected components of {p_ens >= tau_min} are formed.
        Each voxel in component C_k receives score equal to max_{v in C_k} p_ens(v).
        Voxels outside all components receive score 0.0.
        Then flagging voxels with score >= 1 - lambda flags exactly the components
        whose peak probability >= 1 - lambda.
        """
        candidate_mask = p_ens >= tau_min
        if not np.any(candidate_mask):
            return np.zeros_like(p_ens, dtype=np.float32)

        s26 = generate_binary_structure(3, 3)
        labeled_mask, num_features = label(candidate_mask, structure=s26)
        if num_features == 0:
            return np.zeros_like(p_ens, dtype=np.float32)
        
        # Vectorized component peak assignment using scipy.ndimage.maximum
        peaks = maximum(p_ens, labeled_mask, index=np.arange(1, num_features + 1))
        lut = np.zeros(num_features + 1, dtype=np.float32)
        lut[1:] = peaks
        f5_score = lut[labeled_mask]
            
        return f5_score

    @classmethod
    def get_all_score_representations(
        cls,
        member_probs: np.ndarray,
        spacing: Tuple[float, float, float],
        beta_f3: float = 1.0,
        tau_min: float = 0.35
    ) -> Dict[str, Any]:
        """Constructs all 6 family score maps for efficient whole-grid evaluation."""
        p_ens, p_max, p_std = cls.compute_ensemble_stats(member_probs)
        
        f0_score = cls.get_f0_score(member_probs, member_idx=0)
        f1_score = cls.get_f1_score(p_ens)
        f2_score = cls.get_f2_score(p_max)
        f3_score = cls.get_f3_score(p_ens, p_std, beta=beta_f3)
        f4_dist = cls.get_f4_distance_map(p_ens, spacing=spacing, seed_threshold=0.50)
        f5_score = cls.get_f5_score(p_ens, tau_min=tau_min)
        
        return {
            "p_ens": p_ens,
            "p_max": p_max,
            "p_std": p_std,
            "F0": f0_score,
            "F1": f1_score,
            "F2": f2_score,
            "F3": f3_score,
            "F4_dist_mm": f4_dist,
            "F5": f5_score
        }

    @staticmethod
    def get_flagged_mask(
        family_name: str,
        family_repr: Any,
        param_value: float
    ) -> np.ndarray:
        """Returns binary flagged region F_lambda at parameter value.
        
        For F0, F1, F2, F3, F5: param_value is lambda, threshold T = 1 - lambda. Flagged if score >= T.
        For F4: param_value is radius r in mm. Flagged if dist_mm <= r.
        """
        if family_name in ["F0", "F1", "F2", "F3", "F5"]:
            threshold = 1.0 - param_value
            return family_repr >= threshold
        elif family_name == "F4":
            radius_mm = param_value
            return family_repr <= radius_mm
        else:
            raise ValueError(f"Unknown prediction family: {family_name}")
