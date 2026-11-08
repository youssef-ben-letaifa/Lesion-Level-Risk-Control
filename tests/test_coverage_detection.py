"""Unit tests for lesion coverage, detection criteria, and family score representations.

Cites:
    PDF §1.4 (Key terms: Detected lesion coverage fraction gamma)
    PDF §5.3 (Prediction families F0–F5)
    PDF §6.3 (Integrity controls: unit tests with known mathematical ground truth)
"""

import numpy as np
import pytest
from renal_cert.lesions import LesionInstance, LesionExtractor
from renal_cert.families import PredictionFamilies


def test_synthetic_lesion_coverage_known_answer():
    """Validates coverage calculation against exact synthetic geometry."""
    # Create a 3D box of shape (10, 10, 10)
    # Lesion occupies coordinates z in [2, 6), y in [2, 6), x in [2, 6) -> 4 * 4 * 4 = 64 voxels
    coords = []
    for z in range(2, 6):
        for y in range(2, 6):
            for x in range(2, 6):
                coords.append([z, y, x])
    coords = np.array(coords)
    
    lesion = LesionInstance(
        case_id="case_synth_001",
        lesion_id="lesion_01",
        class_type="tumor",
        voxel_count=64,
        volume_ml=0.064,
        equiv_diameter_cm=0.496,
        max_extent_cm=0.69,
        size_stratum="<=4cm",
        voxel_coords=coords
    )
    
    flagged_mask = np.zeros((10, 10, 10), dtype=bool)
    # Flag exactly half of the lesion voxels (32 voxels: z in [2, 4))
    flagged_mask[2:4, 2:6, 2:6] = True
    
    cov = lesion.compute_coverage(flagged_mask)
    assert abs(cov - 0.50) < 1e-9, f"Expected coverage 0.50, got {cov}"
    
    # Check detection at different gamma levels
    assert lesion.is_detected(flagged_mask, gamma=0.50) is True
    assert lesion.is_detected(flagged_mask, gamma=0.10) is True
    assert lesion.is_detected(flagged_mask, gamma=0.51) is False
    assert lesion.is_detected(flagged_mask, gamma=0.90) is False


def test_prediction_families_nested_monotonicity():
    """Property test: as parameter lambda increases, flagged region F_lambda is monotonically non-decreasing."""
    rng = np.random.RandomState(42)
    # Synthetic ensemble of 3 members, shape (3, 16, 16, 16)
    member_probs = rng.uniform(0.0, 1.0, size=(3, 16, 16, 16)).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)
    
    score_reprs = PredictionFamilies.get_all_score_representations(
        member_probs, spacing=spacing, beta_f3=1.0, tau_min=0.3
    )
    
    lambda_grid = np.linspace(0.1, 0.99, 15)
    
    for fam in ["F0", "F1", "F2", "F3", "F5"]:
        prev_flagged = None
        for l_val in lambda_grid:
            flagged = PredictionFamilies.get_flagged_mask(fam, score_reprs[fam], l_val)
            if prev_flagged is not None:
                # Every voxel flagged in previous (smaller) lambda must still be flagged in current lambda
                uncovered_prior = np.logical_and(prev_flagged, ~flagged)
                assert np.count_nonzero(uncovered_prior) == 0, f"Monotonicity violation in family {fam} at lambda={l_val}"
            prev_flagged = flagged


def test_f4_dilation_monotonicity():
    """Property test: as radius r increases, F4 dilation region is monotonically non-decreasing."""
    rng = np.random.RandomState(42)
    p_ens = rng.uniform(0.0, 1.0, size=(16, 16, 16)).astype(np.float32)
    spacing = (1.0, 1.0, 1.0)
    
    dist_map = PredictionFamilies.get_f4_distance_map(p_ens, spacing=spacing, seed_threshold=0.5)
    radius_grid = np.linspace(0.0, 15.0, 10)
    
    prev_flagged = None
    for r in radius_grid:
        flagged = PredictionFamilies.get_flagged_mask("F4", dist_map, r)
        if prev_flagged is not None:
            uncovered_prior = np.logical_and(prev_flagged, ~flagged)
            assert np.count_nonzero(uncovered_prior) == 0, f"Monotonicity violation in F4 dilation at radius={r}"
        prev_flagged = flagged
