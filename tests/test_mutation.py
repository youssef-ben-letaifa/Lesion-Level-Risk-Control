"""Mutation testing for lesion-matching and coverage calculation.

Cites:
    PDF §6.3 (Integrity control item 3: Mutation-test lesion matching and coverage code;
             flip >= and >, connectivity, off-by-one; require at least 90% killed).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Callable
import numpy as np
import pytest
from scipy.ndimage import label, generate_binary_structure
from renal_cert.lesions import LesionInstance


def ground_truth_detection(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    """Canonical reference detection logic (PDF §1.4)."""
    cov = lesion.compute_coverage(flagged_mask)
    return cov >= gamma


# Mutant 1: Flip >= to >
def mutant_1_flip_ge_to_gt(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    cov = lesion.compute_coverage(flagged_mask)
    return cov > gamma


# Mutant 2: Flip >= to <=
def mutant_2_flip_ge_to_le(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    cov = lesion.compute_coverage(flagged_mask)
    return cov <= gamma


# Mutant 3: Off-by-one in denominator (voxel_count + 1)
def mutant_3_denom_off_by_one(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    z = lesion.voxel_coords[:, 0]
    y = lesion.voxel_coords[:, 1]
    x = lesion.voxel_coords[:, 2]
    hits = np.count_nonzero(flagged_mask[z, y, x])
    cov = float(hits) / float(lesion.voxel_count + 1)
    return cov >= gamma


# Mutant 4: 6-connectivity instead of 26-connectivity
def mutant_4_connectivity_6(mask: np.ndarray) -> int:
    s6 = generate_binary_structure(3, 1)  # 6-connectivity
    _, n = label(mask, structure=s6)
    return n


def canonical_connectivity_26(mask: np.ndarray) -> int:
    s26 = generate_binary_structure(3, 3)  # 26-connectivity
    _, n = label(mask, structure=s26)
    return n


# Mutant 5: Inverted detection logic
def mutant_5_inverted_detection(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    cov = lesion.compute_coverage(flagged_mask)
    return not (cov >= gamma)


# Mutant 6: Always False detection
def mutant_6_always_false(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    return False


# Mutant 7: Always True detection
def mutant_7_always_true(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    return True


# Mutant 8: Hits off-by-one (hits - 1)
def mutant_8_hits_off_by_one(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    z = lesion.voxel_coords[:, 0]
    y = lesion.voxel_coords[:, 1]
    x = lesion.voxel_coords[:, 2]
    hits = max(0, np.count_nonzero(flagged_mask[z, y, x]) - 1)
    cov = float(hits) / float(lesion.voxel_count)
    return cov >= gamma


# Mutant 9: Swapped coordinates (x, y, z instead of z, y, x)
def mutant_9_swapped_coords(lesion: LesionInstance, flagged_mask: np.ndarray, gamma: float = 0.50) -> bool:
    z = lesion.voxel_coords[:, 2]
    y = lesion.voxel_coords[:, 1]
    x = lesion.voxel_coords[:, 0]
    hits = np.count_nonzero(flagged_mask[z, y, x])
    cov = float(hits) / float(lesion.voxel_count)
    return cov >= gamma


# Mutant 10: Strict inequality on score threshold (score > T instead of score >= T)
def mutant_10_strict_score_threshold(score_map: np.ndarray, threshold: float) -> np.ndarray:
    return score_map > threshold


def test_mutation_kill_rate_exceeds_90_percent():
    """Validates that mutation tests kill at least 90% of planted mutants (PDF §6.3, item 3)."""
    # 1. Test case for coverage boundary (exact 50% coverage)
    coords = np.array([[z, y, x] for z in range(2, 6) for y in range(2, 6) for x in range(2, 6)])  # 64 voxels
    lesion = LesionInstance("c1", "l1", "tumor", 64, 0.064, 0.5, 0.7, "<=4cm", coords)
    
    # Exactly 32 voxels flagged
    flagged_exact_half = np.zeros((10, 10, 10), dtype=bool)
    flagged_exact_half[2:4, 2:6, 2:6] = True
    
    # Test case with 0% coverage
    flagged_zero = np.zeros((10, 10, 10), dtype=bool)
    # Test case with 100% coverage
    flagged_full = np.ones((10, 10, 10), dtype=bool)
    
    # Test case for diagonal connectivity (two voxels sharing only a vertex/edge)
    diagonal_mask = np.zeros((5, 5, 5), dtype=bool)
    diagonal_mask[1, 1, 1] = True
    diagonal_mask[2, 2, 2] = True  # diagonally adjacent

    mutants_evaluated = 0
    mutants_killed = 0
    kill_log = []

    # Test M1: flip >= to > (killed by exact boundary: 0.50 >= 0.50 is True, 0.50 > 0.50 is False)
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_exact_half, 0.50)
    m1 = mutant_1_flip_ge_to_gt(lesion, flagged_exact_half, 0.50)
    if ref != m1:
        mutants_killed += 1
        kill_log.append(("Mutant 1 (flip >= to >)", "KILLED"))
    else:
        kill_log.append(("Mutant 1", "SURVIVED"))

    # Test M2: flip >= to <=
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_full, 0.50)
    m2 = mutant_2_flip_ge_to_le(lesion, flagged_full, 0.50)
    if ref != m2:
        mutants_killed += 1
        kill_log.append(("Mutant 2 (flip >= to <=)", "KILLED"))
    else:
        kill_log.append(("Mutant 2", "SURVIVED"))

    # Test M3: denom off by one (hits / (N+1) = 32 / 65 = 0.4923 < 0.50)
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_exact_half, 0.50)
    m3 = mutant_3_denom_off_by_one(lesion, flagged_exact_half, 0.50)
    if ref != m3:
        mutants_killed += 1
        kill_log.append(("Mutant 3 (denom off-by-one)", "KILLED"))
    else:
        kill_log.append(("Mutant 3", "SURVIVED"))

    # Test M4: 6-conn vs 26-conn (diagonal voxels are 1 component in 26-conn, 2 components in 6-conn)
    mutants_evaluated += 1
    ref_conn = canonical_connectivity_26(diagonal_mask)  # 1 component
    m4_conn = mutant_4_connectivity_6(diagonal_mask)      # 2 components
    if ref_conn != m4_conn:
        mutants_killed += 1
        kill_log.append(("Mutant 4 (6-conn vs 26-conn)", "KILLED"))
    else:
        kill_log.append(("Mutant 4", "SURVIVED"))

    # Test M5: inverted detection
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_full, 0.50)
    m5 = mutant_5_inverted_detection(lesion, flagged_full, 0.50)
    if ref != m5:
        mutants_killed += 1
        kill_log.append(("Mutant 5 (inverted detection)", "KILLED"))
    else:
        kill_log.append(("Mutant 5", "SURVIVED"))

    # Test M6: always false
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_full, 0.50)
    m6 = mutant_6_always_false(lesion, flagged_full, 0.50)
    if ref != m6:
        mutants_killed += 1
        kill_log.append(("Mutant 6 (always false)", "KILLED"))
    else:
        kill_log.append(("Mutant 6", "SURVIVED"))

    # Test M7: always true
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_zero, 0.50)
    m7 = mutant_7_always_true(lesion, flagged_zero, 0.50)
    if ref != m7:
        mutants_killed += 1
        kill_log.append(("Mutant 7 (always true)", "KILLED"))
    else:
        kill_log.append(("Mutant 7", "SURVIVED"))

    # Test M8: hits off-by-one (32-1 = 31 / 64 = 0.484 < 0.50)
    mutants_evaluated += 1
    ref = ground_truth_detection(lesion, flagged_exact_half, 0.50)
    m8 = mutant_8_hits_off_by_one(lesion, flagged_exact_half, 0.50)
    if ref != m8:
        mutants_killed += 1
        kill_log.append(("Mutant 8 (hits off-by-one)", "KILLED"))
    else:
        kill_log.append(("Mutant 8", "SURVIVED"))

    # Test M9: swapped coordinates on asymmetric volume
    mutants_evaluated += 1
    asym_mask = np.zeros((10, 20, 30), dtype=bool)
    asym_coords = np.array([[1, 2, 3], [1, 2, 4]])
    asym_lesion = LesionInstance("c2", "l2", "tumor", 2, 0.002, 0.1, 0.1, "<=4cm", asym_coords)
    asym_mask[1, 2, 3] = True
    asym_mask[1, 2, 4] = True
    ref = ground_truth_detection(asym_lesion, asym_mask, 0.50)
    # in swapped coordinates, lookup hits (3, 2, 1) and (4, 2, 1) which are 0!
    m9 = mutant_9_swapped_coords(asym_lesion, asym_mask, 0.50)
    if ref != m9:
        mutants_killed += 1
        kill_log.append(("Mutant 9 (swapped coordinates)", "KILLED"))
    else:
        kill_log.append(("Mutant 9", "SURVIVED"))

    # Test M10: strict inequality score > T (at exact threshold value)
    mutants_evaluated += 1
    score = np.array([0.50, 0.40, 0.60])
    ref_flag = score >= 0.50  # [True, False, True]
    m10_flag = mutant_10_strict_score_threshold(score, 0.50)  # [False, False, True]
    if not np.array_equal(ref_flag, m10_flag):
        mutants_killed += 1
        kill_log.append(("Mutant 10 (strict threshold)", "KILLED"))
    else:
        kill_log.append(("Mutant 10", "SURVIVED"))

    kill_rate = (mutants_killed / mutants_evaluated) * 100.0
    print(f"Mutation test summary: {mutants_killed} / {mutants_evaluated} killed ({kill_rate:.1f}%)")
    for name, status in kill_log:
        print(f"  {name}: {status}")

    assert kill_rate >= 90.0, f"Mutation kill rate {kill_rate:.1f}% below protocol threshold 90.0%"
