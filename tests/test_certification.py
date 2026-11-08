"""Unit tests for statistical certification: split-conformal, CRC, LTT, and independent dual reimplementation.

Cites:
    PDF §5.4 (Certification layer formulas and feasibility boundaries at n=98)
    PDF §6.3 (Integrity control item 4: dual reimplementation agreeing to 1e-9)
"""

import numpy as np
import pytest
from renal_cert.certify import RiskCertifier, reference_lambda_star_search


def test_dual_reimplementation_agreement():
    """Validates that main and reference lambda* searches agree to 1e-9 across 50 random datasets (PDF §6.3, item 4)."""
    rng = np.random.RandomState(2026)
    lambda_grid = np.linspace(0.1, 0.999, 100)
    certifier = RiskCertifier(lambda_grid=lambda_grid)
    
    for trial in range(50):
        n = rng.randint(50, 150)
        alpha = rng.choice([0.05, 0.10, 0.15])
        
        # Sample synthetic critical lambdas (with 5% chance of infinity)
        raw_vals = rng.uniform(0.15, 0.95, size=n)
        inf_mask = rng.uniform(0, 1, size=n) < 0.05
        raw_vals[inf_mask] = np.inf
        
        # Main certifier call
        fp_dummy = np.zeros_like(lambda_grid)
        res_main = certifier.certify_split_conformal(raw_vals, alpha=alpha, cal_fp_per_lambda=fp_dummy)
        
        # Independent reference search call
        is_feas_ref, l_ref = reference_lambda_star_search(raw_vals, lambda_grid, alpha=alpha)
        
        assert res_main.is_feasible == is_feas_ref, f"Trial {trial}: Feasibility mismatch: main={res_main.is_feasible}, ref={is_feas_ref}"
        if res_main.is_feasible:
            diff = abs(res_main.lambda_star - l_ref)
            assert diff <= 1e-9, f"Trial {trial}: Lambda* mismatch: main={res_main.lambda_star}, ref={l_ref}, diff={diff}"


def test_split_conformal_feasibility_boundary_n98():
    """Tests exact feasibility conditions derived in PDF §5.4 table for n=98.
    
    At n=98:
    alpha=0.05: k = ceil(99 * 0.95) = 95. Infeasible if > 3 patients have lambda = inf.
    alpha=0.10: k = ceil(99 * 0.90) = 90. Infeasible if > 8 patients have lambda = inf.
    """
    n = 98
    lambda_grid = np.linspace(0.1, 0.99, 50)
    certifier = RiskCertifier(lambda_grid=lambda_grid)
    dummy_fp = np.zeros_like(lambda_grid)
    
    # Exactly 3 unrecoverable patients at alpha=0.05: FEASIBLE (rank 95 is finite)
    crit_3_inf = np.full(n, fill_value=0.5)
    crit_3_inf[-3:] = np.inf
    res_05_feas = certifier.certify_split_conformal(crit_3_inf, alpha=0.05, cal_fp_per_lambda=dummy_fp)
    assert res_05_feas.is_feasible is True
    
    # 4 unrecoverable patients at alpha=0.05: INFEASIBLE (rank 95 is inf)
    crit_4_inf = np.full(n, fill_value=0.5)
    crit_4_inf[-4:] = np.inf
    res_05_infeas = certifier.certify_split_conformal(crit_4_inf, alpha=0.05, cal_fp_per_lambda=dummy_fp)
    assert res_05_infeas.is_feasible is False
    
    # Exactly 8 unrecoverable patients at alpha=0.10: FEASIBLE (rank 90 is finite)
    crit_8_inf = np.full(n, fill_value=0.5)
    crit_8_inf[-8:] = np.inf
    res_10_feas = certifier.certify_split_conformal(crit_8_inf, alpha=0.10, cal_fp_per_lambda=dummy_fp)
    assert res_10_feas.is_feasible is True
    
    # 9 unrecoverable patients at alpha=0.10: INFEASIBLE (rank 90 is inf)
    crit_9_inf = np.full(n, fill_value=0.5)
    crit_9_inf[-9:] = np.inf
    res_10_infeas = certifier.certify_split_conformal(crit_9_inf, alpha=0.10, cal_fp_per_lambda=dummy_fp)
    assert res_10_infeas.is_feasible is False


def test_crc_exact_bound():
    """Tests Conformal Risk Control threshold calculation on synthetic loss matrix."""
    n = 100
    num_grid = 20
    lambda_grid = np.linspace(0.1, 0.99, num_grid)
    certifier = RiskCertifier(lambda_grid=lambda_grid)
    
    # Construct synthetic non-increasing risk: linear from 0.2 down to 0.02
    loss_matrix = np.zeros((n, num_grid))
    for col in range(num_grid):
        val = 0.20 - (0.18 * col / (num_grid - 1))
        loss_matrix[:, col] = val
        
    dummy_fp = np.zeros(num_grid)
    res_crc = certifier.certify_crc(loss_matrix, alpha=0.10, cal_fp_per_lambda=dummy_fp)
    assert res_crc.is_feasible is True
    
    # Upper bound check
    emp_r = res_crc.cal_empirical_risk
    crc_bound = (n / (n + 1.0)) * emp_r + (1.0 / (n + 1.0))
    assert crc_bound <= 0.10 + 1e-9
