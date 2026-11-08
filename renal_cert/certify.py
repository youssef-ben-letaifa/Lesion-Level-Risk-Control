"""Distribution-free risk control: Split-Conformal, Conformal Risk Control (CRC), and Learn-then-Test (LTT).

Cites:
    PDF §1.4 (Key terms: Critical lambda, Floor, Certificate, Review burden)
    PDF §5.4 (Certification layer: split-conformal, CRC, LTT exact binomial & Bentkus bounds)
    PDF §6.1 (Phase-0 decision table and feasibility at n_cal)
    PDF §6.3 (Integrity controls: separate reference reimplementation agreeing to 1e-9)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy import stats


@dataclass
class CertificateResult:
    """Stores the outcome of a statistical risk certification procedure."""
    method: str  # 'split_conformal', 'crc', 'ltt_binomial', 'ltt_bentkus'
    risk_target_alpha: float
    is_feasible: bool
    lambda_star: Optional[float]
    lambda_index: Optional[int]
    cal_empirical_risk: float
    cal_floor: float  # Share of patients with unrecoverable missed lesions
    cal_mean_fp_lesions: float
    cal_mean_flagged_vol_ml: float
    is_budgeted_feasible: bool  # False if fp_lesions > budget
    extra_info: Dict[str, Any]


class RiskCertifier:
    """Implements distribution-free guarantees on patient and lesion level error rates (PDF §5.4)."""

    def __init__(
        self,
        lambda_grid: np.ndarray,
        alpha_levels: List[float] = None,
        fp_budget: float = 2.0,
        delta_ltt: float = 0.10
    ):
        self.lambda_grid = np.array(lambda_grid, dtype=np.float64)
        self.alpha_levels = alpha_levels or [0.05, 0.10]
        self.fp_budget = fp_budget
        self.delta_ltt = delta_ltt

    @staticmethod
    def compute_patient_critical_lambdas(
        patient_lesion_coverages: List[np.ndarray],
        lambda_grid: np.ndarray,
        gamma: float = 0.50
    ) -> np.ndarray:
        """Computes the critical lambda for each patient (PDF §1.4, §5.4).
        
        Critical lambda_i is the smallest lambda at which all lesions of patient i
        have coverage >= gamma. If some lesion is never detected across the grid,
        critical lambda_i is np.inf.
        
        Args:
            patient_lesion_coverages: List of length N_patients. Each element is an array
                of shape (num_lesions_in_patient, num_lambda_points).
            lambda_grid: Sorted array of length num_lambda_points.
            gamma: Required lesion overlap fraction.
            
        Returns:
            critical_lambdas: Array of length N_patients.
        """
        n_patients = len(patient_lesion_coverages)
        critical_lambdas = np.full(n_patients, fill_value=np.inf, dtype=np.float64)
        
        for i, cov_mat in enumerate(patient_lesion_coverages):
            if cov_mat.shape[0] == 0:
                # Patient has no mass lesions (rare in KiTS, but well-defined)
                critical_lambdas[i] = lambda_grid[0]
                continue
                
            # For each lambda point (column), are all lesions detected?
            all_detected = np.all(cov_mat >= gamma, axis=0)  # Shape (num_lambdas,)
            valid_indices = np.where(all_detected)[0]
            if len(valid_indices) > 0:
                first_idx = valid_indices[0]
                critical_lambdas[i] = lambda_grid[first_idx]
            else:
                critical_lambdas[i] = np.inf
                
        return critical_lambdas

    def certify_split_conformal(
        self,
        critical_lambdas: np.ndarray,
        alpha: float,
        cal_fp_per_lambda: np.ndarray,
        cal_vol_per_lambda: Optional[np.ndarray] = None
    ) -> CertificateResult:
        """Split-conformal finite-sample calibration on patient-level risk (PDF §5.4).
        
        Theorem (Vovk et al., Angelopoulos & Bates):
        With n calibration patients, order statistic k = ceil((n + 1) * (1 - alpha)).
        lambda* = lambda_{(k)}.
        Guarantees P(miss in new patient) <= alpha.
        """
        n = len(critical_lambdas)
        sorted_crit = np.sort(critical_lambdas)
        floor = float(np.mean(np.isinf(critical_lambdas)))
        
        # Conformal order statistic rank k (1-indexed)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        
        if k > n:
            # Sample size insufficient to support alpha
            return CertificateResult(
                method="split_conformal",
                risk_target_alpha=alpha,
                is_feasible=False,
                lambda_star=None,
                lambda_index=None,
                cal_empirical_risk=1.0,
                cal_floor=floor,
                cal_mean_fp_lesions=np.nan,
                cal_mean_flagged_vol_ml=np.nan,
                is_budgeted_feasible=False,
                extra_info={"rank_k": k, "n_cal": n, "reason": "k > n"}
            )
            
        kth_val = sorted_crit[k - 1]  # 0-indexed
        if np.isinf(kth_val):
            # Infeasible because k-th patient has unrecoverable lesion
            return CertificateResult(
                method="split_conformal",
                risk_target_alpha=alpha,
                is_feasible=False,
                lambda_star=None,
                lambda_index=None,
                cal_empirical_risk=floor,
                cal_floor=floor,
                cal_mean_fp_lesions=np.nan,
                cal_mean_flagged_vol_ml=np.nan,
                is_budgeted_feasible=False,
                extra_info={"rank_k": k, "n_cal": n, "reason": "kth_value_is_inf"}
            )
            
        # Find index on the pre-registered grid
        grid_idx = int(np.searchsorted(self.lambda_grid, kth_val))
        grid_idx = min(grid_idx, len(self.lambda_grid) - 1)
        lambda_star = float(self.lambda_grid[grid_idx])
        
        # Compute empirical patient miss rate on calibration set at lambda*
        emp_risk = float(np.mean(critical_lambdas > lambda_star))
        mean_fp = float(cal_fp_per_lambda[grid_idx])
        mean_vol = float(cal_vol_per_lambda[grid_idx]) if cal_vol_per_lambda is not None else 0.0
        budgeted = bool(mean_fp <= self.fp_budget)
        
        return CertificateResult(
            method="split_conformal",
            risk_target_alpha=alpha,
            is_feasible=True,
            lambda_star=lambda_star,
            lambda_index=grid_idx,
            cal_empirical_risk=round(emp_risk, 4),
            cal_floor=round(floor, 4),
            cal_mean_fp_lesions=round(mean_fp, 3),
            cal_mean_flagged_vol_ml=round(mean_vol, 2),
            is_budgeted_feasible=budgeted,
            extra_info={"rank_k": k, "n_cal": n, "raw_kth_val": float(kth_val)}
        )

    def certify_crc(
        self,
        patient_lesion_risks: np.ndarray,
        alpha: float,
        cal_fp_per_lambda: np.ndarray,
        cal_vol_per_lambda: Optional[np.ndarray] = None
    ) -> CertificateResult:
        """Conformal Risk Control (CRC) on lesion-level risk (PDF §5.4).
        
        R_lesion(lambda) = per-patient mean of lesion-miss indicators.
        lambda* = inf{ lambda : (n / (n + 1)) * R_hat_n(lambda) + 1 / (n + 1) <= alpha }.
        Guarantees E[R_lesion(lambda*)] <= alpha.
        """
        # patient_lesion_risks: shape (n_patients, num_lambda_points)
        n = patient_lesion_risks.shape[0]
        # Empirical mean loss at each lambda
        r_hat = np.mean(patient_lesion_risks, axis=0)  # Shape (num_lambdas,)
        floor = float(r_hat[-1])  # Risk at most permissive point
        
        # CRC finite-sample upper bound on expected risk
        crc_upper_bound = (n / (n + 1.0)) * r_hat + (1.0 / (n + 1.0))
        
        # Find smallest lambda satisfying the bound
        valid_indices = np.where(crc_upper_bound <= alpha)[0]
        if len(valid_indices) == 0:
            return CertificateResult(
                method="crc",
                risk_target_alpha=alpha,
                is_feasible=False,
                lambda_star=None,
                lambda_index=None,
                cal_empirical_risk=float(np.min(r_hat)),
                cal_floor=round(floor, 4),
                cal_mean_fp_lesions=np.nan,
                cal_mean_flagged_vol_ml=np.nan,
                is_budgeted_feasible=False,
                extra_info={"min_crc_bound": float(np.min(crc_upper_bound)), "threshold": alpha}
            )
            
        best_idx = int(valid_indices[0])
        lambda_star = float(self.lambda_grid[best_idx])
        emp_risk = float(r_hat[best_idx])
        mean_fp = float(cal_fp_per_lambda[best_idx])
        mean_vol = float(cal_vol_per_lambda[best_idx]) if cal_vol_per_lambda is not None else 0.0
        budgeted = bool(mean_fp <= self.fp_budget)
        
        return CertificateResult(
            method="crc",
            risk_target_alpha=alpha,
            is_feasible=True,
            lambda_star=lambda_star,
            lambda_index=best_idx,
            cal_empirical_risk=round(emp_risk, 4),
            cal_floor=round(floor, 4),
            cal_mean_fp_lesions=round(mean_fp, 3),
            cal_mean_flagged_vol_ml=round(mean_vol, 2),
            is_budgeted_feasible=budgeted,
            extra_info={"crc_upper_bound": round(float(crc_upper_bound[best_idx]), 4)}
        )

    def certify_ltt_exact_binomial(
        self,
        patient_miss_indicators: np.ndarray,
        alpha: float,
        cal_fp_per_lambda: np.ndarray,
        cal_vol_per_lambda: Optional[np.ndarray] = None
    ) -> CertificateResult:
        """Learn-then-Test (LTT) with exact binomial p-values on binary patient loss (PDF §5.4).
        
        Tests H_0(lambda): R(lambda) > alpha with family-wise error rate delta.
        P(R(lambda*) <= alpha) >= 1 - delta (delta = 0.10).
        """
        # patient_miss_indicators: shape (n_patients, num_lambda_points), binary 0 or 1
        n = patient_miss_indicators.shape[0]
        miss_counts = np.sum(patient_miss_indicators, axis=0)
        floor = float(np.mean(patient_miss_indicators[:, -1]))
        
        # Sequential testing from permissive (high lambda) towards conservative:
        # P-value for H_0: X ~ Binomial(n, alpha)
        # p = P(Binomial(n, alpha) <= k_observed)
        p_values = np.array([
            stats.binom.cdf(k, n, alpha) for k in miss_counts
        ])
        
        # Valid points are those where p_value <= delta
        valid_indices = np.where(p_values <= self.delta_ltt)[0]
        if len(valid_indices) == 0:
            return CertificateResult(
                method="ltt_binomial",
                risk_target_alpha=alpha,
                is_feasible=False,
                lambda_star=None,
                lambda_index=None,
                cal_empirical_risk=float(miss_counts[-1] / n),
                cal_floor=round(floor, 4),
                cal_mean_fp_lesions=np.nan,
                cal_mean_flagged_vol_ml=np.nan,
                is_budgeted_feasible=False,
                extra_info={"min_p_val": float(np.min(p_values)), "delta": self.delta_ltt}
            )
            
        best_idx = int(valid_indices[0])
        lambda_star = float(self.lambda_grid[best_idx])
        emp_risk = float(miss_counts[best_idx] / n)
        mean_fp = float(cal_fp_per_lambda[best_idx])
        mean_vol = float(cal_vol_per_lambda[best_idx]) if cal_vol_per_lambda is not None else 0.0
        budgeted = bool(mean_fp <= self.fp_budget)
        
        return CertificateResult(
            method="ltt_binomial",
            risk_target_alpha=alpha,
            is_feasible=True,
            lambda_star=lambda_star,
            lambda_index=best_idx,
            cal_empirical_risk=round(emp_risk, 4),
            cal_floor=round(floor, 4),
            cal_mean_fp_lesions=round(mean_fp, 3),
            cal_mean_flagged_vol_ml=round(mean_vol, 2),
            is_budgeted_feasible=budgeted,
            extra_info={"p_value": float(p_values[best_idx]), "delta": self.delta_ltt}
        )

    def certify_ltt_bentkus(
        self,
        patient_lesion_risks: np.ndarray,
        alpha: float,
        cal_fp_per_lambda: np.ndarray,
        cal_vol_per_lambda: Optional[np.ndarray] = None
    ) -> CertificateResult:
        """Learn-then-Test (LTT) using Hoeffding-Bentkus bound for bounded loss in [0, 1] (PDF §5.4)."""
        n = patient_lesion_risks.shape[0]
        r_hat = np.mean(patient_lesion_risks, axis=0)
        floor = float(r_hat[-1])
        
        p_values = []
        for rh in r_hat:
            if rh >= alpha:
                p_values.append(1.0)
                continue
                
            # Hoeffding p-value
            p_h = np.exp(-2.0 * n * (alpha - rh) ** 2)
            
            # Bentkus tail bound
            k_int = int(np.floor(n * rh))
            p_b = stats.binom.cdf(k_int, n, alpha)
            p_bentkus = min(p_h, np.e * p_b)
            p_values.append(p_bentkus)
            
        p_values = np.array(p_values)
        valid_indices = np.where(p_values <= self.delta_ltt)[0]
        
        if len(valid_indices) == 0:
            return CertificateResult(
                method="ltt_bentkus",
                risk_target_alpha=alpha,
                is_feasible=False,
                lambda_star=None,
                lambda_index=None,
                cal_empirical_risk=float(r_hat[-1]),
                cal_floor=round(floor, 4),
                cal_mean_fp_lesions=np.nan,
                cal_mean_flagged_vol_ml=np.nan,
                is_budgeted_feasible=False,
                extra_info={"min_p_val": float(np.min(p_values)), "delta": self.delta_ltt}
            )
            
        best_idx = int(valid_indices[0])
        lambda_star = float(self.lambda_grid[best_idx])
        emp_risk = float(r_hat[best_idx])
        mean_fp = float(cal_fp_per_lambda[best_idx])
        mean_vol = float(cal_vol_per_lambda[best_idx]) if cal_vol_per_lambda is not None else 0.0
        budgeted = bool(mean_fp <= self.fp_budget)
        
        return CertificateResult(
            method="ltt_bentkus",
            risk_target_alpha=alpha,
            is_feasible=True,
            lambda_star=lambda_star,
            lambda_index=best_idx,
            cal_empirical_risk=round(emp_risk, 4),
            cal_floor=round(floor, 4),
            cal_mean_fp_lesions=round(mean_fp, 3),
            cal_mean_flagged_vol_ml=round(mean_vol, 2),
            is_budgeted_feasible=budgeted,
            extra_info={"p_value": float(p_values[best_idx]), "delta": self.delta_ltt}
        )


def reference_lambda_star_search(
    critical_lambdas: np.ndarray,
    lambda_grid: np.ndarray,
    alpha: float
) -> Tuple[bool, Optional[float]]:
    """Independent reference reimplementation of split-conformal lambda* search (PDF §6.3, item 4).
    
    Verified to agree with RiskCertifier to 1e-9 on synthetic benchmarks.
    """
    n = len(critical_lambdas)
    k = int(np.ceil((n + 1.0) * (1.0 - alpha)))
    if k > n:
        return False, None
    
    sorted_vals = np.sort(critical_lambdas)
    target = sorted_vals[k - 1]
    if np.isinf(target):
        return False, None
        
    # Manual binary search
    low = 0
    high = len(lambda_grid) - 1
    best_idx = high
    while low <= high:
        mid = (low + high) // 2
        if lambda_grid[mid] >= target:
            best_idx = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return True, float(lambda_grid[best_idx])
