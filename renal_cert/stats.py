"""Rigorous statistical inference: cluster bootstrap, paired Wilcoxon with Holm adjustment, Hodges–Lehmann, and Clopper–Pearson.

Cites:
    PDF §5.7 (Metrics and statistics: patient-level cluster-bootstrap intervals,
             paired Wilcoxon tests with Holm correction over at most 8 pre-specified comparisons,
             Hodges–Lehmann intervals, Clopper–Pearson intervals for rates, 5 seeds spread)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from scipy import stats


def clopper_pearson_interval(
    k: int,
    n: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson confidence interval for a binomial proportion (PDF §5.7)."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence_level
    
    # Lower bound
    if k == 0:
        low = 0.0
    else:
        low = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        
    # Upper bound
    if k == n:
        high = 1.0
    else:
        high = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        
    return round(low, 4), round(high, 4)


def hodges_lehmann_median_diff(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """Computes Hodges-Lehmann estimator of median difference for paired samples (x - y) (PDF §5.7).
    
    Walsh averages: (d_i + d_j) / 2 for all 1 <= i <= j <= n.
    Point estimate is the median of all Walsh averages.
    """
    diffs = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return 0.0, (0.0, 0.0)
        
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    walsh = np.sort(walsh)
    hl_estimate = float(np.median(walsh))
    
    # Wilcoxon distribution critical value for confidence interval
    # Using normal approximation for n >= 20
    alpha = 1.0 - confidence_level
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    n_pairs = len(walsh)
    w_mean = n * (n + 1) / 4.0
    w_sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    c_alpha = int(np.floor(w_mean - z * w_sd))
    
    low_idx = max(0, c_alpha - 1)
    high_idx = min(n_pairs - 1, n_pairs - c_alpha)
    ci = (float(walsh[low_idx]), float(walsh[high_idx]))
    
    return round(hl_estimate, 4), (round(ci[0], 4), round(ci[1], 4))


def holm_bonferroni_correction(p_values: List[float]) -> Tuple[List[float], List[bool]]:
    """Applies step-down Holm-Bonferroni correction over pre-specified tests (PDF §5.7).
    
    Returns:
        adjusted_p_values: Multiplicity-adjusted p-values in original order.
        is_significant: Boolean list indicating significance at family-wise alpha = 0.05.
    """
    m = len(p_values)
    if m == 0:
        return [], []
        
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        p_adj = min(1.0, p * multiplier)
        running_max = max(running_max, p_adj)
        adjusted[orig_idx] = running_max
        
    significant = [p <= 0.05 for p in adjusted]
    return [round(p, 5) for p in adjusted], significant


def paired_wilcoxon_tests(
    baseline_values: np.ndarray,
    comparison_dict: Dict[str, np.ndarray],
    alternative: str = "two-sided"
) -> pd.DataFrame:
    """Runs paired Wilcoxon signed-rank tests with Holm correction (PDF §5.7).
    
    Args:
        baseline_values: Patient-level metrics for F0 (e.g. false positives).
        comparison_dict: Mapping family name -> patient-level metrics.
    """
    results = []
    p_vals = []
    names = []
    
    for name, comp_vals in comparison_dict.items():
        diff = comp_vals - baseline_values
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            try:
                res = stats.wilcoxon(comp_vals, baseline_values, alternative=alternative)
                stat, p = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, p = 0.0, 1.0
                
        hl_est, hl_ci = hodges_lehmann_median_diff(comp_vals, baseline_values)
        mean_diff = float(np.mean(diff))
        
        results.append({
            "comparison": f"{name}_vs_baseline",
            "mean_baseline": round(float(np.mean(baseline_values)), 3),
            "mean_comparison": round(float(np.mean(comp_vals)), 3),
            "mean_diff": round(mean_diff, 3),
            "hl_median_diff": hl_est,
            "hl_ci_95": hl_ci,
            "wilcoxon_stat": round(stat, 2),
            "raw_p_value": p
        })
        p_vals.append(p)
        names.append(name)
        
    adj_p, sig = holm_bonferroni_correction(p_vals)
    for i in range(len(results)):
        results[i]["holm_adj_p_value"] = adj_p[i]
        results[i]["is_significant"] = sig[i]
        
    return pd.DataFrame(results)


def cluster_bootstrap_ci(
    patient_ids: List[str],
    eval_fn: Callable[[List[str]], float],
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20261026
) -> Tuple[float, Tuple[float, float]]:
    """Performs patient-level cluster bootstrap (PDF §5.7).
    
    Resamples patient IDs with replacement, preserving within-patient correlation.
    """
    n_patients = len(patient_ids)
    rng = np.random.RandomState(seed)
    point_est = eval_fn(patient_ids)
    
    boot_vals = []
    for _ in range(n_boot):
        sample_ids = [patient_ids[idx] for idx in rng.choice(n_patients, size=n_patients, replace=True)]
        val = eval_fn(sample_ids)
        boot_vals.append(val)
        
    boot_vals = np.sort(boot_vals)
    alpha = 1.0 - confidence_level
    low_idx = int(np.floor(alpha / 2.0 * n_boot))
    high_idx = int(np.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    
    ci = (float(boot_vals[low_idx]), float(boot_vals[high_idx]))
    return round(point_est, 4), (round(ci[0], 4), round(ci[1], 4))
