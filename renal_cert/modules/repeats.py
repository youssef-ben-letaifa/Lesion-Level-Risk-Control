"""Module C: Sensitivity to reference annotations using KiTS multi-reader repeats.

Cites:
    PDF §4.2 (Annotation repeats: 3 separate annotations per ROI)
    PDF §5.6 (Module C: inter-repeat Dice, risk against union/intersection, Spearman correlation)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib

from renal_cert.metrics import compute_dice


class RepeatAnalyzer:
    """Evaluates certificate stability and uncertainty alignment against repeated annotations (PDF §5.6)."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def find_cases_with_repeats(self) -> List[str]:
        """Identifies cases with instance annotations across all 3 reader repeats (PDF §4.2, §5.6)."""
        valid = []
        for c in sorted(self.dataset_dir.glob("case_*")):
            inst_dir = c / "instances"
            if inst_dir.exists():
                r1 = list(inst_dir.glob("*_annotation-1.nii.gz"))
                r2 = list(inst_dir.glob("*_annotation-2.nii.gz"))
                r3 = list(inst_dir.glob("*_annotation-3.nii.gz"))
                if r1 and r2 and r3:
                    valid.append(c.name)
        return valid

    def run_module_c_analysis(self, case_ids: List[str]) -> Dict[str, Any]:
        """Evaluates inter-repeat Dice and reader agreement (PDF §5.6)."""
        records = []
        for cid in case_ids:
            res = self.compute_case_inter_repeat_dice(cid)
            if res is not None:
                records.append(res)
        df_dice = pd.DataFrame(records)
        return {
            "num_cases_analyzed": len(df_dice),
            "df_dice": df_dice,
            "mean_inter_repeat_dice": float(df_dice["dice_mean_repeats"].mean()) if not df_dice.empty else 0.0
        }

    def generate_module_c_figures(
        self,
        c_results: Dict[str, Any],
        fig_dir: Path,
        registry=None,
        tier: str = "SMOKE"
    ) -> List[Path]:
        """Plots Module C Group C figures (C1) for multi-reader repeat agreement."""
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        df_dice = c_results["df_dice"]
        csv_path = fig_dir / f"C1_inter_repeat_dice_{tier}.csv"
        png_path = fig_dir / f"C1_inter_repeat_dice_{tier}.png"
        pdf_path = fig_dir / f"C1_inter_repeat_dice_{tier}.pdf"
        
        df_dice.to_csv(csv_path, index=False)
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        vals = [
            df_dice["dice_rep1_rep2"].values,
            df_dice["dice_rep1_rep3"].values,
            df_dice["dice_rep2_rep3"].values,
            df_dice["dice_mean_repeats"].values
        ]
        labels = ["R1 vs R2", "R1 vs R3", "R2 vs R3", "Mean Pairwise"]
        colors = ["#386cb0", "#fdb462", "#7fc97f", "#ef3b2c"]
        
        bp = ax.boxplot(vals, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            
        ax.set_ylabel("Dice Similarity Coefficient", fontsize=10)
        ax.set_title(f"Module C: Multi-Reader Annotation Repeat Agreement (n={len(df_dice)}, {tier})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        
        if registry is not None:
            registry.register(
                fig_id="C1_REPEAT_DICE",
                title=f"Module C: Multi-Reader Inter-Repeat Agreement ({tier})",
                script_name="repeats.py:generate_module_c_figures",
                csv_path=csv_path,
                png_path=png_path,
                split_used="REPEATS",
                n_samples=len(df_dice),
                tier=tier,
                description="Distribution of pairwise Dice agreement across reader repeats 1, 2, and 3"
            )
            
        return [png_path, pdf_path]

    def compute_case_inter_repeat_dice(self, case_id: str) -> Optional[Dict[str, float]]:
        """Calculates pairwise Dice between repeat 1, 2, and 3 for a given case."""
        inst_dir = self.dataset_dir / case_id / "instances"
        if not inst_dir.exists():
            return None
            
        # Collect masks for repeat 1, 2, 3
        repeat_masks = {1: [], 2: [], 3: []}
        for rep in [1, 2, 3]:
            files = list(inst_dir.glob(f"*_annotation-{rep}.nii.gz"))
            if not files:
                continue
            combined = None
            for f in files:
                arr = np.asanyarray(nib.load(str(f)).dataobj) > 0
                if combined is None:
                    combined = arr
                else:
                    combined |= arr
            if combined is not None:
                repeat_masks[rep] = combined

        if not (len(repeat_masks[1]) > 0 and len(repeat_masks[2]) > 0 and len(repeat_masks[3]) > 0):
            return None

        d12 = compute_dice(repeat_masks[1], repeat_masks[2])
        d13 = compute_dice(repeat_masks[1], repeat_masks[3])
        d23 = compute_dice(repeat_masks[2], repeat_masks[3])
        
        return {
            "case_id": case_id,
            "dice_rep1_rep2": round(d12, 4),
            "dice_rep1_rep3": round(d13, 4),
            "dice_rep2_rep3": round(d23, 4),
            "dice_mean_repeats": round(float((d12 + d13 + d23) / 3.0), 4)
        }

    @staticmethod
    def compute_disagreement_correlation(
        lesion_stats_df: pd.DataFrame,
        repeat_disagreement_scores: np.ndarray
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Computes Spearman correlation and bootstrap CI between ensemble SD and annotator disagreement (PDF §5.6)."""
        ens_sd = lesion_stats_df["mean_ens_std"].values
        rep_dis = np.array(repeat_disagreement_scores, dtype=np.float64)
        
        res = stats.spearmanr(ens_sd, rep_dis)
        corr = float(res.correlation)
        p_val = float(res.pvalue)
        
        # Bootstrap CI
        n = len(ens_sd)
        boot_corrs = []
        rng = np.random.RandomState(20261026)
        for _ in range(1000):
            idx = rng.choice(n, size=n, replace=True)
            bc = stats.spearmanr(ens_sd[idx], rep_dis[idx]).correlation
            if not np.isnan(bc):
                boot_corrs.append(bc)
                
        boot_corrs = np.sort(boot_corrs)
        ci = (float(boot_corrs[25]), float(boot_corrs[975])) if len(boot_corrs) >= 1000 else (corr, corr)
        
        return round(corr, 4), p_val, (round(ci[0], 4), round(ci[1], 4))
