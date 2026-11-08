"""Data audit and Figure Group A generator.

Cites:
    PDF §4.1 (KiTS23 dataset characteristics and counts)
    PDF §4.2 (Cohort, labels, and annotation repeats)
    PDF §4.4 (Checks before any modelling: case uniqueness, repeats, phase proxy)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from renal_cert.lesions import LesionExtractor


class DataAuditor:
    """Performs comprehensive data audit and generates Group A figures (PDF §4)."""

    def __init__(self, dataset_dir: Path, output_dir: Path, tier: str = "SMOKE"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.tier = tier.upper()
        self.fig_dir = self.output_dir / "figures" / self.tier / "group_A"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def run_full_audit(self) -> Dict[str, Any]:
        """Runs audit over all cases and writes CSV data + PNG figures at 300 DPI."""
        json_path = self.dataset_dir / "kits23.json"
        with open(json_path, "r") as f:
            meta = json.load(f)
            
        case_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
        
        # 1. Slice thickness distribution
        slice_data = []
        for d in case_dirs:
            seg_p = d / "segmentation.nii.gz"
            if seg_p.exists():
                hdr = nib.load(str(seg_p)).header
                st = float(hdr.get_zooms()[0])
                slice_data.append({"case_id": d.name, "slice_thickness_mm": round(st, 2)})
        df_slices = pd.DataFrame(slice_data)
        df_slices.to_csv(self.fig_dir / f"A3_slice_thickness_{self.tier}.csv", index=False)
        
        # Plot A3: Slice thickness histogram
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        counts = df_slices["slice_thickness_mm"].value_counts().sort_index()
        bars = ax.bar([str(x) for x in counts.index], counts.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_title(f"A3: Slice Thickness Distribution (KiTS23, n={len(df_slices)}, Tier: {self.tier})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Slice Thickness (mm)", fontsize=10)
        ax.set_ylabel("Number of Patients", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 2, f"{int(h)}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"A3_slice_thickness_{self.tier}.png", dpi=300)
        fig.savefig(self.fig_dir / f"A3_slice_thickness_{self.tier}.pdf")
        plt.close(fig)

        # 2. Histologic subtypes
        df_meta = pd.DataFrame(meta)
        hist_counts = df_meta["tumor_histologic_subtype"].value_counts(dropna=False)
        df_hist = pd.DataFrame({"histology": [str(x) for x in hist_counts.index], "count": hist_counts.values})
        df_hist.to_csv(self.fig_dir / f"A4_histology_distribution_{self.tier}.csv", index=False)
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.barh(df_hist["histology"][:8], df_hist["count"][:8], color="#008080", edgecolor="black", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(f"A4: Histologic Subtypes (n={len(df_meta)}, Tier: {self.tier})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Cases", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for b in bars:
            w = b.get_width()
            ax.text(w + 3, b.get_y() + b.get_height()/2., f"{int(w)} ({w/len(df_meta)*100:.1f}%)", ha="left", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"A4_histology_distribution_{self.tier}.png", dpi=300)
        fig.savefig(self.fig_dir / f"A4_histology_distribution_{self.tier}.pdf")
        plt.close(fig)

        # 3. Tumour size distribution with TNM strata
        sizes = [m["radiographic_size"] for m in meta if m.get("radiographic_size") is not None]
        df_sizes = pd.DataFrame({"radiographic_size_cm": sizes})
        df_sizes.to_csv(self.fig_dir / f"A2_lesion_size_distribution_{self.tier}.csv", index=False)
        
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        n_bins = 20
        ax.hist(sizes, bins=n_bins, color="#346888", edgecolor="black", alpha=0.8)
        ax.axvline(4.0, color="#d95f02", linestyle="--", linewidth=2, label="TNM Small (<=4 cm)")
        ax.axvline(7.0, color="#7570b3", linestyle="--", linewidth=2, label="TNM Medium (4-7 cm)")
        ax.set_title(f"A2: Radiographic Tumour Size Distribution (n={len(sizes)}, Tier: {self.tier})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tumour Diameter (cm)", fontsize=10)
        ax.set_ylabel("Number of Patients", fontsize=10)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"A2_lesion_size_distribution_{self.tier}.png", dpi=300)
        fig.savefig(self.fig_dir / f"A2_lesion_size_distribution_{self.tier}.pdf")
        plt.close(fig)

        return {
            "num_cases_audited": len(case_dirs),
            "slice_thickness_counts": counts.to_dict(),
            "histology_counts": hist_counts.to_dict()
        }
