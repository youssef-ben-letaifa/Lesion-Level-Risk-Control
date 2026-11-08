"""Dry-run script verifying all non-training pipeline steps locally on CPU.

Cites:
    PDF §5.1, §5.3, §5.4, §5.7
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from renal_cert.config import StudyConfig
from renal_cert.splits import DatasetSplitter
from renal_cert.lesions import LesionExtractor
from renal_cert.infer.stream import StreamingReducer
from renal_cert.certify import RiskCertifier
from renal_cert.viz.palette import FigureRegistry
from renal_cert.viz.plots import StudyPlotter
from renal_cert.report.build_report import ReportBuilder


def dry_run():
    root = Path(__file__).resolve().parent.parent
    work_dir = root / "work"
    work_dir.mkdir(exist_ok=True)
    out_dir = root / "outputs" / "SMOKE"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Step 1: Checking Config and Splits ===", flush=True)
    cfg = StudyConfig.from_yaml(root / "configs" / "smoke.yaml")
    splits_file = work_dir / "splits.json"
    if not splits_file.exists():
        splitter = DatasetSplitter(root / "dataset" / "kits23.json")
        splits = splitter.partition(output_path=splits_file)
    else:
        with open(splits_file, "r") as f:
            splits = json.load(f)
            
    print(f"Splits loaded: Train {len(splits['train_cases'])}, Cal {len(splits['cal_cases'])}, Test {len(splits['test_cases'])}", flush=True)

    print("\n=== Step 2: Testing Lesion Extraction on 3 Real Cases ===", flush=True)
    sample_cases = ["case_00001", "case_00002", "case_00003"]
    case_lesions = {}
    for cid in sample_cases:
        case_dir = root / "dataset" / cid
        lesions = LesionExtractor.extract_case_lesions(cid, case_dir)
        case_lesions[cid] = lesions
        print(f"  {cid}: extracted {len(lesions)} lesions. Details:", flush=True)
        for les in lesions:
            print(f"    - {les.lesion_id} ({les.class_type}): {les.volume_ml} mL, diam {les.equiv_diameter_cm} cm, stratum {les.size_stratum}", flush=True)

    print("\n=== Step 3: Testing Streaming Reduction on 3 Real Cases ===", flush=True)
    # Use compact grid for fast CPU dry run verification
    lambda_grid = np.linspace(0.2, 0.95, 6)
    radius_grid = np.linspace(0.0, 10.0, 5)
    reducer = StreamingReducer(lambda_grid=lambda_grid, radius_grid=radius_grid)

    reduced_coverages = []
    reduced_burdens = []
    reduced_metrics = []

    for cid in sample_cases:
        seg_path = root / "dataset" / cid / "segmentation.nii.gz"
        img = nib.load(str(seg_path))
        gt_seg = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        
        # Subsample for blazing fast CPU dry run verification
        gt_seg_sub = gt_seg[::4, ::4, ::4].copy()
        spacing_sub = (spacing[0] * 4.0, spacing[1] * 4.0, spacing[2] * 4.0)
        
        # Synthesize 2 ensemble members with realistic mass probability signal
        # member 0: gt mass mask with slight noise
        gt_mass = (gt_seg_sub == 2) | (gt_seg_sub == 3)
        rng = np.random.RandomState(42)
        m0 = np.zeros_like(gt_seg_sub, dtype=np.float32)
        m0[gt_mass] = rng.uniform(0.7, 0.99, size=np.count_nonzero(gt_mass))
        m0[~gt_mass] = rng.exponential(0.01, size=np.count_nonzero(~gt_mass)).clip(0, 0.3)
        
        m1 = np.zeros_like(gt_seg_sub, dtype=np.float32)
        m1[gt_mass] = rng.uniform(0.6, 0.95, size=np.count_nonzero(gt_mass))
        m1[~gt_mass] = rng.exponential(0.015, size=np.count_nonzero(~gt_mass)).clip(0, 0.4)
        
        member_probs = np.stack([m0, m1], axis=0)
        
        # Subsampled lesions for dry run
        sub_lesions = LesionExtractor.extract_from_segmentation_26cc(cid, gt_seg_sub, spacing_sub)
        
        red = reducer.reduce_case(cid, member_probs, sub_lesions, gt_seg_sub, spacing_sub)
        reduced_coverages.append(red["lesion_coverage"])
        reduced_burdens.append(red["patient_burden"])
        reduced_metrics.append(red["case_metrics"])
        print(f"  {cid}: reduced to {len(red['lesion_coverage'])} coverage rows, {len(red['patient_burden'])} burden rows.", flush=True)

    df_cov_all = pd.concat(reduced_coverages, ignore_index=True)
    df_bur_all = pd.concat(reduced_burdens, ignore_index=True)
    print(f"Total reduced burden records: {len(df_bur_all)}", flush=True)

    print("\n=== Step 4: Testing Risk Certification ===", flush=True)
    certifier = RiskCertifier(lambda_grid=lambda_grid)
    dummy_fp = df_bur_all.groupby("param_idx")["fp_lesions"].mean().values
    
    # Patient critical lambdas for F1
    f1_burden = df_bur_all[df_bur_all["family"] == "F1"]
    crit_lambdas = []
    for cid, c_df in f1_burden.groupby("case_id"):
        zero_miss = c_df[c_df["patient_miss_05"] == 0]
        if len(zero_miss) > 0:
            crit_lambdas.append(float(zero_miss["param_value"].min()))
        else:
            crit_lambdas.append(np.inf)
            
    res_split = certifier.certify_split_conformal(np.array(crit_lambdas), alpha=0.10, cal_fp_per_lambda=dummy_fp)
    print(f"Split-conformal F1 alpha=0.10 result: feasible={res_split.is_feasible}, lambda*={res_split.lambda_star}, emp_risk={res_split.cal_empirical_risk}", flush=True)

    print("\n=== Step 5: Testing Visualisations and Figures ===", flush=True)
    registry = FigureRegistry(out_dir / "figure_index.csv")
    plotter = StudyPlotter(registry, tier="SMOKE")
    
    # 1. Operating points frontier
    frontier_data = []
    for fam in ["F0", "F1", "F2", "F3", "F4", "F5"]:
        for idx, l_val in enumerate(lambda_grid[:15]):
            frontier_data.append({
                "family": fam,
                "param_value": l_val,
                "mean_risk": max(0.01, 0.20 - 0.015 * idx),
                "mean_fp": 0.2 + 0.15 * idx
            })
    df_front = pd.DataFrame(frontier_data)
    certified_pts = [
        {"family": "F1", "alpha": 0.10, "lambda_star": 0.85, "risk": 0.08, "fp": 1.1},
        {"family": "F2", "alpha": 0.05, "lambda_star": 0.92, "risk": 0.04, "fp": 1.6}
    ]
    p1 = plotter.plot_operating_points_frontier(df_front, certified_pts, out_dir / "figures", n_patients=3)
    print(f"  Generated frontier plot: {p1}", flush=True)

    # 2. Resplits boxplots
    resplits_data = []
    for s in range(50):
        for fam in ["F0", "F1", "F2", "F3", "F4", "F5"]:
            resplits_data.append({
                "split_id": s,
                "family": fam,
                "realized_risk": max(0.0, np.random.normal(0.08, 0.02)),
                "fp_lesions": max(0.0, np.random.normal(1.2, 0.3)),
                "flagged_vol_ml": 5.0
            })
    df_resp = pd.DataFrame(resplits_data)
    p2 = plotter.plot_boxplots_resplits(df_resp, out_dir / "figures", alpha=0.10)
    print(f"  Generated resplits boxplot: {p2}", flush=True)

    # 3. 3D Cluster planes
    p3d_data = []
    for i, cid in enumerate(sample_cases):
        p3d_data.append({
            "case_id": cid,
            "crit_f1": 0.6 + 0.1 * i,
            "crit_f2": 0.5 + 0.1 * i,
            "tumor_size_cm": 3.0 + 1.5 * i,
            "cohort": "KiTS21" if i < 2 else "KiTS23"
        })
    df_p3d = pd.DataFrame(p3d_data)
    p3_png, p3_html = plotter.plot_3d_cluster_planes(df_p3d, 0.85, 0.75, out_dir / "figures")
    print(f"  Generated 3D cluster plot: {p3_png} and {p3_html}", flush=True)

    print("\n=== Step 6: Testing Report Builder and numbers.json Ledger ===", flush=True)
    builder = ReportBuilder(output_dir=out_dir)
    builder.update_numbers_ledger({
        "F1_split_conformal_a10_risk": 0.08,
        "F2_split_conformal_a5_risk": 0.04,
        "baseline_hec2_dice": 0.82
    })
    
    cert_table = [
        {"family": "F1", "method": "split_conformal", "alpha": 0.10, "is_feasible": True, "lambda_star": 0.85, "risk": 0.08, "cal_mean_fp_lesions": 1.1, "is_budgeted_feasible": True},
        {"family": "F2", "method": "split_conformal", "alpha": 0.05, "is_feasible": True, "lambda_star": 0.92, "risk": 0.04, "cal_mean_fp_lesions": 1.6, "is_budgeted_feasible": True}
    ]
    report_html = builder.build_html_report(
        tier="SMOKE",
        cert_results_table=cert_table,
        baseline_metrics={"dice_hec2": 0.82, "dice_hec3": 0.78},
        audit_summary={"num_cases": 489, "n_train": 293, "n_cal": 98, "n_test": 98},
        figures_list=[
            {"title": "D4: Risk-Burden Frontier", "description": "Operating points and certified thresholds", "img_src": str(p1.name)},
            {"title": "E1: Re-splits Boxplots", "description": "Realized risk distribution", "img_src": str(p2.name)}
        ]
    )
    print(f"  Generated HTML report: {report_html}", flush=True)
    print("\nALL DRY RUN STEPS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    dry_run()
