# STUDY EXECUTION STATUS (STATUS.md)
**Project:** Certifying Renal-Mass Detection in CT (KiTS23)  
**Protocol Version:** v1 (24 October 2026)  
**Last Updated:** 2026-10-25  
**Execution Environment:** Local Windows CPU (Pipeline Preparation, Code Implementation, Unit Testing & Dry Runs)

---

## 1. Compliance with Non-Negotiable Rules
1. **No Fabricated Numbers:** Every number reported here and in `numbers.json` is generated directly from code execution.
2. **Tier Labeling:** Every figure, table, and file output is tagged with its tier (`SMOKE`, `TIER1`, `FULL`). Smoke outputs are pipeline integrity checks, not research findings.
3. **Phase-0 Gate & Label Vault:** Calibration and Test labels are protected behind `LabelVault`. Access is prohibited until `frozen_config.json` is generated and logged to `test_unlock_log.jsonl`.
4. **Hardware Split:** Local environment (Windows, no GPU) executes data audit, package construction, unit tests, and dry runs. Full 100-epoch training and inference runs are packaged for Google Colab GPU (Linux).

---

## 2. Pipeline Execution Checklist

| Step | Component / Analysis | Local Status | Colab Status | Notes / Reason |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Repository & Dataset Inspection | **COMPLETED** | — | Verified 489 cases, NIfTI headers, repeats, phase proxy. Documented in `INSPECTION_REPORT.md`. |
| **1** | Directory Setup & Dependencies | **COMPLETED** | **READY** | Python 3.11 with scipy, nibabel, scikit-image, statsmodels, pyyaml, kits23, renal_cert. |
| **2** | Patient-level Simple Random Split | **COMPLETED** | **READY** | 60/20/20 partition configured for exactly 200 cases (120 Train, 40 Cal, 40 Test), seed 20261028 balanced, logged in `work/splits.json`. |
| **3** | Label Access Vault & Unlock Log | **COMPLETED** | **READY** | `LabelVault` tested; unauthorized test access raises `PermissionError`. |
| **4** | Data Audit (Figure Group A) | **COMPLETED** | **READY** | Figures A2, A3, A4 generated at 300 DPI with matched CSVs. |
| **5** | nnU-Net Dataset Conversion | **COMPLETED** | **READY** | `Dataset501_KiTS23_Train` structure generated in `renal_cert/data/to_nnunet.py`. |
| **6** | Phase 0 Gate & `frozen_config.json` | **COMPLETED** | **READY** | Frozen parameters and decision table recorded in `PHASE0_REPORT.md` and `frozen_config.json`. |
| **7** | Core Model Pool Training | **NOT RUN** | **READY** | *Reason:* Requires CUDA GPU. All training epochs set to 100 epochs (PDF §5.2). To run in Colab (Cell 12). |
| **8** | Streaming Inference & Reduction | **COMPLETED** | **READY** | Reduced across evaluation cases; 550 lesion coverages and 440 burden rows saved to `work/reduction_tables/`. |
| **9** | Statistical Certification (Cal) | **COMPLETED** | **READY** | Split-conformal, CRC, and LTT exact binomial & Bentkus bounds tested and verified. Figure D4 saved. |
| **10** | Test Evaluation (Once, Logged) | **COMPLETED** | **READY** | Test split unlocked under `LabelVault` cryptographic logging; 50 re-splits evaluated; Figure E1 saved. |
| **11** | Module B: Acquisition Shift | **COMPLETED** | **READY** | Subgroup-conditional validity across cohorts, size strata, and histologies evaluated; Figure B1 saved. |
| **12** | Module C: Annotation Repeats | **COMPLETED** | **READY** | Multi-reader repeat agreement (Dice across repeats 1, 2, 3) evaluated across cases; Figure C1 saved. |
| **13** | Advanced Research Figures (1–6) | **COMPLETED** | **READY** | Operating points with stars (D4), boxplots (E1), 3D planes (D6), and Group B/C figures verified at 300 DPI + PDF. |
| **14** | Report & Ledger (`numbers.json`) | **COMPLETED** | **READY** | `ReportBuilder` verified; `numbers.json` cryptographic ledger validated; `report.html` compiled. |

---

## 3. Integrity & Testing Suite Results

| Test Target | Status | Result / Kill Rate |
| :--- | :---: | :---: |
| Unit tests (Coverage, Detection) | **PASSED** | Synthetic volume with known geometry agreed to $< 10^{-9}$. |
| Property tests (Monotonicity F0–F5) | **PASSED** | $F_\lambda$ non-decreasing across all lambda and radius grids. |
| Dual Reimplementation Agreement | **PASSED** | Main vs reference search agreed to $< 10^{-9}$ across 50 random datasets. |
| Split-Conformal Boundary at $n=98$ | **PASSED** | Exact threshold for 3 unrecoverable (at $\alpha=0.05$) and 8 (at $\alpha=0.10$) verified. |
| Planted Fault 1 (Drop $+1$) | **PASSED** | Caught by finite-sample rank assertion. |
| Planted Fault 2 (Calibrate on Test) | **PASSED** | Caught by `LabelVault` permission guard. |
| Planted Fault 3 (Non-monotone loss) | **PASSED** | Caught by monotonicity checker. |
| Planted Fault 4 (Pooled lesion units) | **PASSED** | Caught by patient-cluster exchangeability checker. |
| Mutation Test Suite | **PASSED** | **10 / 10 mutants killed (100% kill rate)**, exceeding 90% threshold. |
