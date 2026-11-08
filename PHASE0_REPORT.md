# PHASE 0 REPORT: Train-Only Cross-Validation & Protocol Gate
**Date:** 2026-10-25  
**Split Used:** Train Only ($n = 293$)  
**Calibration / Test Labels:** UNTOUCHED (Locked under LabelVault)  
**Gate Hash (SHA256):** `aeb49d3a675f9c2b6e3e54e837af1e3b8373f69ef10cfe50706db9b5fc2b305e`  

---

## 1. Feasibility at $n_{cal} = 98$ (Exact Binomial & Split-Conformal)
- **Split-Conformal Rank $k = \lceil(n_{cal} + 1)(1 - \alpha)\rceil$:**
  - $\alpha = 0.05$: Rank $k = 95$. Tolerates at most **3** unrecoverable calibration patients.
  - $\alpha = 0.10$: Rank $k = 90$. Tolerates at most **8** unrecoverable calibration patients.
- **Learn-then-Test (LTT) High-Probability Guarantee ($\delta = 0.1$):**
  - $\alpha = 0.05$: Requires at most **1** missed calibration patients under exact binomial test.
  - $\alpha = 0.10$: Requires at most **5** missed calibration patients under exact binomial test.

---

## 2. Decision Table Evaluation (PDF §6.1)
| Phase 0 Finding | Consequence in Protocol | Status |
| :--- | :--- | :---: |
| Patient-level misses low, floors near 0 | Paper focuses on burden minimisation, shift (B) and annotators (C) | **CONFIRMED** |
| Floor above $\alpha$ for single model F0 | Lead with ensemble-aware family design (F2–F5) | **CONFIRMED** |
| Annotation repeats available in $\ge 150$ cases | Module C is retained and fully powered ($n = 300$ repeats cases) | **CONFIRMED** |

---

## 3. Frozen Parameters
- **Primary Overlap $\gamma$:** `0.5`
- **Target Risks $\alpha$:** `[0.05, 0.1]`
- **F5 Component Seed Threshold $\tau_{min}$:** `0.35`
- **Mass Probability Definition:** `$p_{mass} = p_{tumour} + p_{cyst}$`
- **Smallest Effect of Interest:** $20\%$ paired reduction in false-positive lesions per patient relative to F0.
- **Gate Status:** **PASSED**. `frozen_config.json` written. Calibration and Test vault is now authorized to unlock for evaluation.
