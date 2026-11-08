# Contributing Guidelines

Thank you for your interest in contributing to the **KiTS23 Lesion-Level Risk Certification** benchmark!

This repository implements a pre-registered, distribution-free risk certification protocol for volumetric Computed Tomography (CT) segmentation under the KiTS23 challenge framework.

---

## Code of Conduct & Research Integrity

All contributors are expected to uphold the highest standards of scientific reproducibility and computational integrity:
1. **Never Bypass Cryptographic Label Vaults:** Ground-truth labels on the Calibration and Test splits are firewalled (`renal_cert/label_vault.py`). Calibrating or tuning model weights on test data invalidates statistical exchangeability and constitutes methodological failure.
2. **Deterministic Reproducibility:** Ensure all experimental scripts accept explicit random seeds (`--seed 101`). Seeded runs must produce bitwise reproducible results.
3. **Preserve Documentation Integrity:** When submitting pull requests, do not overwrite or modify pre-registered protocol logs (`frozen_config.json`, `docs/reports/PHASE0_REPORT.md`).

---

## Development Workflow

### 1. Environment Setup
```bash
git clone https://github.com/youssef-ben-letaifa/Lesion-Level-Risk-Control.git
cd Lesion-Level-Risk-Control

python -m venv venv
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -e ./kits23
```

### 2. Running Test Suites
Before submitting changes, ensure all automated unit tests, property checks, and planted-fault detectors pass:
```bash
pytest tests/ -v
```

### 3. Adding New Prediction Families
To propose a new prediction family:
1. Implement the family set function in `renal_cert/families.py`, ensuring monotonicity in the set parameter $\lambda$.
2. Add corresponding property assertions in `tests/test_coverage_detection.py`.
3. Evaluate the review burden (FP lesions / patient) against the Pareto frontier in `scripts/generate_publication_figures.py`.

---

## Submitting Pull Requests

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/new-calibration-bound
   ```
2. Commit changes with clear, descriptive messages adhering to conventional commit format.
3. Push to your branch and open a Pull Request using `.github/pull_request_template.md`.

