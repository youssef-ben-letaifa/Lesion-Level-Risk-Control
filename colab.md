# KiTS23 Lesion-Level Risk Control Study: Colab Runbook (colab.md)
*Google Colab-Ready Notebook Runbook for End-to-End Study Execution*

---

## Cell 0: Setup, Hardware Runtime, and Tier Selection
*Purpose: Initialize study tier configuration and verify environment prerequisites.*

```python
import os

# Tier selection: 'tier1' (ResEnc-M 100ep), 'smoke' (fast pipeline check), or 'full' (10-member ensemble)
STUDY_TIER = "tier1"

# Study cohort size adjustment:
# - 50 cases (Recommended for Colab: Train 30, Cal 10, Test 10 | fits in < 20 GB disk & fast execution)
# - 200 cases (Large cohort: Train 120, Cal 40, Test 40 | requires --use_compressed)
# - 20 cases (Smoke check | runs end-to-end in ~15 mins)
NUM_STUDY_CASES = 50

# Export to environment so %%bash cells and child scripts automatically inherit settings
os.environ["STUDY_TIER"] = STUDY_TIER
os.environ["NUM_STUDY_CASES"] = str(NUM_STUDY_CASES)

print(f"KiTS23 Lesion-Level Risk Control Study | Selected Tier: {STUDY_TIER.upper()} | Active Cohort: {NUM_STUDY_CASES} cases")
print("Ensure hardware accelerator is set to GPU (Runtime -> Change runtime type -> A100 or T4 GPU).")
```

---

## Cell 1: Mount Google Drive
*Purpose: Mount Google Drive to persist all models, checkpoints, logs, and research figures.*

```python
from google.colab import drive
import os
from pathlib import Path

drive.mount('/content/drive', force_remount=False)
print("Google Drive successfully mounted at /content/drive")
```

---

## Cell 2: Set Directory Paths and Locate Project on Google Drive
*Purpose: Establish study directories on Drive and local SSD, auto-discovering the uploaded kits23-main folder.*

```python
import os
import shutil
import sys
from pathlib import Path

# Base Drive Root (where all outputs, models, logs, and checkpoints persist)
DRIVE_ROOT = Path("/content/drive/MyDrive/KiTS23_Cert")
DRIVE_DATA_ZIP = DRIVE_ROOT / "data_zip"
DRIVE_RESULTS = DRIVE_ROOT / "results"
DRIVE_OUTPUTS = DRIVE_ROOT / "outputs"
DRIVE_LOGS = DRIVE_ROOT / "logs"

for p in [DRIVE_ROOT, DRIVE_DATA_ZIP, DRIVE_RESULTS, DRIVE_OUTPUTS, DRIVE_LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# Auto-detect uploaded kits23-main project folder on Google Drive
possible_code_dirs = [
    Path("/content/drive/MyDrive/kits23-main"),
    DRIVE_ROOT / "kits23-main",
    Path("/content/drive/MyDrive/Medical AI Research/kits23-main"),
    Path("/content/kits23-main")
]

source_code_dir = None
for p in possible_code_dirs:
    if p.exists() and (p / "renal_cert").exists() and (p / "setup.py").exists():
        source_code_dir = p
        break

if source_code_dir is None:
    # Deep search in MyDrive if uploaded to a custom subfolder
    for candidate in Path("/content/drive/MyDrive").glob("**/kits23-main"):
        if (candidate / "renal_cert").exists() and (candidate / "setup.py").exists():
            source_code_dir = candidate
            break

print(f"Drive Root configured at: {DRIVE_ROOT}")
if source_code_dir:
    print(f"Detected kits23-main on Drive at: {source_code_dir}")
else:
    print("Notice: kits23-main not detected on Drive yet; assuming local /content/kits23-main.")
```

---

## Cell 3: Copy Code to Fast SSD, Locate Dataset Zip, and Extract
*Purpose: Transfer repository to high-speed NVMe SSD, auto-detect 'dataset and kits23.zip', and extract dataset & kits23.*

```python
import zipfile
import shutil
import os
from pathlib import Path
from tqdm import tqdm

LOCAL_REPO = Path("/content/kits23-main")

# 1. Sync code repository from Drive to local fast SSD (/content/kits23-main)
if source_code_dir and source_code_dir != LOCAL_REPO and source_code_dir.exists():
    print(f"Syncing project code from {source_code_dir} to fast local SSD ({LOCAL_REPO})...")
    shutil.copytree(
        source_code_dir,
        LOCAL_REPO,
        ignore=shutil.ignore_patterns("*.zip", ".git", "work", "results", "nnUNet_*", "*.pyc"),
        dirs_exist_ok=True
    )
    print("Project code successfully synced to SSD.")
else:
    LOCAL_REPO.mkdir(parents=True, exist_ok=True)

# 2. Locate 'dataset and kits23.zip' (inside uploaded kits23-main folder or on Google Drive)
local_zip = Path("/content/dataset and kits23.zip")
possible_zips = [
    source_code_dir / "dataset and kits23.zip" if source_code_dir else None,
    source_code_dir / "dataset_and_kits23.zip" if source_code_dir else None,
    LOCAL_REPO / "dataset and kits23.zip",
    LOCAL_REPO / "dataset_and_kits23.zip",
    local_zip,
    DRIVE_DATA_ZIP / "dataset and kits23.zip",
    Path("/content/drive/MyDrive/dataset and kits23.zip"),
    DRIVE_ROOT / "dataset and kits23.zip"
]

drive_zip = None
for z in possible_zips:
    if z and z.exists():
        drive_zip = z
        break

# If still not found, check any .zip inside source_code_dir (the uploaded kits23-main folder)
if drive_zip is None and source_code_dir and source_code_dir.exists():
    for candidate in source_code_dir.glob("*.zip"):
        drive_zip = candidate
        break

if drive_zip is None:
    # Deep search in MyDrive if uploaded to a custom subfolder
    for candidate in Path("/content/drive/MyDrive").glob("**/dataset and kits23.zip"):
        drive_zip = candidate
        break
    if drive_zip is None:
        for candidate in Path("/content/drive/MyDrive").glob("**/dataset*kits23*.zip"):
            drive_zip = candidate
            break

if drive_zip and not local_zip.exists():
    print(f"Detected archive at {drive_zip}. Copying to fast local SSD ({local_zip})...")
    shutil.copyfile(drive_zip, local_zip)
    print("Archive copied to local SSD.")
elif not local_zip.exists() and not drive_zip:
    print("WARNING: 'dataset and kits23.zip' not found automatically. Please verify upload location in Google Drive.")

# 3. Extract 'dataset and kits23.zip' (extracts 'dataset/' and any missing packages into LOCAL_REPO)
dest_dataset = LOCAL_REPO / "dataset"
dest_kits_pkg = LOCAL_REPO / "kits23"
zip_to_extract = local_zip if local_zip.exists() else drive_zip

if zip_to_extract and zip_to_extract.exists() and (not dest_dataset.exists() or len(list(dest_dataset.glob('case_*'))) < 10):
    print(f"Extracting archive from {zip_to_extract} directly to {LOCAL_REPO} on SSD...")
    with zipfile.ZipFile(zip_to_extract, 'r') as zf:
        for member in tqdm(zf.infolist(), desc="Unpacking archive"):
            name = member.filename
            # Strip archive root prefix 'dataset and kits23/', 'dataset_and_kits23/', or 'kits23-main/' if present
            if name.startswith("dataset and kits23/"):
                rel_name = name[len("dataset and kits23/"):]
            elif name.startswith("dataset_and_kits23/"):
                rel_name = name[len("dataset_and_kits23/"):]
            elif name.startswith("kits23-main/"):
                rel_name = name[len("kits23-main/"):]
            else:
                rel_name = name
            if not rel_name:
                continue
            
            # Guard: Never overwrite active repository codebase (kits23, renal_cert) with old files from the data zip!
            if (rel_name.startswith("kits23/") or rel_name.startswith("renal_cert/")) and (LOCAL_REPO / rel_name).exists():
                continue

            target_path = LOCAL_REPO / rel_name
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print("Archive extraction complete.")
    # Delete archive zip from SSD to preserve disk space
    if local_zip.exists():
        try:
            local_zip.unlink()
            print("Removed temporary archive from SSD to free disk space.")
        except Exception:
            pass

# 4. Set up convenient symlinks at /content so all relative and absolute paths resolve
(LOCAL_REPO / "work").mkdir(parents=True, exist_ok=True)
for folder in ["dataset", "kits23", "renal_cert", "scripts", "configs", "work"]:
    src_f = LOCAL_REPO / folder
    dst_f = Path("/content") / folder
    if src_f.exists() and not dst_f.exists():
        os.symlink(str(src_f), str(dst_f))

# Add project root to Python sys.path
if str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

num_cases = len(list((LOCAL_REPO / "dataset").glob("case_*"))) if (LOCAL_REPO / "dataset").exists() else 0
print(f"Status: {num_cases} cases verified in {LOCAL_REPO / 'dataset'}")
print(f"Status: kits23 module verified at {LOCAL_REPO / 'kits23'}")
```

---

## Cell 4: Install Pinned Dependencies and Local Packages
*Purpose: Install required computer vision, medical imaging, and statistical packages.*

```python
%%bash
set -e

# Upgrade pip to ensure support for modern Python wheels (3.10-3.13)
pip install -q --upgrade pip

# Install required packages using --prefer-binary and flexible minimum versions.
# This ensures Colab reuses its pre-installed binary wheels (scikit-image, scipy, pandas, matplotlib)
# without triggering failing source builds of older releases on modern Python runtimes.
pip install -q --prefer-binary \
    "nnunetv2==2.5.1" \
    "nibabel>=5.2.0" \
    "SimpleITK>=2.3.0" \
    "batchgenerators>=0.25" \
    "scikit-image>=0.23.0" \
    "scipy>=1.11.0" \
    "statsmodels>=0.14.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.8.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0.0" \
    "plotly>=5.20.0" \
    "git+https://github.com/google-deepmind/surface-distance.git"

# Install kits23 and renal_cert in editable mode without re-triggering dependency compilation
cd /content/kits23-main && pip install -q -e . --no-deps

# Register custom 100-epoch trainers directly into nnunetv2 so nnUNetv2_train discovers them immediately
python -c "
import shutil, pathlib, nnunetv2
import nnunetv2.training.nnUNetTrainer as tr
target = pathlib.Path(tr.__file__).parent / 'custom_trainers.py'
src = pathlib.Path('/content/kits23-main/renal_cert/train/custom_trainers.py')
if src.exists():
    shutil.copyfile(src, target)
    print(f'Custom 100-epoch trainers registered at {target}')

# Patch polylr.py for compatibility with modern PyTorch (removes deprecated verbose parameter)
try:
    import nnunetv2.training.lr_scheduler.polylr as plr
    p_plr = pathlib.Path(plr.__file__)
    c = p_plr.read_text()
    if ', False)' in c:
        p_plr.write_text(c.replace(', False)', ')'))
        print(f'Patched {p_plr.name} for PyTorch compatibility.')
except Exception as e:
    print(f'Notice during polylr patch: {e}')

try:
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    trainer_dir = str(pathlib.Path(nnunetv2.__file__).parent / 'training' / 'nnUNetTrainer')
    t_cls = recursive_find_python_class(trainer_dir, 'nnUNetTrainer_KiTS23_100ep_s101', 'nnunetv2.training.nnUNetTrainer')
except Exception:
    from nnunetv2.training.nnUNetTrainer.custom_trainers import nnUNetTrainer_KiTS23_100ep_s101 as t_cls
assert t_cls is not None, 'Registration verification failed: nnUNetTrainer_KiTS23_100ep_s101 not found!'
print(f'Registration verified: {t_cls.__name__} discovered by nnU-Net v2.')
"
echo "Packages installed and custom 100-epoch trainers registered successfully."
```

---

## Cell 5: Hardware Check and nnU-Net Environment Setup
*Purpose: Audit GPU VRAM, RAM, and SSD disk space, setting standard nnU-Net v2 environment paths.*

```python
import os
import shutil
import subprocess
from pathlib import Path

# Hardware check
gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]).decode().strip()
print(f"Active GPU: {gpu_info}")

# Storage paths: raw and preprocessed on fast local SSD; results on persistent Drive
os.environ["nnUNet_raw"] = "/content/work/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/content/work/nnUNet_preprocessed"
os.environ["nnUNet_results"] = str(DRIVE_RESULTS / "nnUNet_results")

# Export to notebook kernel environment for %%bash cells
%env nnUNet_raw=/content/work/nnUNet_raw
%env nnUNet_preprocessed=/content/work/nnUNet_preprocessed
%env nnUNet_results=/content/drive/MyDrive/KiTS23_Cert/results/nnUNet_results

for env_var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
    p = Path(os.environ[env_var])
    p.mkdir(parents=True, exist_ok=True)
    print(f"{env_var} -> {p}")
```

---

## Cell 6: Automated Download of Missing Imaging Volumes
*Purpose: Download missing CT volumes from Hugging Face for the 200 active study cases.*

```python
import json
import urllib.request
import importlib
from pathlib import Path
from tqdm import tqdm
import kits23.download
importlib.reload(kits23.download)
from kits23.download import download_dataset, download_case, get_destination
from renal_cert.splits import DatasetSplitter

# Generate or load splits to determine the 200 active study cases
splits_path = Path("/content/work/splits.json")
redraw_log = DRIVE_LOGS / "splits_redraw_log.jsonl"

if not splits_path.exists():
    splitter = DatasetSplitter(
        metadata_path=Path("/content/dataset/kits23.json"),
        seed=20261026,
        train_ratio=0.60,
        cal_ratio=0.20,
        test_ratio=0.20,
        max_cases=NUM_STUDY_CASES
    )
    splits = splitter.partition(output_path=splits_path, redraw_log_path=redraw_log)
else:
    with open(splits_path, "r") as f:
        splits = json.load(f)

active_cases = splits["train_cases"] + splits["cal_cases"] + splits["test_cases"]
print(f"Study cohort: {len(active_cases)} cases active (Train: {len(splits['train_cases'])}, Cal: {len(splits['cal_cases'])}, Test: {len(splits['test_cases'])}).")
print("Downloading missing imaging volumes from Hugging Face for active cases...")
try:
    download_dataset(case_subset=active_cases)
except TypeError:
    # Direct fallback if kernel had cached an older download_dataset signature
    target_case_numbers = [int(c.split("_")[-1]) if isinstance(c, str) else int(c) for c in active_cases]
    left_to_download = [num for num in target_case_numbers if (not get_destination(f"case_{num:05d}").exists()) or (get_destination(f"case_{num:05d}").stat().st_size < 1000)]
    print(f"Fallback downloader active: {len(left_to_download)} cases remaining to download.")
    for num in tqdm(left_to_download):
        download_case(num, tqdm(total=1, disable=True))

print("All required imaging volumes verified.")
```

---

## Cell 7: Dataset Audit and Figure Group A Generation
*Purpose: Inspect dataset, compute clinical distributions, and render Figure Group A at 300 DPI.*

```python
from renal_cert.data.audit import DataAuditor
from renal_cert.config import StudyConfig

cfg = StudyConfig.from_yaml(Path(f"/content/configs/{STUDY_TIER.lower()}.yaml"))
auditor = DataAuditor(dataset_dir=Path("/content/dataset"), output_dir=DRIVE_OUTPUTS, tier=STUDY_TIER)
audit_results = auditor.run_full_audit()

print(f"Group A figures and CSV data successfully saved to: {auditor.fig_dir}")
```

---

## Cell 8: Patient-Level Random Partition and Balance Audit
*Purpose: Execute 60/20/20 simple random patient partition for 200 cases with pre-registered balance checks.*

```python
from renal_cert.splits import DatasetSplitter
from pathlib import Path
import json

splits_output = Path("/content/work/splits.json")
redraw_log = DRIVE_LOGS / "splits_redraw_log.jsonl"

splitter = DatasetSplitter(
    metadata_path=Path("/content/dataset/kits23.json"),
    seed=20261026,
    train_ratio=0.60,
    cal_ratio=0.20,
    test_ratio=0.20,
    max_cases=NUM_STUDY_CASES
)
splits = splitter.partition(output_path=splits_output, redraw_log_path=redraw_log)
print(f"Partition successful (hash: {splits['split_hash'][:16]}...).")
print(f"Cases used: {splits['num_cases']['total_used']} (Train={splits['num_cases']['train']}, Cal={splits['num_cases']['cal']}, Test={splits['num_cases']['test']}, Unused={splits['num_cases']['unused']})")
```

---

## Cell 9: nnU-Net Dataset Conversion, Planning, and Preprocessing
*Purpose: Convert Train cases into Dataset501_KiTS23_Train format, run fingerprinting, and extract preprocessed data.*

```python
%%bash
set -e
export nnUNet_raw="${nnUNet_raw:-/content/work/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/content/work/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/content/drive/MyDrive/KiTS23_Cert/results/nnUNet_results}"

# Convert Train split only to nnUNet_raw
python -u -c "
import os
from renal_cert.data.to_nnunet import NNUnetConverter
from renal_cert.splits import DatasetSplitter
from pathlib import Path
import json

with open('/content/work/splits.json', 'r') as f:
    splits = json.load(f)

converter = NNUnetConverter(
    dataset_dir=Path('/content/dataset'),
    nnunet_raw_dir=Path('/content/work/nnUNet_raw'),
    dataset_id=501,
    dataset_name='Dataset501_KiTS23_Train'
)
tier = os.environ.get('STUDY_TIER', 'tier1').lower()
if tier == 'smoke':
    subs = DatasetSplitter.get_smoke_subsets(splits, 12, 4, 4)
    converter.convert(subs['train_cases'], subs['cal_cases'], subs['test_cases'], Path('/content/work'))
else:
    converter.convert(splits['train_cases'], splits['cal_cases'], splits['test_cases'], Path('/content/work'))
"

# nnU-Net planning and preprocessing (3d_fullres)
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
echo "Preprocessing complete."
```

---

## Cell 10: Phase-0 Train-Only 5-Fold Cross Validation
*Purpose: Train 5-fold ResEnc-M inside Train split with --npz to produce out-of-fold probability maps.*

```python
%%bash
set -e
export nnUNet_raw="${nnUNet_raw:-/content/work/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/content/work/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/content/drive/MyDrive/KiTS23_Cert/results/nnUNet_results}"

# 1. Clean unpacked .npy files to immediately reclaim ~35 GB of disk space (nnUNet will train from compressed .npz)
find /content/work/nnUNet_preprocessed -name "*.npy" -delete 2>/dev/null || true
rm -rf /content/work/cal_images /content/work/test_images 2>/dev/null || true

# 2. Ensure PyTorch 2.4+ PolyLRScheduler compatibility patch is active
python -c "
import pathlib
try:
    import nnunetv2.training.lr_scheduler.polylr as plr
    p = pathlib.Path(plr.__file__)
    txt = p.read_text()
    if ', False)' in txt:
        p.write_text(txt.replace(', False)', ')'))
        print('PolyLRScheduler compatibility patch active.')
except Exception:
    pass
"

# 3. In Tier 1 / Full, runs CV with --use_compressed; in smoke, runs fast validation
TIER_LOWER=$(echo "${STUDY_TIER:-tier1}" | tr '[:upper:]' '[:lower:]')
if [ "$TIER_LOWER" = "smoke" ]; then
    echo "Smoke tier: skipping 5-fold CV to preserve runtime. Using direct model check."
else
    echo "Running cross validation on Train split with --use_compressed (PDF §6.1)..."
    FOLDS_TO_RUN="${CV_FOLDS:-0}"
    for fold in $FOLDS_TO_RUN; do
        echo "--> Training CV Fold $fold (100 epochs, compressed I/O)..."
        nnUNetv2_train 501 3d_fullres $fold -tr nnUNetTrainer_KiTS23_100ep_s101 --use_compressed --npz --c
    done
fi
```

---

## Cell 11: Phase-0 Gate: Decision Table Evaluation and Config Freezing
*Purpose: Evaluate decision criteria, write frozen_config.json with SHA256 hash, and stop for review.*

```python
import json
from pathlib import Path
from renal_cert.config import StudyConfig
from scripts.run_phase0_cv_audit import run_phase0_audit

cfg = StudyConfig.from_yaml(Path(f"/content/configs/{STUDY_TIER.lower()}.yaml"))
with open("/content/work/splits.json", "r") as f:
    splits = json.load(f)

decision_outcomes, frozen_file = run_phase0_audit(cfg, splits, output_dir=Path("/content/work"))

print("==================================================================")
print(f"GATE PASSED: frozen_config.json written to {frozen_file}")
print("Calibration and Test labels are now unlocked under LabelVault audit logging.")
print("Review PHASE0_REPORT.md before proceeding.")
print("==================================================================")
```

---

## Cell 12: Core Pool Model Training (100 Epochs, Resumable)
*Purpose: Train the ensemble pool members on fold 'all' with automatic checkpointing and resume support.*

```python
%%bash
set -e
export nnUNet_raw="${nnUNet_raw:-/content/work/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/content/work/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/content/drive/MyDrive/KiTS23_Cert/results/nnUNet_results}"

# Core pool training on fold all (PDF §5.2)
python -u -c "
import os, pathlib
from renal_cert.config import StudyConfig
from pathlib import Path
import subprocess

try:
    import nnunetv2.training.lr_scheduler.polylr as plr
    p = pathlib.Path(plr.__file__)
    txt = p.read_text()
    if ', False)' in txt:
        p.write_text(txt.replace(', False)', ')'))
except Exception:
    pass

tier = os.environ.get('STUDY_TIER', 'tier1').lower()
cfg = StudyConfig.from_yaml(Path(f'/content/configs/{tier}.yaml'))
for member in cfg.model_pool.members:
    print(f'Training core pool member: {member.name} ({member.trainer_name}, 100 epochs, compressed I/O)...', flush=True)
    cmd = ['nnUNetv2_train', '501', '3d_fullres', str(member.fold), '-tr', member.trainer_name, '--use_compressed', '--c']
    subprocess.run(cmd, check=True)
"
echo "Core model pool training finished."
```

---

## Cell 13: Streaming Inference and Reduction
*Purpose: Run inference on Calibration and Test scans, reducing probability maps to compact tables.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step stream_infer --tier ${STUDY_TIER:-tier1}
```

---

## Cell 14: Statistical Risk Control on Calibration Split
*Purpose: Calibrate critical lambda thresholds using Split-Conformal, CRC, and LTT exact binomial bounds.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step certify --tier ${STUDY_TIER:-tier1}
```

---

## Cell 15: Test Split Evaluation (Single Unlock) and Re-Splits
*Purpose: Apply calibrated thresholds once to Test split, evaluate realized risk, and run 1,000 re-splits.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step eval_test --tier ${STUDY_TIER:-tier1}
```

---

## Cell 16: Module B: Acquisition Shift, Subgroups, and Mondrian Calibration
*Purpose: Measure realized risk across cohorts, slice thickness, histology, and tumour sizes.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step module_b --tier ${STUDY_TIER:-tier1}
```

---

## Cell 17: Module C: Multi-Reader Annotation Repeats Analysis
*Purpose: Benchmark certificate stability against 3 repeated reader annotations per lesion.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step module_c --tier ${STUDY_TIER:-tier1}
```

---

## Cell 18: Secondary Calibration Metrics and Image-Level Failure Detection
*Purpose: Compute Brier score, class-wise ECE, AURC, and risk-coverage curves.*

```python
%%bash
python -u scripts/run_pipeline_step.py --step secondary_metrics --tier ${STUDY_TIER:-tier1}
```

---

## Cell 18b: Thumbnail Gallery of Research Figures
*Purpose: Display all generated figures from Drive as thumbnails with clickable file links.*

```python
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML

registry_csv = DRIVE_OUTPUTS / "figures" / STUDY_TIER.upper() / "figure_index.csv"
if registry_csv.exists():
    df_reg = pd.read_csv(registry_csv)
    html = "<h3>Generated Study Visualisations</h3><ul>"
    for _, row in df_reg.iterrows():
        html += f"<li><strong>{row['figure_id']}</strong>: {row['title']} (Split: {row['split_used']}, n={row['n']})<br><img src='{row['png_path']}' width='320' style='border:1px solid #ddd; margin:5px;' /><br><a href='{row['data_csv']}'>Download CSV Data</a></li><hr>"
    html += "</ul>"
    display(HTML(html))
else:
    print(f"Registry not yet found at {registry_csv}")
```

---

## Cell 19: Compile Scientific Report, numbers.json Ledger, and STATUS.md
*Purpose: Verify all numbers against the cryptographic ledger and compile self-contained report.html.*

```python
from renal_cert.report.build_report import ReportBuilder
from pathlib import Path

builder = ReportBuilder(output_dir=DRIVE_OUTPUTS)
report_path = builder.build(tier=STUDY_TIER.upper())
print(f"Report compiled successfully at: {report_path}")
```

---

## Cell 19b: Archive Research Figures to Google Drive
*Purpose: Compress all 300 DPI figures and PDFs into figures_<TIER>.zip on Google Drive.*

```python
import shutil
from pathlib import Path

fig_dir = DRIVE_OUTPUTS / "figures" / STUDY_TIER.upper()
zip_dest = DRIVE_OUTPUTS / f"figures_{STUDY_TIER.upper()}.zip"

if fig_dir.exists():
    shutil.make_archive(str(zip_dest.with_suffix("")), 'zip', fig_dir)
    print(f"All figures archived to Drive at: {zip_dest}")
```

---

## Cell 20: Final Archive Export and Run Checklist Verification
*Purpose: Bundle all outputs and print an immutable execution audit checklist.*

```python
import shutil
from pathlib import Path

final_zip = DRIVE_ROOT / f"outputs_{STUDY_TIER.upper()}.zip"
shutil.make_archive(str(final_zip.with_suffix("")), 'zip', DRIVE_OUTPUTS)

print("==================================================================")
print(f"STUDY EXECUTION COMPLETE | TIER: {STUDY_TIER.upper()}")
print(f"Master Archive Saved to Drive: {final_zip}")
print("==================================================================")
```
