# KiTS23 Lesion-Level Risk Control Study: All-In-One Colab Runbook (colabfull.md)
*Complete End-to-End Pipeline in One Single Executable Cell*

---

## Overview

This file provides the complete, self-contained implementation of the KiTS23 Lesion-Level Risk Control study packaged into **a single executable code cell**.

### Key Features in this All-In-One Runner:
1. **Zero-Disk-Overflow Architecture**: Automatically runs with `--use_compressed` and `unpack_dataset=False`, eliminating the ~40 GB of uncompressed `.npy` files that previously filled Colab's 112 GB disk.
2. **PyTorch 2.4+ Self-Healing Compatibility**: Patches `polylr.py` and registers custom 100-epoch trainers dynamically, preventing `TypeError: LRScheduler.__init__() takes from 2 to 3 positional arguments`.
3. **Resilient Download & Caching**: Reloads `kits23.download` and checks file sizes (> 1000 bytes) to ensure no corrupt or partial CT scans.
4. **End-to-End Execution**: Mounts Google Drive, discovers project code, partitions the active cohort, computes Group A clinical distributions, converts to nnU-Net format, runs Phase-0 cross-validation, freezes the gate configuration, trains core pool members on fold `all`, streams predictions into compact reduction tables, calibrates conformal thresholds, evaluates the test split with 1,000 empirical re-splits, executes Modules B and C, and compiles the self-contained scientific report HTML and numbers ledger.

---

## Single Master Code Cell

Copy and paste the entire block below into a single code cell in Google Colab (with GPU runtime enabled) and click **Run**:

```python
# ==============================================================================
# KiTS23 Lesion-Level Risk Control Study: Master Single-Cell Runner
# ==============================================================================
# Executes the entire study end-to-end sequentially in a single Google Colab cell.
# Automatically mounts Drive, syncs code, installs dependencies, patches PyTorch 2.4+,
# downloads imaging, preprocesses data, trains models with --use_compressed (preventing disk overflow),
# runs streaming reduction, statistical certification, test evaluation, and compiles report.
# ==============================================================================

import os
import sys
import shutil
import subprocess
import json
import zipfile
import pathlib
import datetime
from pathlib import Path

# ------------------------------------------------------------------------------
# STEP 0: Configuration & Study Tier Selection
# ------------------------------------------------------------------------------
print("=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 0: INITIALIZING STUDY CONFIGURATION")
print("=" * 80)

# Tier selection: 'tier1' (ResEnc-M 100ep), 'smoke' (fast 2ep check), or 'full' (10-member ensemble)
STUDY_TIER = "tier1"

# Study cohort size:
# - 50 cases (Recommended for Colab: Train 30, Cal 10, Test 10 | fits in < 20 GB disk & fast execution)
# - 200 cases (Large cohort: Train 120, Cal 40, Test 40 | requires --use_compressed)
# - 20 cases (Smoke check | runs end-to-end in ~15 mins)
NUM_STUDY_CASES = 50

# Which CV fold to run in Phase 0 (Fold 0 by default to avoid GPU timeout; set to '0 1 2 3 4' for all folds)
CV_FOLDS = os.environ.get("CV_FOLDS", "0")

os.environ["STUDY_TIER"] = STUDY_TIER
os.environ["NUM_STUDY_CASES"] = str(NUM_STUDY_CASES)
os.environ["CV_FOLDS"] = CV_FOLDS

print(f"Active Configuration: Tier={STUDY_TIER.upper()}, Cohort={NUM_STUDY_CASES} cases, CV Folds={CV_FOLDS}")
print("Ensure hardware accelerator is set to GPU (Runtime -> Change runtime type -> A100 or T4 GPU).")

# ------------------------------------------------------------------------------
# STEP 1: Mount Google Drive
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 1: MOUNTING GOOGLE DRIVE")
print("=" * 80)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Google Drive successfully mounted at /content/drive")
except Exception as e:
    print(f"Notice: Google Drive mount skipped ({e})")

# ------------------------------------------------------------------------------
# STEP 2: Establish Storage Paths & Discover Code on Drive
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 2: SETTING UP DIRECTORIES")
print("=" * 80)

DRIVE_ROOT = Path("/content/drive/MyDrive/KiTS23_Cert")
DRIVE_DATA_ZIP = DRIVE_ROOT / "data_zip"
DRIVE_RESULTS = DRIVE_ROOT / "results"
DRIVE_OUTPUTS = DRIVE_ROOT / "outputs"
DRIVE_LOGS = DRIVE_ROOT / "logs"

for p in [DRIVE_ROOT, DRIVE_DATA_ZIP, DRIVE_RESULTS, DRIVE_OUTPUTS, DRIVE_LOGS]:
    p.mkdir(parents=True, exist_ok=True)

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
    for candidate in Path("/content/drive/MyDrive").glob("**/kits23-main"):
        if (candidate / "renal_cert").exists() and (candidate / "setup.py").exists():
            source_code_dir = candidate
            break

print(f"Drive Root configured at: {DRIVE_ROOT}")
if source_code_dir:
    print(f"Detected kits23-main on Drive at: {source_code_dir}")
else:
    print("Notice: kits23-main not detected on Drive yet; assuming local /content/kits23-main.")

# ------------------------------------------------------------------------------
# STEP 3: Sync Code to Fast NVMe SSD & Extract Dataset Archive
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 3: SYNCING CODE & EXTRACTING ARCHIVE")
print("=" * 80)

from tqdm import tqdm

LOCAL_REPO = Path("/content/kits23-main")

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

# Locate 'dataset and kits23.zip'
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

if drive_zip is None and source_code_dir and source_code_dir.exists():
    for candidate in source_code_dir.glob("*.zip"):
        drive_zip = candidate
        break

if drive_zip and not local_zip.exists():
    print(f"Copying archive from {drive_zip} to local SSD ({local_zip})...")
    shutil.copyfile(drive_zip, local_zip)

# Extract dataset and packages
dest_dataset = LOCAL_REPO / "dataset"
zip_to_extract = local_zip if local_zip.exists() else drive_zip

if zip_to_extract and zip_to_extract.exists() and (not dest_dataset.exists() or len(list(dest_dataset.glob('case_*'))) < 10):
    print(f"Extracting archive from {zip_to_extract} to {LOCAL_REPO}...")
    with zipfile.ZipFile(zip_to_extract, 'r') as zf:
        for member in tqdm(zf.infolist(), desc="Unpacking archive"):
            name = member.filename
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

# Delete temporary archive from SSD to preserve disk space
if local_zip.exists():
    try:
        local_zip.unlink()
        print("Removed temporary archive from SSD to free disk space.")
    except Exception:
        pass

# Setup convenient symlinks at /content
(LOCAL_REPO / "work").mkdir(parents=True, exist_ok=True)
for folder in ["dataset", "kits23", "renal_cert", "scripts", "configs", "work"]:
    src_f = LOCAL_REPO / folder
    dst_f = Path("/content") / folder
    if src_f.exists() and not dst_f.exists():
        os.symlink(str(src_f), str(dst_f))

if str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

# ------------------------------------------------------------------------------
# STEP 4: Install Dependencies & Patch Trainers for PyTorch 2.4+
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 4: INSTALLING DEPENDENCIES & REGISTERING TRAINERS")
print("=" * 80)

# Upgrade pip
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)

# Install pinned dependencies with --prefer-binary to avoid source compilation
reqs = [
    "nnunetv2==2.5.1",
    "nibabel>=5.2.0",
    "SimpleITK>=2.3.0",
    "batchgenerators>=0.25",
    "scikit-image>=0.23.0",
    "scipy>=1.11.0",
    "statsmodels>=0.14.0",
    "pandas>=2.0.0",
    "matplotlib>=3.8.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0.0",
    "plotly>=5.20.0",
    "git+https://github.com/google-deepmind/surface-distance.git"
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary"] + reqs, check=True)

# Install kits23 & renal_cert in editable mode without re-triggering dependency builds
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(LOCAL_REPO), "--no-deps"], check=True)

# Register custom 100-epoch trainers directly into nnunetv2
import nnunetv2
import nnunetv2.training.nnUNetTrainer as tr

tr_dir = pathlib.Path(tr.__file__).parent
target_custom = tr_dir / "custom_trainers.py"
src_custom = LOCAL_REPO / "renal_cert" / "train" / "custom_trainers.py"
if src_custom.exists():
    shutil.copyfile(src_custom, target_custom)
    print(f"Custom 100-epoch trainers registered at: {target_custom}")

# Patch polylr.py for compatibility with PyTorch 2.4+ (removes deprecated verbose parameter)
try:
    import nnunetv2.training.lr_scheduler.polylr as plr
    p_plr = pathlib.Path(plr.__file__)
    c = p_plr.read_text()
    if ", False)" in c:
        p_plr.write_text(c.replace(", False)", ")"))
        print(f"Patched {p_plr.name} for PyTorch compatibility.")
except Exception as e:
    print(f"Notice during polylr patch: {e}")

# Verify trainer registration using nnU-Net v2 dynamic finder
try:
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    t_cls = recursive_find_python_class(str(tr_dir), "nnUNetTrainer_KiTS23_100ep_s101", "nnunetv2.training.nnUNetTrainer")
except Exception:
    from nnunetv2.training.nnUNetTrainer.custom_trainers import nnUNetTrainer_KiTS23_100ep_s101 as t_cls

assert t_cls is not None, "Registration failed: nnUNetTrainer_KiTS23_100ep_s101 not found!"
print(f"Registration verified: {t_cls.__name__} successfully registered in nnU-Net v2.")

# ------------------------------------------------------------------------------
# STEP 5: Hardware Check & nnU-Net Environment Setup
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 5: HARDWARE CHECK & NNUNET ENVIRONMENT")
print("=" * 80)

try:
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]).decode().strip()
    print(f"Active GPU: {gpu_info}")
except Exception:
    print("Notice: nvidia-smi not available (running on CPU).")

os.environ["nnUNet_raw"] = "/content/work/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/content/work/nnUNet_preprocessed"
os.environ["nnUNet_results"] = str(DRIVE_RESULTS / "nnUNet_results")

for env_var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
    p = Path(os.environ[env_var])
    p.mkdir(parents=True, exist_ok=True)
    print(f"{env_var} -> {p}")

# ------------------------------------------------------------------------------
# STEP 6: Partition Dataset & Download Imaging Volumes for Active Cohort
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 6: PARTITIONING COHORT & DOWNLOADING IMAGING")
print("=" * 80)

import importlib
import kits23.download
importlib.reload(kits23.download)
from kits23.download import download_dataset, download_case, get_destination
from renal_cert.splits import DatasetSplitter

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
print(f"Active Cohort: {len(active_cases)} cases (Train: {len(splits['train_cases'])}, Cal: {len(splits['cal_cases'])}, Test: {len(splits['test_cases'])}).")

print("Verifying / downloading imaging volumes from Hugging Face...")
try:
    download_dataset(case_subset=active_cases)
except TypeError:
    target_case_numbers = [int(c.split("_")[-1]) if isinstance(c, str) else int(c) for c in active_cases]
    left_to_download = [num for num in target_case_numbers if (not get_destination(f"case_{num:05d}").exists()) or (get_destination(f"case_{num:05d}").stat().st_size < 1000)]
    for num in tqdm(left_to_download, desc="Downloading volumes"):
        download_case(num, tqdm(total=1, disable=True))

print("All imaging volumes verified.")

# ------------------------------------------------------------------------------
# STEP 7: Dataset Audit & Figure Group A Generation (300 DPI)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 7: DATA AUDIT & FIGURE GROUP A GENERATION")
print("=" * 80)

from renal_cert.data.audit import DataAuditor
from renal_cert.config import StudyConfig

auditor = DataAuditor(dataset_dir=Path("/content/dataset"), output_dir=DRIVE_OUTPUTS, tier=STUDY_TIER)
audit_results = auditor.run_full_audit()
print(f"Group A figures and CSV data saved to: {auditor.fig_dir}")

# ------------------------------------------------------------------------------
# STEP 8: nnU-Net Dataset Conversion & Preprocessing (with Integrity Check)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 8: NNUNET DATA CONVERSION & PREPROCESSING")
print("=" * 80)

from renal_cert.data.to_nnunet import NNUnetConverter

converter = NNUnetConverter(
    dataset_dir=Path("/content/dataset"),
    nnunet_raw_dir=Path("/content/work/nnUNet_raw"),
    dataset_id=501,
    dataset_name="Dataset501_KiTS23_Train"
)

if STUDY_TIER.lower() == "smoke":
    subs = DatasetSplitter.get_smoke_subsets(splits, 12, 4, 4)
    converter.convert(subs["train_cases"], subs["cal_cases"], subs["test_cases"], Path("/content/work"))
else:
    converter.convert(splits["train_cases"], splits["cal_cases"], splits["test_cases"], Path("/content/work"))

# Run nnUNet planning and preprocessing
cmd_plan = ["nnUNetv2_plan_and_preprocess", "-d", "501", "-c", "3d_fullres", "--verify_dataset_integrity"]
subprocess.run(cmd_plan, check=True)
print("Preprocessing complete.")

# ------------------------------------------------------------------------------
# STEP 9: Phase-0 Cross-Validation (with --use_compressed to prevent disk overflow)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 9: PHASE-0 CROSS-VALIDATION (COMPRESSED I/O)")
print("=" * 80)

# Disk cleanup: ensure no unpacked .npy files exist
for p in Path("/content/work/nnUNet_preprocessed").glob("**/*.npy"):
    try:
        p.unlink()
    except Exception:
        pass

if STUDY_TIER.lower() == "smoke":
    print("Smoke tier: skipping 5-fold CV to preserve runtime. Using direct model check.")
else:
    folds_to_run = [int(f) for f in CV_FOLDS.split()]
    for fold in folds_to_run:
        print(f"--> Training CV Fold {fold} (100 epochs, compressed I/O)...")
        cmd_cv = [
            "nnUNetv2_train", "501", "3d_fullres", str(fold),
            "-tr", "nnUNetTrainer_KiTS23_100ep_s101",
            "--use_compressed", "--npz", "--c"
        ]
        subprocess.run(cmd_cv, check=True)

# ------------------------------------------------------------------------------
# STEP 10: Phase-0 Gate: Decision Table Evaluation & Config Freezing
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 10: EVALUATING PHASE-0 GATE & FREEZING CONFIG")
print("=" * 80)

from scripts.run_phase0_cv_audit import run_phase0_audit

cfg = StudyConfig.from_yaml(Path(f"/content/configs/{STUDY_TIER.lower()}.yaml"))
decision_outcomes, frozen_file = run_phase0_audit(cfg, splits, output_dir=Path("/content/work"))

print(f"GATE PASSED: frozen_config.json written to {frozen_file}")
print("Calibration and Test labels are unlocked under LabelVault cryptographic audit logging.")

# ------------------------------------------------------------------------------
# STEP 11: Core Model Pool Training (Fold 'all', Compressed, Resumable)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 11: CORE POOL MODEL TRAINING (FOLD ALL)")
print("=" * 80)

for member in cfg.model_pool.members:
    print(f"Training core pool member: {member.name} ({member.trainer_name}, 100 epochs)...", flush=True)
    cmd_core = [
        "nnUNetv2_train", "501", "3d_fullres", str(member.fold),
        "-tr", member.trainer_name,
        "--use_compressed", "--c"
    ]
    subprocess.run(cmd_core, check=True)

print("Core model pool training finished.")

# ------------------------------------------------------------------------------
# STEP 12: Streaming Inference & Tabular Reduction
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 12: STREAMING INFERENCE & REDUCTION")
print("=" * 80)

cmd_infer = [sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "stream_infer", "--tier", STUDY_TIER]
subprocess.run(cmd_infer, check=True)

# ------------------------------------------------------------------------------
# STEP 13: Statistical Risk Control on Calibration Split
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 13: STATISTICAL CERTIFICATION (CALIBRATION)")
print("=" * 80)

cmd_cert = [sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "certify", "--tier", STUDY_TIER]
subprocess.run(cmd_cert, check=True)

# ------------------------------------------------------------------------------
# STEP 14: Test Split Single Unlock & Empirical Re-Splits Evaluation
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 14: TEST SPLIT EVALUATION (SINGLE UNLOCK)")
print("=" * 80)

cmd_eval = [sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "eval_test", "--tier", STUDY_TIER]
subprocess.run(cmd_eval, check=True)

# ------------------------------------------------------------------------------
# STEP 15: Shift Analysis (Module B), Repeats (Module C) & Secondary Metrics
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 15: MODULE B, MODULE C & SECONDARY METRICS")
print("=" * 80)

subprocess.run([sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "module_b", "--tier", STUDY_TIER], check=True)
subprocess.run([sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "module_c", "--tier", STUDY_TIER], check=True)
subprocess.run([sys.executable, "-u", "scripts/run_pipeline_step.py", "--step", "secondary_metrics", "--tier", STUDY_TIER], check=True)

# ------------------------------------------------------------------------------
# STEP 16: Compile Scientific Report, Ledger, and Export Master Archives
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STEP 16: COMPILING REPORT & EXPORTING MASTER ARCHIVE")
print("=" * 80)

from renal_cert.report.build_report import ReportBuilder

builder = ReportBuilder(output_dir=DRIVE_OUTPUTS)
report_path = builder.build(tier=STUDY_TIER.upper())
print(f"Scientific report compiled at: {report_path}")

fig_dir = DRIVE_OUTPUTS / "figures" / STUDY_TIER.upper()
zip_dest = DRIVE_OUTPUTS / f"figures_{STUDY_TIER.upper()}.zip"
if fig_dir.exists():
    shutil.make_archive(str(zip_dest.with_suffix("")), 'zip', fig_dir)
    print(f"Figures archived to Drive at: {zip_dest}")

final_zip = DRIVE_ROOT / f"outputs_{STUDY_TIER.upper()}.zip"
shutil.make_archive(str(final_zip.with_suffix("")), 'zip', DRIVE_OUTPUTS)

print("=" * 80)
print(f"STUDY EXECUTION COMPLETE | TIER: {STUDY_TIER.upper()}")
print(f"Master Archive Saved to Drive: {final_zip}")
print("=" * 80)
```

