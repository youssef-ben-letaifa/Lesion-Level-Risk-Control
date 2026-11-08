"""KiTS23 Segmentation Testing Suite & Risk-Controlled Clinical Viewer.

Discovers trained nnU-Net v2 checkpoints (.pth) in 'Training Result' and 'models',
and provides an interactive medical web interface to test, visualize, and verify
segmentations, conformal risk thresholds, and tumor volumetric metrics.
"""

import os
import io
import json
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from scipy.ndimage import label, generate_binary_structure

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Workspace paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_RESULT_DIR = os.path.join(BASE_DIR, "Training Result")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions_s101")

# Global in-memory cache for fast slice rendering
_CASE_CACHE = {}
_MODEL_INFO_CACHE = {}


def discover_models():
    """Scans Training Result and models/ for all .pth checkpoints."""
    models_found = []
    seen = set()

    search_dirs = [MODELS_DIR, TRAINING_RESULT_DIR]
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, dirs, files in os.walk(sdir):
            for f in files:
                if f.endswith(".pth"):
                    fp = os.path.join(root, f)
                    try:
                        real_fp = os.path.realpath(fp)
                    except Exception:
                        real_fp = os.path.normpath(fp)
                    if real_fp in seen:
                        continue
                    seen.add(real_fp)

                    sz_mb = round(os.path.getsize(fp) / (1024 * 1024), 2)
                    rel = os.path.relpath(fp, BASE_DIR)

                    # Categorize model
                    name = f
                    desc = "nnU-Net Checkpoint"
                    fold = "unknown"
                    seed = "101"

                    if "fold_all" in fp:
                        fold = "All (100% Train Split)"
                    elif "fold_0" in fp:
                        fold = "Fold 0 (Cross-Validation)"

                    if "s102" in fp:
                        seed = "102"
                    elif "s101" in fp:
                        seed = "101"

                    if "checkpoint_final" in f:
                        desc = f"Production Final 100-Epoch (Seed {seed})"
                    elif "checkpoint_best" in f:
                        desc = f"Best EMA Validation Checkpoint (Seed {seed})"

                    models_found.append({
                        "id": len(models_found) + 1,
                        "filename": f,
                        "path": fp,
                        "rel_path": rel,
                        "size_mb": sz_mb,
                        "fold": fold,
                        "seed": seed,
                        "description": desc,
                        "is_default": "checkpoint_final.pth" in f and "fold_all" in fp and "s101" in fp
                    })

    # Sort so default final production model is first
    models_found.sort(key=lambda x: (not x["is_default"], x["seed"], x["fold"]))
    return models_found


def discover_cases():
    """Discovers available test cases with precomputed predictions and dataset volumes."""
    cases_dict = {}

    # 1. Cases with predictions in predictions_s101
    if os.path.exists(PREDICTIONS_DIR):
        for f in os.listdir(PREDICTIONS_DIR):
            if f.endswith(".nii.gz") and f.startswith("case_"):
                cid = f.replace(".nii.gz", "")
                cases_dict[cid] = {
                    "case_id": cid,
                    "has_prediction": True,
                    "pred_path": os.path.join(PREDICTIONS_DIR, f),
                    "has_gt": os.path.exists(os.path.join(DATASET_DIR, cid, "segmentation.nii.gz")),
                    "has_imaging": os.path.exists(os.path.join(DATASET_DIR, cid, "imaging.nii.gz")),
                    "source": "Tier-1 Fold All Prediction"
                }

    # 2. Check fold_0 validation in models
    f0_dir = os.path.join(MODELS_DIR, "nnUNetTrainer_KiTS23_100ep_s101__nnUNetPlans__3d_fullres", "fold_0", "validation")
    if os.path.exists(f0_dir):
        for f in os.listdir(f0_dir):
            if f.endswith(".nii.gz") and f.startswith("case_"):
                cid = f.replace(".nii.gz", "")
                if cid not in cases_dict:
                    cases_dict[cid] = {
                        "case_id": cid,
                        "has_prediction": True,
                        "pred_path": os.path.join(f0_dir, f),
                        "has_gt": os.path.exists(os.path.join(DATASET_DIR, cid, "segmentation.nii.gz")),
                        "has_imaging": os.path.exists(os.path.join(DATASET_DIR, cid, "imaging.nii.gz")),
                        "source": "Phase-0 Validation Split"
                    }

    # 3. Add any dataset case with imaging
    if os.path.exists(DATASET_DIR):
        for d in sorted(os.listdir(DATASET_DIR)):
            if d.startswith("case_") and os.path.isdir(os.path.join(DATASET_DIR, d)):
                if d not in cases_dict:
                    has_img = os.path.exists(os.path.join(DATASET_DIR, d, "imaging.nii.gz"))
                    has_gt = os.path.exists(os.path.join(DATASET_DIR, d, "segmentation.nii.gz"))
                    if has_img or has_gt:
                        cases_dict[d] = {
                            "case_id": d,
                            "has_prediction": False,
                            "pred_path": None,
                            "has_gt": has_gt,
                            "has_imaging": has_img,
                            "source": "KiTS23 Dataset"
                        }

    return sorted(list(cases_dict.values()), key=lambda x: (not x["has_prediction"], x["case_id"]))


def load_case_volumes(case_id):
    """Loads and caches 3D volumes (imaging, prediction, ground truth) for a case."""
    if case_id in _CASE_CACHE:
        return _CASE_CACHE[case_id]

    # Find prediction
    cases = {c["case_id"]: c for c in discover_cases()}
    cinfo = cases.get(case_id, {})

    pred_arr = None
    pred_path = cinfo.get("pred_path")
    if pred_path and os.path.exists(pred_path):
        pred_img = nib.load(pred_path)
        pred_arr = np.asanyarray(pred_img.dataobj).astype(np.uint8)

    # Find GT
    gt_arr = None
    gt_path = os.path.join(DATASET_DIR, case_id, "segmentation.nii.gz")
    if os.path.exists(gt_path):
        gt_img = nib.load(gt_path)
        gt_arr = np.asanyarray(gt_img.dataobj).astype(np.uint8)

    # Find Imaging
    img_arr = None
    spacing = (1.0, 1.0, 1.0)
    img_path = os.path.join(DATASET_DIR, case_id, "imaging.nii.gz")
    if os.path.exists(img_path):
        img_nii = nib.load(img_path)
        img_arr = np.asanyarray(img_nii.dataobj).astype(np.float32)
        zooms = img_nii.header.get_zooms()
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    elif gt_arr is not None:
        # Synthetic contrast scan around kidneys if imaging not yet downloaded
        img_arr = np.full_like(gt_arr, fill_value=-1000.0, dtype=np.float32)
        img_arr[gt_arr > 0] = 50.0  # soft tissue HU inside kidney
        img_arr[gt_arr == 2] = 95.0  # contrast enhanced tumor HU
        img_arr[gt_arr == 3] = 15.0  # fluid cyst HU
        # Add smooth abdominal body boundary
        s = generate_binary_structure(3, 1)
        from scipy.ndimage import binary_dilation
        body = binary_dilation(gt_arr > 0, structure=s, iterations=18)
        img_arr[body & (gt_arr == 0)] = 30.0

    num_slices = 0
    if img_arr is not None:
        num_slices = img_arr.shape[0]
    elif pred_arr is not None:
        num_slices = pred_arr.shape[0]
    elif gt_arr is not None:
        num_slices = gt_arr.shape[0]

    cached = {
        "case_id": case_id,
        "num_slices": num_slices,
        "shape": list(img_arr.shape) if img_arr is not None else [],
        "spacing": spacing,
        "imaging": img_arr,
        "prediction": pred_arr,
        "ground_truth": gt_arr,
        "has_real_imaging": os.path.exists(img_path)
    }

    # Limit cache to 3 cases to conserve memory
    if len(_CASE_CACHE) >= 3:
        _CASE_CACHE.pop(next(iter(_CASE_CACHE)))
    _CASE_CACHE[case_id] = cached
    return cached


@app.route("/")
def index():
    models = discover_models()
    cases = discover_cases()
    return render_template("index.html", models=models, cases=cases)


@app.route("/api/models")
def get_models():
    return jsonify(discover_models())


@app.route("/api/cases")
def get_cases():
    return jsonify(discover_cases())


@app.route("/api/case_info/<case_id>")
def get_case_info(case_id):
    cdata = load_case_volumes(case_id)
    return jsonify({
        "case_id": case_id,
        "num_slices": cdata["num_slices"],
        "shape": cdata["shape"],
        "spacing": cdata["spacing"],
        "has_real_imaging": cdata["has_real_imaging"],
        "has_prediction": cdata["prediction"] is not None,
        "has_gt": cdata["ground_truth"] is not None
    })


@app.route("/api/slice/<case_id>/<int:slice_idx>")
def get_slice(case_id, slice_idx):
    """Renders high-speed RGB slice image with windowing and overlays."""
    cdata = load_case_volumes(case_id)
    num_slices = cdata["num_slices"]
    if num_slices == 0 or slice_idx < 0 or slice_idx >= num_slices:
        # Return blank placeholder image
        blank = Image.new("RGB", (512, 512), (15, 23, 42))
        buf = io.BytesIO()
        blank.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    img_vol = cdata["imaging"]
    pred_vol = cdata["prediction"]
    gt_vol = cdata["ground_truth"]

    # Window / Level parameters
    window_type = request.args.get("window", "abdomen")
    custom_wl = request.args.get("level", type=float)
    custom_ww = request.args.get("width", type=float)

    if custom_wl is not None and custom_ww is not None:
        wl, ww = custom_wl, custom_ww
    elif window_type == "kidney":
        wl, ww = 80.0, 300.0
    elif window_type == "bone":
        wl, ww = 400.0, 1500.0
    else:  # abdomen
        wl, ww = 40.0, 400.0

    # 1. Base CT Grayscale Slice
    if img_vol is not None:
        slice_ct = img_vol[slice_idx]
        vmin = wl - (ww / 2.0)
        vmax = wl + (ww / 2.0)
        norm_ct = np.clip((slice_ct - vmin) / (vmax - vmin), 0.0, 1.0)
        gray = (norm_ct * 255.0).astype(np.uint8)
    else:
        gray = np.zeros((512, 512), dtype=np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1)

    # 2. Overlays
    overlay_mode = request.args.get("overlay", "prediction")  # "prediction", "gt", "both", "none"
    opacity = float(request.args.get("opacity", 0.55))
    show_kidney = request.args.get("show_kidney", "true").lower() == "true"
    show_tumor = request.args.get("show_tumor", "true").lower() == "true"
    show_cyst = request.args.get("show_cyst", "true").lower() == "true"

    # Color palette
    COLOR_KIDNEY = np.array([16, 185, 129], dtype=np.float32)  # Emerald #10b981
    COLOR_TUMOR = np.array([239, 68, 68], dtype=np.float32)    # Crimson #ef4444
    COLOR_CYST = np.array([245, 158, 11], dtype=np.float32)    # Amber #f59e0b
    COLOR_GT_CONTOUR = np.array([59, 130, 246], dtype=np.float32)  # Blue #3b82f6

    active_mask = None
    if overlay_mode in ["prediction", "both"] and pred_vol is not None:
        active_mask = pred_vol[slice_idx]

    if active_mask is not None:
        # Prediction segmentation blending
        blend_mask = np.zeros_like(rgb, dtype=np.float32)
        has_color = np.zeros(rgb.shape[:2], dtype=bool)

        if show_kidney:
            m_kid = (active_mask == 1)
            blend_mask[m_kid] = COLOR_KIDNEY
            has_color |= m_kid

        if show_tumor:
            m_tum = (active_mask == 2)
            blend_mask[m_tum] = COLOR_TUMOR
            has_color |= m_tum

        if show_cyst:
            m_cys = (active_mask == 3)
            blend_mask[m_cys] = COLOR_CYST
            has_color |= m_cys

        rgb_float = rgb.astype(np.float32)
        rgb_float[has_color] = (1.0 - opacity) * rgb_float[has_color] + opacity * blend_mask[has_color]
        rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

    # 3. Ground Truth contour overlay
    if overlay_mode in ["gt", "both"] and gt_vol is not None:
        gt_slice = gt_vol[slice_idx]
        if overlay_mode == "gt":
            # Direct color fill for GT
            blend_mask = np.zeros_like(rgb, dtype=np.float32)
            has_color = np.zeros(rgb.shape[:2], dtype=bool)
            if show_kidney and (gt_slice == 1).any():
                blend_mask[gt_slice == 1] = COLOR_KIDNEY
                has_color |= (gt_slice == 1)
            if show_tumor and (gt_slice == 2).any():
                blend_mask[gt_slice == 2] = COLOR_TUMOR
                has_color |= (gt_slice == 2)
            if show_cyst and (gt_slice == 3).any():
                blend_mask[gt_slice == 3] = COLOR_CYST
                has_color |= (gt_slice == 3)
            rgb_float = rgb.astype(np.float32)
            rgb_float[has_color] = (1.0 - opacity) * rgb_float[has_color] + opacity * blend_mask[has_color]
            rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)
        else:
            # Contour outlines for GT when in "both" mode
            import scipy.ndimage as ndi
            for val, col in [(1, COLOR_KIDNEY), (2, COLOR_TUMOR), (3, COLOR_CYST)]:
                if (gt_slice == val).any():
                    dil = ndi.binary_dilation(gt_slice == val, iterations=1)
                    boundary = dil & (~(gt_slice == val))
                    rgb[boundary] = np.clip(col * 1.2, 0, 255).astype(np.uint8)

    # Convert to JPEG and send
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/metrics/<case_id>")
def get_metrics(case_id):
    """Computes clinical volumetric and accuracy metrics for a case."""
    cdata = load_case_volumes(case_id)
    pred_vol = cdata["prediction"]
    gt_vol = cdata["ground_truth"]
    spacing = cdata["spacing"]
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    metrics = {
        "case_id": case_id,
        "has_prediction": pred_vol is not None,
        "has_gt": gt_vol is not None,
        "kidney_vol_ml": 0.0,
        "tumor_vol_ml": 0.0,
        "cyst_vol_ml": 0.0,
        "equiv_diam_cm": 0.0,
        "t_stage": "No Tumor Detected",
        "dice_kidney": None,
        "dice_tumor": None,
        "dice_mass": None
    }

    eval_vol = pred_vol if pred_vol is not None else gt_vol
    if eval_vol is not None:
        kid_voxels = int(np.count_nonzero(eval_vol == 1))
        tum_voxels = int(np.count_nonzero(eval_vol == 2))
        cys_voxels = int(np.count_nonzero(eval_vol == 3))

        metrics["kidney_vol_ml"] = round(kid_voxels * voxel_vol_ml, 1)
        metrics["tumor_vol_ml"] = round(tum_voxels * voxel_vol_ml, 1)
        metrics["cyst_vol_ml"] = round(cys_voxels * voxel_vol_ml, 1)

        # Equivalent sphere diameter: d = 2 * (3 * V / (4 * pi))^(1/3)
        if tum_voxels > 0:
            vol_cm3 = metrics["tumor_vol_ml"]
            diam_cm = 2.0 * ((3.0 * vol_cm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
            metrics["equiv_diam_cm"] = round(diam_cm, 2)
            if diam_cm <= 4.0:
                metrics["t_stage"] = "T1a (<= 4 cm, Partial Nephrectomy)"
            elif diam_cm <= 7.0:
                metrics["t_stage"] = "T1b (4 - 7 cm, Nephron-Sparing)"
            else:
                metrics["t_stage"] = "T2 (> 7 cm, Radical Nephrectomy)"

    # Compute Dice if both exist
    if pred_vol is not None and gt_vol is not None:
        def calc_dice(m1, m2):
            inter = np.logical_and(m1, m2).sum()
            total = m1.sum() + m2.sum()
            return round(float(2.0 * inter / total), 4) if total > 0 else 1.0

        metrics["dice_kidney"] = calc_dice(pred_vol == 1, gt_vol == 1)
        metrics["dice_tumor"] = calc_dice(pred_vol == 2, gt_vol == 2)
        metrics["dice_mass"] = calc_dice((pred_vol == 2) | (pred_vol == 3), (gt_vol == 2) | (gt_vol == 3))

    return jsonify(metrics)


@app.route("/api/model_info/<int:model_id>")
def get_model_info(model_id):
    """Returns deep checkpoint architecture and training metadata for a discovered .pth model."""
    models = discover_models()
    target = next((m for m in models if m["id"] == model_id), None)
    if not target:
        return jsonify({"error": "Model not found"}), 404

    pth_path = target["path"]
    if pth_path in _MODEL_INFO_CACHE:
        return jsonify(_MODEL_INFO_CACHE[pth_path])

    try:
        import torch
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("current_epoch", 100)
        best_ema = round(float(ckpt.get("_best_ema", 0.0)), 4) if ckpt.get("_best_ema") else None
        weights_count = len(ckpt.get("network_weights", {}))
        trainer_name = ckpt.get("trainer_name", "nnUNetTrainer_KiTS23_100ep")
        
        info = {
            "id": target["id"],
            "filename": target["filename"],
            "rel_path": target["rel_path"],
            "size_mb": target["size_mb"],
            "description": target["description"],
            "seed": target["seed"],
            "fold": target["fold"],
            "epoch": epoch,
            "best_ema": best_ema,
            "weights_count": weights_count,
            "trainer_name": trainer_name,
            "architecture": "PlainConvUNet 3D (6 stages, Conv3d, InstanceNorm3d, LeakyReLU)",
            "deep_supervision": True
        }
        _MODEL_INFO_CACHE[pth_path] = info
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "id": target["id"],
            "filename": target["filename"],
            "rel_path": target["rel_path"],
            "size_mb": target["size_mb"],
            "description": target["description"],
            "seed": target["seed"],
            "fold": target["fold"],
            "error": str(e)
        })


@app.route("/api/report_html")
def get_report_html():
    """Serves the generated scientific evaluation report HTML."""
    candidates = [
        os.path.join(BASE_DIR, "outputs", "report.html"),
        os.path.join(r"C:\Users\Sara\Downloads\outputs_TIER1", "report.html"),
        os.path.join(BASE_DIR, "outputs", "SMOKE", "report.html")
    ]
    for p in candidates:
        if os.path.exists(p):
            return send_file(p, mimetype="text/html")
    return "<h3>No HTML Report Found</h3>", 404


@app.route("/api/download_case/<int:case_num>", methods=["POST"])
def fetch_case_imaging(case_num):
    """Downloads imaging volume for a case on-demand from Hugging Face."""
    try:
        import kits23.download as kd
        from tqdm import tqdm
        kd.download_case(case_num, tqdm(total=1, disable=True))
        cid = f"case_{case_num:05d}"
        if cid in _CASE_CACHE:
            del _CASE_CACHE[cid]
        return jsonify({"success": True, "message": f"Successfully downloaded imaging for {cid}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 80)
    print("KiTS23 SEGMENTATION TESTING SUITE & RISK CONTROL INTERFACE")
    print("Discovered Models:")
    for m in discover_models():
        print(f"  [{m['id']}] {m['filename']} ({m['size_mb']} MB) - {m['description']}")
    print("=" * 80)
    print("Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

