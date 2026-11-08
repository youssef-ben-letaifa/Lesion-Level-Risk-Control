"""Test contrast enhancement patterns on sample cases to evaluate phase proxy (cohort < 300 vs >= 300)."""
import urllib.request
from pathlib import Path
import numpy as np
import nibabel as nib

def test_phase():
    root = Path(__file__).resolve().parent.parent
    scratch_dir = root / "scratch_phase"
    scratch_dir.mkdir(exist_ok=True)
    
    test_cases = [("case_00010", 10), ("case_00410", 410)]
    results = {}
    
    for cid, cnum in test_cases:
        local_img = scratch_dir / f"{cid}_imaging.nii.gz"
        if not local_img.exists():
            url = f"https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/{cid}.nii.gz"
            print(f"Downloading sample {cid} from {url}...", flush=True)
            urllib.request.urlretrieve(url, str(local_img))
            print(f"Downloaded {cid} ({local_img.stat().st_size / (1024*1024):.2f} MB)", flush=True)
            
        seg_path = root / "dataset" / cid / "segmentation.nii.gz"
        img = nib.load(str(local_img))
        seg = nib.load(str(seg_path))
        
        img_data = img.get_fdata()
        seg_data = np.asanyarray(seg.dataobj)
        
        # Kidney parenchyma is label 1
        kidney_voxels = img_data[seg_data == 1]
        tumor_voxels = img_data[seg_data == 2]
        cyst_voxels = img_data[seg_data == 3] if np.any(seg_data == 3) else np.array([])
        
        k_mean = float(np.mean(kidney_voxels))
        k_std = float(np.std(kidney_voxels))
        k_p10 = float(np.percentile(kidney_voxels, 10))
        k_p50 = float(np.percentile(kidney_voxels, 50))
        k_p90 = float(np.percentile(kidney_voxels, 90))
        k_iqr = float(np.percentile(kidney_voxels, 75) - np.percentile(kidney_voxels, 25))
        
        # High intensity region (e.g. aorta/arterial peak inside body)
        body_voxels = img_data[img_data > -100]
        aorta_proxy = float(np.percentile(body_voxels, 99.5))
        
        results[cid] = {
            "cohort": "KiTS21 (<300)" if cnum < 300 else "KiTS23 (>=300)",
            "kidney_mean_HU": round(k_mean, 1),
            "kidney_std_HU": round(k_std, 1),
            "kidney_p10_HU": round(k_p10, 1),
            "kidney_median_HU": round(k_p50, 1),
            "kidney_p90_HU": round(k_p90, 1),
            "kidney_IQR_HU": round(k_iqr, 1),
            "aorta_peak_proxy_HU": round(aorta_proxy, 1),
            "tumor_mean_HU": round(float(np.mean(tumor_voxels)), 1) if len(tumor_voxels) > 0 else None,
            "cyst_mean_HU": round(float(np.mean(cyst_voxels)), 1) if len(cyst_voxels) > 0 else None
        }
        print(f"Results for {cid}: {results[cid]}", flush=True)

    # Clean up downloaded files to avoid consuming local disk space
    for f in scratch_dir.iterdir():
        f.unlink()
    scratch_dir.rmdir()
    print("Cleaned up temporary scratch files.", flush=True)
    return results

if __name__ == "__main__":
    test_phase()
