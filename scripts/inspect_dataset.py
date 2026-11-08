"""Comprehensive inspection script for KiTS23 dataset on disk."""
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

def inspect():
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset"
    print(f"Inspecting dataset at: {dataset_dir}", flush=True)
    
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} does not exist!", flush=True)
        return

    # 1. Case counts
    print("Step 1: Listing case directories...", flush=True)
    case_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    print(f"Total case folders found: {len(case_dirs)}", flush=True)
    case_ids = [d.name for d in case_dirs]
    print(f"Case range: {case_ids[0]} to {case_ids[-1]}", flush=True)
    
    # Check gap
    case_nums = [int(cid.split("_")[1]) for cid in case_ids]
    cohort_kits21 = [num for num in case_nums if num < 300]
    cohort_kits23 = [num for num in case_nums if num >= 300]
    print(f"Cases < 300 (KiTS21 cohort): {len(cohort_kits21)} (min={min(cohort_kits21)}, max={max(cohort_kits21)})", flush=True)
    print(f"Cases >= 300 (KiTS23 new cohort): {len(cohort_kits23)} (min={min(cohort_kits23)}, max={max(cohort_kits23)})", flush=True)
    
    # 2. File presence
    print("Step 2: Checking file presence...", flush=True)
    has_imaging = []
    has_seg = []
    has_instances = []
    
    for d in case_dirs:
        if (d / "imaging.nii.gz").exists():
            has_imaging.append(d.name)
        if (d / "segmentation.nii.gz").exists():
            has_seg.append(d.name)
        inst_dir = d / "instances"
        if inst_dir.exists() and any(inst_dir.iterdir()):
            has_instances.append(d.name)
            
    print(f"Cases with imaging.nii.gz: {len(has_imaging)} / {len(case_dirs)}", flush=True)
    print(f"Cases with segmentation.nii.gz: {len(has_seg)} / {len(case_dirs)}", flush=True)
    print(f"Cases with instances/ folder: {len(has_instances)} / {len(case_dirs)}", flush=True)
    
    # 3. Size analysis
    print("Step 3: Calculating file sizes...", flush=True)
    total_seg_size = sum((d / "segmentation.nii.gz").stat().st_size for d in case_dirs if (d / "segmentation.nii.gz").exists())
    total_inst_size = 0
    total_inst_files = 0
    for d in case_dirs:
        inst_dir = d / "instances"
        if inst_dir.exists():
            for f in inst_dir.iterdir():
                if f.is_file():
                    total_inst_files += 1
                    total_inst_size += f.stat().st_size
    print(f"Total segmentation.nii.gz size: {total_seg_size / (1024*1024):.2f} MB", flush=True)
    print(f"Total instances files: {total_inst_files}, total size: {total_inst_size / (1024*1024):.2f} MB", flush=True)

    # 4. Read headers and label values on sample
    print("\nStep 4: Inspecting NIfTI Headers & Labels...", flush=True)
    sample_cases = [case_dirs[0], case_dirs[10], case_dirs[50], case_dirs[100], case_dirs[150],
                    case_dirs[200], case_dirs[250], case_dirs[299], case_dirs[300], case_dirs[350],
                    case_dirs[400], case_dirs[450], case_dirs[-1]]
    
    sample_details = []
    unique_labels_set = set()
    for d in sample_cases:
        seg_path = d / "segmentation.nii.gz"
        img = nib.load(str(seg_path))
        hdr = img.header
        shape = img.shape
        zooms = hdr.get_zooms()
        dtype = img.get_data_dtype()
        # Read data array for labels
        data = np.asanyarray(img.dataobj)
        unq = np.unique(data)
        unique_labels_set.update(unq)
        ornt = nib.aff2axcodes(img.affine)
        sample_details.append({
            "case": d.name,
            "shape": list(shape),
            "spacing": [round(float(z), 3) for z in zooms],
            "dtype": str(dtype),
            "orientation": "".join(ornt),
            "labels": [int(x) for x in unq]
        })
    print(f"Unique label values observed in sample: {sorted(list(unique_labels_set))}", flush=True)
    for s in sample_details[:5]:
        print(f"  {s['case']}: shape={s['shape']}, spacing={s['spacing']}, ornt={s['orientation']}, labels={s['labels']}", flush=True)
        
    # Full slice thickness distribution from headers without loading data
    print("\nStep 5: Extracting slice thickness from all 489 headers...", flush=True)
    slice_thicknesses = []
    for d in case_dirs:
        seg_path = d / "segmentation.nii.gz"
        if seg_path.exists():
            hdr = nib.load(str(seg_path)).header
            zooms = hdr.get_zooms()
            slice_thicknesses.append(round(float(zooms[0]), 2))
    
    st_counter = Counter(slice_thicknesses)
    print("Slice thickness (zooms[0]) counts:", flush=True)
    for st, count in sorted(st_counter.items()):
        print(f"  {st:.2f} mm: {count} cases ({count/len(case_dirs)*100:.1f}%)", flush=True)
    
    # 5. Instances and annotation repeats
    print("\nStep 6: Inspecting instances and annotation repeats...", flush=True)
    cases_with_ge2_repeats = 0
    cases_with_ge3_repeats = 0
    instance_type_counts = Counter()
    
    for d in case_dirs:
        inst_dir = d / "instances"
        if not inst_dir.exists():
            continue
        files = [f.name for f in inst_dir.iterdir() if f.name.endswith(".nii.gz")]
        if not files:
            continue
        
        inst_repeats = defaultdict(set)
        for f in files:
            parts = f.replace(".nii.gz", "").split("_")
            label_name = parts[0]
            inst_id = parts[1] if len(parts) > 1 else "inst0"
            annot_id = parts[2] if len(parts) > 2 else "annot1"
            instance_key = f"{label_name}_{inst_id}"
            inst_repeats[instance_key].add(annot_id)
            instance_type_counts[label_name] += 1
            
        max_repeats = max(len(r) for r in inst_repeats.values()) if inst_repeats else 0
        if max_repeats >= 2:
            cases_with_ge2_repeats += 1
        if max_repeats >= 3:
            cases_with_ge3_repeats += 1
            
    print(f"Cases with >= 2 repeats: {cases_with_ge2_repeats} / {len(case_dirs)}", flush=True)
    print(f"Cases with >= 3 repeats: {cases_with_ge3_repeats} / {len(case_dirs)}", flush=True)
    print(f"Instance types encountered: {dict(instance_type_counts)}", flush=True)

    # 6. Metadata inspection (kits23.json)
    print("\nStep 7: Inspecting clinical metadata (kits23.json)...", flush=True)
    json_path = dataset_dir / "kits23.json"
    with open(json_path, "r") as f:
        meta = json.load(f)
    print(f"Total entries in kits23.json: {len(meta)}", flush=True)
    meta_case_ids = [m["case_id"] for m in meta]
    unq_meta_cases = set(meta_case_ids)
    print(f"Unique case_ids in kits23.json: {len(unq_meta_cases)} (Duplicates: {len(meta_case_ids) - len(unq_meta_cases)})", flush=True)
    
    # Fields
    all_fields = list(meta[0].keys())
    print(f"Number of fields: {len(all_fields)}", flush=True)
    print(f"Fields sample: {all_fields[:15]}", flush=True)
    
    # Histology
    histologies = [m.get("tumor_histologic_subtype") for m in meta]
    hist_counter = Counter(histologies)
    print("Histologic subtypes:", flush=True)
    for h, c in hist_counter.most_common():
        print(f"  {h}: {c} ({c/len(meta)*100:.1f}%)", flush=True)
        
    # Sizes
    sizes = [m.get("radiographic_size") for m in meta if m.get("radiographic_size") is not None]
    print(f"Tumour size available: {len(sizes)} / {len(meta)}", flush=True)
    small_tumors = sum(1 for s in sizes if s <= 4.0)
    med_tumors = sum(1 for s in sizes if 4.0 < s <= 7.0)
    large_tumors = sum(1 for s in sizes if s > 7.0)
    print(f"Size <= 4 cm: {small_tumors} ({small_tumors/len(sizes)*100:.1f}%)", flush=True)
    print(f"Size 4-7 cm: {med_tumors} ({med_tumors/len(sizes)*100:.1f}%)", flush=True)
    print(f"Size > 7 cm: {large_tumors} ({large_tumors/len(sizes)*100:.1f}%)", flush=True)

    # 7. Lesion definition: instances vs 26-connected components of tumour U cyst
    print("\nStep 8: Comparing instances vs 26-connected components on sample...", flush=True)
    s26 = generate_binary_structure(3, 3) # 26-connectivity
    lesion_comparison = []
    
    # Sample 10 cases
    sample_eval_cases = [case_dirs[0], case_dirs[1], case_dirs[2], case_dirs[3], case_dirs[4],
                         case_dirs[300], case_dirs[301], case_dirs[302], case_dirs[303], case_dirs[304]]
    diff_count = 0
    total_compared = 0
    for d in sample_eval_cases:
        seg_path = d / "segmentation.nii.gz"
        inst_dir = d / "instances"
        if not seg_path.exists():
            continue
        seg_data = np.asanyarray(nib.load(str(seg_path)).dataobj)
        mass_mask = (seg_data == 2) | (seg_data == 3)
        labeled_cc, num_cc = label(mass_mask, structure=s26)
        
        inst_files = list(inst_dir.glob("tumor_*.nii.gz")) + list(inst_dir.glob("cyst_*.nii.gz"))
        unique_insts = set()
        for f in inst_files:
            parts = f.stem.split("_")
            unique_insts.add(f"{parts[0]}_{parts[1]}")
        
        num_inst = len(unique_insts)
        total_compared += 1
        if num_cc != num_inst:
            diff_count += 1
            lesion_comparison.append({
                "case": d.name,
                "num_cc": int(num_cc),
                "num_inst": int(num_inst),
                "diff": int(num_inst - num_cc)
            })
    print(f"Compared {total_compared} cases: {diff_count} cases had disagreements between 26-CC and instances.", flush=True)
    for lc in lesion_comparison[:5]:
        print(f"  {lc['case']}: 26-CC count = {lc['num_cc']}, instance count = {lc['num_inst']} (diff={lc['diff']})", flush=True)

    # Save summary dictionary to json
    summary = {
        "total_cases": len(case_dirs),
        "cohort_kits21": len(cohort_kits21),
        "cohort_kits23": len(cohort_kits23),
        "has_imaging": len(has_imaging),
        "has_seg": len(has_seg),
        "has_instances": len(has_instances),
        "ge2_repeats": cases_with_ge2_repeats,
        "ge3_repeats": cases_with_ge3_repeats,
        "sample_headers": sample_details,
        "slice_thickness_counts": dict(st_counter),
        "histology_counts": dict(hist_counter),
        "disagreement_rate": diff_count / max(1, total_compared)
    }
    with open(root / "inspection_facts.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved inspection_facts.json successfully.", flush=True)

    # Save summary dictionary to json for easy reporting
    summary = {
        "total_cases": len(case_dirs),
        "cohort_kits21": len(cohort_kits21),
        "cohort_kits23": len(cohort_kits23),
        "has_imaging": len(has_imaging),
        "has_seg": len(has_seg),
        "has_instances": len(has_instances),
        "ge2_repeats": cases_with_ge2_repeats,
        "ge3_repeats": cases_with_ge3_repeats,
        "sample_headers": sample_details,
        "slice_thickness_counts": dict(st_counter),
        "histology_counts": dict(hist_counter),
        "disagreement_rate": diff_count / max(1, total_compared)
    }
    with open(root / "inspection_facts.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved inspection_facts.json successfully.")

if __name__ == "__main__":
    inspect()
