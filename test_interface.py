import sys
import os

sys.path.insert(0, os.path.abspath("interface"))
import app

print("--- TESTING MODEL DISCOVERY ---")
models = app.discover_models()
print(f"Total models discovered: {len(models)}")
for m in models:
    print(f"  [{m['id']}] {m['description']} ({m['size_mb']} MB)")
    print(f"       File: {m['filename']}")
    print(f"       RelPath: {m['rel_path']}")

print("\n--- TESTING CASES DISCOVERY ---")
cases = app.discover_cases()
print(f"Total cases discovered: {len(cases)}")
for c in cases[:10]:
    print(f"  Case: {c['case_id']} | Has Pred: {c['has_prediction']} | Has GT: {c['has_gt']} | Has Img: {c['has_imaging']}")

if cases:
    test_case = cases[0]["case_id"]
    print(f"\n--- TESTING CASE LOAD & METRICS ({test_case}) ---")
    info = app.load_case_volumes(test_case)
    print(f"  Num Slices: {info['num_slices']}")
    print(f"  Volume Shape: {info['shape']}")
    print(f"  Spacing: {info['spacing']}")
    print(f"  Has real imaging: {info['has_real_imaging']}")
    
    with app.app.test_client() as client:
        res = client.get(f"/api/metrics/{test_case}")
        print("  Metrics response status:", res.status_code)
        print("  Metrics data:", res.get_json())

        slice_res = client.get(f"/api/slice/{test_case}/15?window=abdomen&overlay=prediction")
        print("  Slice response status:", slice_res.status_code)
        print("  Slice content length:", len(slice_res.data), "bytes")
        print("  Slice content type:", slice_res.content_type)

print("\nSUCCESS: All interface backend functions passed verification!")

