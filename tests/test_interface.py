"""Automated unit and integration tests for the clinical PACS web interface."""

import os
import sys
import pytest

# Ensure interface package is on path
INTERFACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interface"))
if INTERFACE_DIR not in sys.path:
    sys.path.insert(0, INTERFACE_DIR)

import app


def test_model_discovery():
    """Verifies nnU-Net checkpoint discovery in models/ directory."""
    models = app.discover_models()
    assert isinstance(models, list), "Models discovery should return a list"
    assert len(models) >= 1, "Should discover at least 1 model checkpoint"
    for m in models:
        assert "filename" in m
        assert "path" in m
        assert "size_mb" in m
        assert os.path.exists(m["path"]), f"Discovered checkpoint path does not exist: {m['path']}"


def test_cases_discovery():
    """Verifies test cases discovery across dataset and predictions_s101."""
    cases = app.discover_cases()
    assert isinstance(cases, list)
    assert len(cases) > 0, "Should discover available cases"
    # Verify structure
    first = cases[0]
    assert "case_id" in first
    assert "has_prediction" in first
    assert "has_gt" in first
    assert "has_imaging" in first


def test_interface_api_endpoints():
    """Tests Flask REST endpoints for metrics calculation and slice rendering."""
    cases = app.discover_cases()
    assert len(cases) > 0, "No cases available to test endpoints"
    test_case = cases[0]["case_id"]

    # Test loading volume
    vol_info = app.load_case_volumes(test_case)
    assert vol_info["num_slices"] > 0
    assert len(vol_info["shape"]) == 3

    # Test Flask client endpoints
    with app.app.test_client() as client:
        # 1. Metrics API
        res_metrics = client.get(f"/api/metrics/{test_case}")
        assert res_metrics.status_code == 200
        data = res_metrics.get_json()
        assert "case_id" in data
        assert "tumor_vol_ml" in data
        assert "t_stage" in data

        # 2. Slice rendering API
        mid_slice = vol_info["num_slices"] // 2
        res_slice = client.get(f"/api/slice/{test_case}/{mid_slice}?window=abdomen&overlay=prediction")
        assert res_slice.status_code == 200
        assert res_slice.content_type == "image/jpeg"
        assert len(res_slice.data) > 0

