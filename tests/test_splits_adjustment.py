"""Tests for dataset fraction adjustment and balanced partitioning.

Verifies:
- Configurable dataset fraction (e.g. 50% of the dataset)
- Deterministic random sampling and balanced patient partition
- Unused cases tracking
"""

from pathlib import Path
import pytest
from renal_cert.splits import DatasetSplitter
from renal_cert.config import StudyConfig


def test_dataset_fraction_50_percent():
    meta_path = Path(__file__).resolve().parent.parent / "dataset" / "kits23.json"
    if not meta_path.exists():
        pytest.skip("kits23.json not available locally.")

    splitter = DatasetSplitter(
        metadata_path=meta_path,
        seed=20261026,
        train_ratio=0.60,
        cal_ratio=0.20,
        test_ratio=0.20,
        dataset_fraction=0.50,
        max_imbalance=0.08
    )

    res = splitter.partition()

    assert res["dataset_fraction"] == 0.50
    total_avail = res["num_cases"]["total_available"]
    total_used = res["num_cases"]["total_used"]
    n_train = res["num_cases"]["train"]
    n_cal = res["num_cases"]["cal"]
    n_test = res["num_cases"]["test"]
    n_unused = res["num_cases"]["unused"]

    assert total_avail == 489
    assert total_used in (244, 245)
    assert n_train + n_cal + n_test == total_used
    assert total_used + n_unused == total_avail

    # Check 60/20/20 proportions
    assert abs(n_train / total_used - 0.60) < 0.02
    assert abs(n_cal / total_used - 0.20) < 0.02
    assert abs(n_test / total_used - 0.20) < 0.02

    # Verify no overlap between any sets
    train_set = set(res["train_cases"])
    cal_set = set(res["cal_cases"])
    test_set = set(res["test_cases"])
    unused_set = set(res["unused_cases"])

    assert len(train_set.intersection(cal_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(cal_set.intersection(test_set)) == 0
    assert len(unused_set.intersection(train_set)) == 0
    assert len(unused_set.intersection(cal_set)) == 0
    assert len(unused_set.intersection(test_set)) == 0

    # Verify balance passed
    assert res["balance_audit"]["cohort_balance"]["passed"] is True
    assert res["balance_audit"]["size_balance"]["passed"] is True
    assert res["balance_audit"]["histology_balance"]["passed"] is True


def test_max_cases_200():
    meta_path = Path(__file__).resolve().parent.parent / "dataset" / "kits23.json"
    if not meta_path.exists():
        pytest.skip("kits23.json not available locally.")

    splitter = DatasetSplitter(
        metadata_path=meta_path,
        seed=20261026,
        train_ratio=0.60,
        cal_ratio=0.20,
        test_ratio=0.20,
        max_cases=200,
        max_imbalance=0.08
    )

    res = splitter.partition()

    assert res["max_cases"] == 200
    assert res["num_cases"]["total_available"] == 489
    assert res["num_cases"]["total_used"] == 200
    assert res["num_cases"]["train"] == 120
    assert res["num_cases"]["cal"] == 40
    assert res["num_cases"]["test"] == 40
    assert res["num_cases"]["unused"] == 289

    assert res["balance_audit"]["cohort_balance"]["passed"] is True
    assert res["balance_audit"]["size_balance"]["passed"] is True
    assert res["balance_audit"]["histology_balance"]["passed"] is True


def test_yaml_config_parses_dataset_fraction_or_max_cases():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "tier1.yaml"
    cfg = StudyConfig.from_yaml(cfg_path)
    assert hasattr(cfg.splits, "max_cases")
    assert cfg.splits.max_cases == 200

