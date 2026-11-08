"""Patient-level simple random partition and balance audit.

Cites:
    PDF §4.3 (Splits and phases: 60/20/20 simple random assignment, seed logged,
             balance checks across cohort, size, thickness, cysts).
    PDF §6.3 (Integrity controls: pre-register and log partition hash).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class DatasetSplitter:
    """Manages simple random patient-level partitioning and pre-registered balance checks."""

    def __init__(
        self,
        metadata_path: Path,
        seed: int = 20261026,
        train_ratio: float = 0.60,
        cal_ratio: float = 0.20,
        test_ratio: float = 0.20,
        max_cases: Optional[int] = 200,
        dataset_fraction: Optional[float] = None,
        max_imbalance: float = 0.08
    ):
        self.metadata_path = Path(metadata_path)
        self.initial_seed = seed
        self.train_ratio = train_ratio
        self.cal_ratio = cal_ratio
        self.test_ratio = test_ratio
        # If dataset_fraction is specified and max_cases is at default 200, use dataset_fraction
        if dataset_fraction is not None and max_cases == 200:
            self.max_cases = None
        else:
            self.max_cases = max_cases
        self.dataset_fraction = float(dataset_fraction) if dataset_fraction is not None else None
        
        # Adaptive imbalance tolerance: for small cohort sizes, a single discrete patient can represent > 8%
        if max_imbalance == 0.08 and self.max_cases is not None and self.max_cases < 150:
            n_cal = max(1, int(round(self.max_cases * self.cal_ratio)))
            self.max_imbalance = max(0.08, 1.5 / n_cal)
        else:
            self.max_imbalance = max_imbalance

    def load_metadata_df(self) -> pd.DataFrame:
        """Loads and normalizes patient metadata from kits23.json."""
        with open(self.metadata_path, "r") as f:
            meta = json.load(f)
        
        records = []
        for item in meta:
            cid = item["case_id"]
            cnum = int(cid.split("_")[1])
            cohort = "KiTS21" if cnum < 300 else "KiTS23"
            
            # Radiographic size
            r_size = item.get("radiographic_size")
            if r_size is None or np.isnan(r_size):
                size_bin = "unknown"
            elif r_size <= 4.0:
                size_bin = "small_le_4cm"
            elif r_size <= 7.0:
                size_bin = "medium_4_7cm"
            else:
                size_bin = "large_gt_7cm"
                
            # Histology
            subtype = item.get("tumor_histologic_subtype")
            hist_group = "clear_cell_rcc" if subtype == "clear_cell_rcc" else "other_or_benign"
            
            records.append({
                "case_id": cid,
                "case_num": cnum,
                "cohort": cohort,
                "radiographic_size": r_size,
                "size_bin": size_bin,
                "histology": subtype,
                "hist_group": hist_group
            })
            
        df = pd.DataFrame(records).sort_values("case_id").reset_index(drop=True)
        return df

    def _check_balance(self, df: pd.DataFrame, train_ids: List[str], cal_ids: List[str], test_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Verifies balance across splits within tolerance bounds (PDF §4.3)."""
        df_train = df[df["case_id"].isin(train_ids)]
        df_cal = df[df["case_id"].isin(cal_ids)]
        df_test = df[df["case_id"].isin(test_ids)]
        
        checks = {}
        all_passed = True
        
        # 1. Cohort balance (% KiTS21 vs KiTS23)
        train_k21 = (df_train["cohort"] == "KiTS21").mean()
        cal_k21 = (df_cal["cohort"] == "KiTS21").mean()
        test_k21 = (df_test["cohort"] == "KiTS21").mean()
        
        diff_cal = abs(cal_k21 - train_k21)
        diff_test = abs(test_k21 - train_k21)
        passed_cohort = (diff_cal <= self.max_imbalance) and (diff_test <= self.max_imbalance)
        checks["cohort_balance"] = {
            "passed": bool(passed_cohort),
            "train_KiTS21": round(float(train_k21), 4),
            "cal_KiTS21": round(float(cal_k21), 4),
            "test_KiTS21": round(float(test_k21), 4),
            "max_diff": round(float(max(diff_cal, diff_test)), 4)
        }
        if not passed_cohort:
            all_passed = False
            
        # 2. Size stratum balance (% small <= 4cm)
        train_small = (df_train["size_bin"] == "small_le_4cm").mean()
        cal_small = (df_cal["size_bin"] == "small_le_4cm").mean()
        test_small = (df_test["size_bin"] == "small_le_4cm").mean()
        diff_cal_sz = abs(cal_small - train_small)
        diff_test_sz = abs(test_small - train_small)
        passed_sz = (diff_cal_sz <= self.max_imbalance) and (diff_test_sz <= self.max_imbalance)
        checks["size_balance"] = {
            "passed": bool(passed_sz),
            "train_small": round(float(train_small), 4),
            "cal_small": round(float(cal_small), 4),
            "test_small": round(float(test_small), 4),
            "max_diff": round(float(max(diff_cal_sz, diff_test_sz)), 4)
        }
        if not passed_sz:
            all_passed = False
            
        # 3. Histology balance (% clear_cell_rcc)
        train_cc = (df_train["hist_group"] == "clear_cell_rcc").mean()
        cal_cc = (df_cal["hist_group"] == "clear_cell_rcc").mean()
        test_cc = (df_test["hist_group"] == "clear_cell_rcc").mean()
        diff_cal_cc = abs(cal_cc - train_cc)
        diff_test_cc = abs(test_cc - train_cc)
        passed_cc = (diff_cal_cc <= self.max_imbalance) and (diff_test_cc <= self.max_imbalance)
        checks["histology_balance"] = {
            "passed": bool(passed_cc),
            "train_ccRCC": round(float(train_cc), 4),
            "cal_ccRCC": round(float(cal_cc), 4),
            "test_ccRCC": round(float(test_cc), 4),
            "max_diff": round(float(max(diff_cal_cc, diff_test_cc)), 4)
        }
        if not passed_cc:
            all_passed = False

        return all_passed, checks

    def partition(self, output_path: Optional[Path] = None, redraw_log_path: Optional[Path] = None) -> Dict[str, Any]:
        """Performs simple random partition with audited redraws if balance fails (PDF §4.3)."""
        df = self.load_metadata_df()
        case_ids_all = df["case_id"].tolist()
        n_available = len(case_ids_all)
        
        # Dataset adjustment: calculate number of cases to use (PDF §1.5 / user specification)
        if self.max_cases is not None and self.max_cases > 0:
            n_total = min(self.max_cases, n_available)
        elif self.dataset_fraction is not None and self.dataset_fraction < 1.0:
            n_total = int(round(n_available * self.dataset_fraction))
        else:
            n_total = n_available
            
        n_train = int(round(n_total * self.train_ratio))
        n_cal = int(round(n_total * self.cal_ratio))
        n_test = n_total - n_train - n_cal
        
        current_seed = self.initial_seed
        attempts = []
        chosen_split = None
        
        for attempt in range(100):
            rng = np.random.RandomState(current_seed)
            shuffled = list(rng.permutation(case_ids_all))
            
            # Active pool for this draw; unused cases remain held out
            active_pool = shuffled[:n_total]
            unused_pool = sorted(shuffled[n_total:]) if n_total < n_available else []
            
            train = sorted(active_pool[:n_train])
            cal = sorted(active_pool[n_train : n_train + n_cal])
            test = sorted(active_pool[n_train + n_cal :])
            
            passed, balance_info = self._check_balance(df, train, cal, test)
            
            # Compute hash of partition including case counts
            split_str = f"cases:{n_total}|train:{','.join(train)}|cal:{','.join(cal)}|test:{','.join(test)}"
            split_hash = hashlib.sha256(split_str.encode("utf-8")).hexdigest()
            
            attempt_record = {
                "attempt": attempt + 1,
                "seed": current_seed,
                "passed": passed,
                "max_cases": self.max_cases,
                "dataset_fraction": self.dataset_fraction,
                "split_hash": split_hash,
                "balance_info": balance_info
            }
            attempts.append(attempt_record)
            
            if passed:
                chosen_split = {
                    "seed": current_seed,
                    "split_hash": split_hash,
                    "max_cases": self.max_cases,
                    "dataset_fraction": self.dataset_fraction if self.dataset_fraction is not None else round(n_total / n_available, 4),
                    "num_cases": {
                        "total_available": n_available,
                        "total_used": n_total,
                        "train": len(train),
                        "cal": len(cal),
                        "test": len(test),
                        "unused": len(unused_pool)
                    },
                    "train_cases": train,
                    "cal_cases": cal,
                    "test_cases": test,
                    "unused_cases": unused_pool,
                    "balance_audit": balance_info,
                    "redraw_history": attempts
                }
                break
            else:
                current_seed += 1

        if chosen_split is None:
            raise RuntimeError(f"Failed to find balanced simple random partition after 100 attempts starting from seed {self.initial_seed}")

        # Save redraw log if requested
        if redraw_log_path is not None:
            redraw_log_path = Path(redraw_log_path)
            redraw_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(redraw_log_path, "w") as f:
                for att in attempts:
                    f.write(json.dumps(att) + "\n")

        # Save split json if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(chosen_split, f, indent=2)

        return chosen_split

    @staticmethod
    def get_smoke_subsets(split_dict: Dict[str, Any], n_train: int = 12, n_cal: int = 4, n_test: int = 4) -> Dict[str, List[str]]:
        """Extracts deterministic subsets for smoke tier pipeline check (PDF §6.2)."""
        return {
            "train_cases": split_dict["train_cases"][:n_train],
            "cal_cases": split_dict["cal_cases"][:n_cal],
            "test_cases": split_dict["test_cases"][:n_test]
        }
