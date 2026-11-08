"""Consistent styling, color-blind safe palettes, and index logging for all study visualisations.

Cites:
    PDF §5.3 (Prediction families F0–F5)
    PDF §5.7 (Visualisation and reporting standards)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt

# Fixed color-blind safe palette for prediction families F0–F5
FAMILY_COLORS: Dict[str, str] = {
    "F0": "#d95f02",  # Single model baseline (vermilion)
    "F1": "#1b9e77",  # Ensemble mean (teal / green)
    "F2": "#7570b3",  # Member maximum (purple)
    "F3": "#e7298a",  # Disagreement-aware (magenta)
    "F4": "#e6ab02",  # Dilation (gold)
    "F5": "#66a61e"   # Component selection (leaf green)
}

# Subgroup colors
COHORT_COLORS: Dict[str, str] = {
    "KiTS21": "#2b5c8f",
    "KiTS23": "#d95f02"
}

HISTOLOGY_COLORS: Dict[str, str] = {
    "clear_cell_rcc": "#008080",
    "other": "#b07aa1"
}


class FigureRegistry:
    """Maintains figure_index.csv recording metadata for every generated plot."""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.records: List[Dict[str, Any]] = []

    def register(
        self,
        fig_id: str,
        title: str,
        script_name: str,
        csv_path: Path,
        png_path: Path,
        split_used: str,
        n_samples: int,
        tier: str,
        description: str
    ):
        """Registers a figure and updates figure_index.csv."""
        record = {
            "figure_id": fig_id,
            "title": title,
            "generating_script": script_name,
            "data_csv": str(csv_path),
            "png_path": str(png_path),
            "split_used": split_used,
            "n": n_samples,
            "tier": tier,
            "description": description
        }
        self.records.append(record)
        self.save()

    def save(self):
        """Flushes records to CSV."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_csv(self.registry_path, index=False)
