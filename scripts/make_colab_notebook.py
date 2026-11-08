"""Converts colab.md into an official Jupyter Notebook colab_notebook.ipynb.

Cites:
    PDF §6.2 (Compute, colab.md specification, and notebook generator)
"""

import json
import re
from pathlib import Path


def md_to_notebook(md_path: Path, ipynb_path: Path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by ## Cell
    cell_blocks = re.split(r'\n(?=## Cell )', content)
    
    nb_cells = []

    # Title / header cell
    header_part = cell_blocks[0].strip()
    if header_part:
        nb_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in header_part.split("\n")]
        })

    for block in cell_blocks[1:]:
        lines = block.strip().split("\n")
        title_line = lines[0]
        purpose_line = lines[1] if len(lines) > 1 and lines[1].startswith("*Purpose:") else ""
        
        # Markdown explanation cell
        md_text = f"{title_line}\n\n{purpose_line}\n"
        nb_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in md_text.strip().split("\n")]
        })

        # Find fenced code block
        code_match = re.search(r'```(?:python|bash)?\n(.*?)```', block, re.DOTALL)
        if code_match:
            code_text = code_match.group(1).rstrip()
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_text.split("\n")]
            })

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "colab": {
                "name": "KiTS23_Lesion_Level_Risk_Control.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Generated {ipynb_path} with {len(nb_cells)} cells from {md_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    md_file = root / "colab.md"
    ipynb_file = root / "colab_notebook.ipynb"
    md_to_notebook(md_file, ipynb_file)
