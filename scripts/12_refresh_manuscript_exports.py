#!/usr/bin/env python3
"""Refresh tables, regional labels, Figures 3/7 and joint QoL sensitivity.

Uses the same writers as the full analysis pipeline. No patient-level models,
correlations, cross-validation folds or permutations are fitted by this command.
Release checksums must be updated separately after inspecting the outputs.
"""
from __future__ import annotations

from importlib import import_module
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.mplconfig"))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from _regional_figure_style import generate_figure_03
from _shared import REGIONAL_OUTCOME_LABELS
from _table_exports import write_table_02


def refresh_display_labels(path: Path, key: str, label: str, mapping: dict[str, str]) -> None:
    """Update text labels while preserving every numeric CSV field verbatim."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or key not in fields or label not in fields:
            return
        rows = list(reader)
    changed = False
    for row in rows:
        replacement = mapping.get(row[key], row[label])
        changed |= row[label] != replacement
        row[label] = replacement
    if changed:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    prediction = import_module("07_compute_prediction_performance")
    qol = import_module("08_compute_qol_associations")
    outputs = import_module("09_generate_manuscript_outputs")
    word = import_module("11_export_tables_to_docx")
    refresh_display_labels(
        ROOT / "results/analysis/cohort/cohort_characteristics.csv",
        "Variable", "Variable", {"Tumor volume": "Tumour volume"},
    )
    regional_dir = ROOT / "results/analysis/regional_chaco"
    for path in sorted(regional_dir.glob("*.csv")):
        refresh_display_labels(path, "outcome", "outcome_label", REGIONAL_OUTCOME_LABELS)
    prediction_dir = ROOT / "results/analysis/prediction"
    tfnbs_dir = ROOT / "results/analysis/tfnbs"
    write_table_02(
        pd.read_csv(tfnbs_dir / "tfnbs_edge_counts_from_mrtrix.csv"),
        pd.read_csv(tfnbs_dir / "tfnbs_anatomical_group_pairs.csv"),
        tfnbs_dir / "table_02_tfnbs_subtest_summary.csv",
    )
    prediction.write_table_03(
        pd.read_csv(prediction_dir / "binary_model_screen_results.csv"),
        pd.read_csv(prediction_dir / "demtect_profile_model_summary.csv"),
        pd.read_csv(prediction_dir / "reported_demtect_subtest_model_details.csv"),
    )
    outputs.copy_tables()
    for spec in word.TABLES:
        table_path = word.export_table(spec, word.load_table_metadata())
        print(f"Wrote {table_path.relative_to(ROOT)}", flush=True)

    qol_dir = ROOT / "results/analysis/qol"
    structural = pd.read_csv(qol_dir / "qol_structural_disconnection_correlations.csv")
    qol.joint_structural_sensitivity(structural).to_csv(
        qol_dir / "qol_structural_joint_51_sensitivity.csv", index=False
    )
    qol.make_figure_07(pd.read_csv(qol_dir / "qol_heatmap_source.csv"))

    partial = pd.read_csv(
        regional_dir / "regional_chaco_partial_spearman.csv", float_precision="round_trip"
    )
    labels = pd.read_csv(regional_dir / "fs191_anatomical_groups.csv")
    figure_path, selection = generate_figure_03(
        partial,
        labels,
        ROOT / "data/resources/nemo_fs191_parcellation.nii.gz",
        ROOT / "outputs/figures/figure_03_global_subtest_disconnection_associations.png",
    )
    selection.to_csv(regional_dir / "figure_03_heatmap_selected_parcels.csv", index=False)
    outputs.write_figure_provenance()
    print(f"Wrote {figure_path.relative_to(ROOT)}", flush=True)
    print("OK: refreshed manuscript exports and joint 51-test QoL sensitivity")


if __name__ == "__main__":
    main()
