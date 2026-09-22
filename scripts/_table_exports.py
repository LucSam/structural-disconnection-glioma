"""Manuscript table writers shared by analysis and display-only refreshes."""
from pathlib import Path

import pandas as pd


def write_table_02(counts: pd.DataFrame, pairs: pd.DataFrame, destination: Path) -> None:
    rows = []
    # The saved count table preserves the analysis-defined battery/subtest order.
    for battery, label in counts[["battery", "subtest_label"]].drop_duplicates().itertuples(index=False, name=None):
        subset = counts[counts["battery"].eq(battery) & counts["subtest_label"].eq(label)]
        clinical = subset[subset["model_family"].eq("clinical")].iloc[0]
        location = subset[subset["model_family"].eq("lobe")].iloc[0]
        selected = pairs[pairs["battery"].eq(battery) & pairs["subtest_label"].eq(label) & pairs["model_family"].eq("clinical")]
        pattern = "; ".join(f"{r.class_pair} ({int(r.n_edges)})" for r in selected.head(2).itertuples()) or "—"
        rows.append({
            "Battery": battery,
            "Subtest": label,
            "Clinical: edge FWE": f"{int(clinical.n_fwe05_edges):,}",
            "Clinical: across subtests": f"{int(clinical.n_battery_bonf05_edges):,}",
            "Location-adjusted: edge FWE": f"{int(location.n_fwe05_edges):,}",
            "Location-adjusted: across subtests": f"{int(location.n_battery_bonf05_edges):,}",
            "Leading anatomical pairs (clinical model)": pattern,
        })
    pd.DataFrame(rows).to_csv(destination, index=False)
