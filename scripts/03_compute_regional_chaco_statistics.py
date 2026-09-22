#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from _shared import REGIONAL_OUTCOME_LABELS, generated_cohort, generated_outcomes, load_fs191_labels, load_nemo_regional_chaco
from _regional_figure_style import generate_figure_02, generate_figure_03

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/regional_chaco"
FIG_OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = REGIONAL_OUTCOME_LABELS
COVARIATES = ["age_at_inclusion", "grade", "tumor_volume_ml"]
COVARIATES_HEMISPHERE = [*COVARIATES, "hemisphere_R"]
GLOBAL_OUTCOMES = ["AAT_total", "DemTect_global"]


def bh_fdr(p: pd.Series) -> pd.Series:
    values = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    ranked = values[valid]
    if ranked.size == 0:
        return pd.Series(out, index=p.index)
    order = np.argsort(ranked)
    sorted_p = ranked[order]
    q = sorted_p * len(sorted_p) / np.arange(1, len(sorted_p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(ranked)
    tmp[order] = np.clip(q, 0, 1)
    out[valid] = tmp
    return pd.Series(out, index=p.index)


def compute_global_sensitivity(
    data: pd.DataFrame,
    roi_cols: list[str],
    covariates: list[str],
    sensitivity: str,
) -> pd.DataFrame:
    rows = []
    for outcome in GLOBAL_OUTCOMES:
        for col in roi_cols:
            rho, p_value, n, dof = rank_residualized_partial_spearman(data, col, outcome, covariates)
            rows.append(
                {
                    "sensitivity": sensitivity,
                    "outcome": outcome,
                    "outcome_label": OUTCOMES[outcome],
                    "roi_index": int(col.replace("chaco_roi_", "")),
                    "n": n,
                    "dof_partial": dof,
                    "rho_partial": rho,
                    "deficit_rho_partial": -rho if np.isfinite(rho) else np.nan,
                    "p_partial": p_value,
                    "method": "partial_spearman_pearson_on_rank_residuals",
                    "covariates": "+".join(covariates),
                }
            )
    result = pd.DataFrame(rows)
    result["q_joint_382"] = bh_fdr(result["p_partial"])
    result["significant_joint_382"] = result["q_joint_382"] < 0.05
    return result


def rank_residualized_partial_spearman(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: list[str],
) -> tuple[float, float, int, int]:
    cols = [x_col, y_col, *covariates]
    sub = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(covariates) + 5 or sub[x_col].nunique() < 3 or sub[y_col].nunique() < 3:
        return np.nan, np.nan, int(len(sub)), 0
    ranked = sub.rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked[covariates].to_numpy(dtype=float)])
    x = ranked[x_col].to_numpy(dtype=float)
    y = ranked[y_col].to_numpy(dtype=float)
    beta_x = np.linalg.lstsq(design, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, y, rcond=None)[0]
    x_resid = x - design @ beta_x
    y_resid = y - design @ beta_y
    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    design_rank = int(np.linalg.matrix_rank(design))
    dof = int(len(sub) - design_rank - 1)
    if not np.isfinite(rho) or dof <= 0 or abs(rho) >= 1.0:
        p_value = 0.0 if np.isfinite(rho) and abs(rho) >= 1.0 and dof > 0 else np.nan
    else:
        denominator = float(max(1.0 - rho * rho, float(np.finfo(float).eps)))
        t_value = float(rho * np.sqrt(float(dof) / denominator))
        p_value = float(2.0 * stats.t.sf(abs(t_value), dof))
    return rho, p_value, int(len(sub)), dof

roi = load_nemo_regional_chaco()
roi.to_csv(OUT / "fs191_regional_chaco_wide.csv", index=False)
long_rows = []
for row in roi.itertuples(index=False):
    sid = str(row.subject_id)
    for col, value in zip(roi.columns[1:], row[1:], strict=True):
        long_rows.append({"subject_id": sid, "roi_index": int(col.replace("chaco_roi_", "")), "chaco": value})
pd.DataFrame(long_rows).to_csv(OUT / "fs191_regional_chaco_long.csv", index=False)
outcomes = generated_outcomes()
cohort = generated_cohort()[["subject_id", *COVARIATES, "hemisphere"]].copy()
cohort["hemisphere_R"] = cohort["hemisphere"].astype(str).str.upper().eq("R").astype(float)
cohort = cohort.drop(columns=["hemisphere"])
labels = load_fs191_labels()[["roi_index", "roi_name", "anatomical_group", "yeo7_name"]]
labels.to_csv(OUT / "fs191_anatomical_groups.csv", index=False)
data = outcomes.merge(cohort, on="subject_id", how="left").merge(roi, on="subject_id", how="inner")
roi_cols = [col for col in roi.columns if col.startswith("chaco_roi_")]

rows = []
partial_rows = []
for outcome_col, outcome_label in OUTCOMES.items():
    y = pd.to_numeric(data[outcome_col], errors="coerce") if outcome_col in data.columns else pd.Series(dtype=float)
    for col in roi_cols:
        x = pd.to_numeric(data[col], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 20 or x[mask].nunique() < 3 or y[mask].nunique() < 3:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = stats.spearmanr(x[mask], y[mask])
        roi_index = int(col.replace("chaco_roi_", ""))
        rows.append(
            {
                "outcome": outcome_col,
                "outcome_label": outcome_label,
                "roi_index": roi_index,
                "n": int(mask.sum()),
                "rho": rho,
                "deficit_rho": -rho if np.isfinite(rho) else np.nan,
                "p_value": p_value,
            }
        )
        rho_partial, p_partial, n_partial, dof_partial = rank_residualized_partial_spearman(
            data, col, outcome_col, COVARIATES
        )
        partial_rows.append(
            {
                "outcome": outcome_col,
                "outcome_label": outcome_label,
                "roi_index": roi_index,
                "n": n_partial,
                "rho_partial": rho_partial,
                "deficit_rho_partial": -rho_partial if np.isfinite(rho_partial) else np.nan,
                "p_partial": p_partial,
                "dof_partial": dof_partial,
                "method": "partial_spearman_pearson_on_rank_residuals",
                "covariates": "+".join(COVARIATES),
            }
        )

result = pd.DataFrame(rows)
result["q_fdr_within_outcome"] = result.groupby("outcome", group_keys=False)["p_value"].apply(bh_fdr)
result = result.merge(labels, on="roi_index", how="left")
result.to_csv(OUT / "regional_chaco_spearman.csv", index=False)

summary = (
    result.assign(significant=lambda d: d["q_fdr_within_outcome"] < 0.05)
    .groupby(["outcome", "outcome_label"], as_index=False)
    .agg(n_parcels=("roi_index", "count"), n_fdr_significant=("significant", "sum"), max_deficit_rho=("deficit_rho", "max"))
)
summary.to_csv(OUT / "regional_chaco_summary.csv", index=False)
partial = pd.DataFrame(partial_rows)
partial["q_partial_within_outcome"] = partial.groupby("outcome", group_keys=False)["p_partial"].apply(bh_fdr)
partial["significant_partial_fdr"] = partial["q_partial_within_outcome"] < 0.05
partial = partial.merge(labels, on="roi_index", how="left")
partial.to_csv(OUT / "regional_chaco_partial_spearman.csv", index=False)

partial_summary = (
    partial.groupby(["outcome", "outcome_label"], as_index=False)
    .agg(
        n_parcels=("roi_index", "count"),
        n_fdr_significant=("significant_partial_fdr", "sum"),
        max_deficit_rho_partial=("deficit_rho_partial", "max"),
        min_deficit_rho_partial=("deficit_rho_partial", "min"),
    )
)
partial_summary.to_csv(OUT / "regional_chaco_partial_summary.csv", index=False)

global_mask = partial["outcome"].isin(GLOBAL_OUTCOMES)
global_partial = partial.loc[global_mask].copy()
global_partial["q_partial_global_score_family"] = bh_fdr(global_partial["p_partial"])
global_partial["significant_global_score_family"] = global_partial["q_partial_global_score_family"] < 0.05
global_partial.to_csv(OUT / "regional_chaco_global_score_partial_spearman.csv", index=False)
global_summary = (
    global_partial.groupby(["outcome", "outcome_label"], as_index=False)
    .agg(
        n_parcels=("roi_index", "count"),
        n_fdr_significant=("significant_global_score_family", "sum"),
        max_deficit_rho_partial=("deficit_rho_partial", "max"),
        min_deficit_rho_partial=("deficit_rho_partial", "min"),
    )
)
global_summary.to_csv(OUT / "regional_chaco_global_score_partial_summary.csv", index=False)
sensitivity = pd.concat(
    [
        compute_global_sensitivity(data, roi_cols, COVARIATES_HEMISPHERE, "hemisphere_adjusted_all_subjects"),
        compute_global_sensitivity(
            data[data["hemisphere_R"].eq(0)].copy(),
            roi_cols,
            COVARIATES,
            "left_hemisphere_only",
        ),
    ],
    ignore_index=True,
).merge(labels, on="roi_index", how="left")
sensitivity.to_csv(OUT / "regional_chaco_global_score_partial_sensitivity.csv", index=False)
sensitivity_summary = (
    sensitivity.groupby(["sensitivity", "outcome", "outcome_label"], as_index=False)
    .agg(
        n_parcels=("roi_index", "size"),
        n_fdr_significant=("significant_joint_382", "sum"),
        n_min=("n", "min"),
        n_max=("n", "max"),
        max_deficit_rho_partial=("deficit_rho_partial", "max"),
        min_deficit_rho_partial=("deficit_rho_partial", "min"),
    )
)
sensitivity_summary.to_csv(OUT / "regional_chaco_global_score_partial_sensitivity_summary.csv", index=False)
pattern_figure_path, centroids, group_chaco_summary = generate_figure_02(
    data,
    labels,
    partial,
    ROOT / "data/resources/nemo_fs191_parcellation.nii.gz",
    FIG_OUT / "figure_02_regional_disconnection_patterns.png",
)
centroids.merge(labels[["roi_index", "roi_name", "anatomical_group"]], on="roi_index", how="left").to_csv(
    OUT / "fs191_centroids.csv", index=False
)
group_chaco_summary.to_csv(OUT / "figure_02_group_chaco_summary.csv", index=False)
figure_path, heatmap_selection = generate_figure_03(
    partial,
    labels,
    ROOT / "data/resources/nemo_fs191_parcellation.nii.gz",
    FIG_OUT / "figure_03_global_subtest_disconnection_associations.png",
)
heatmap_selection.to_csv(OUT / "figure_03_heatmap_selected_parcels.csv", index=False)
print(
    "OK: computed regional ChaCo statistics and "
    f"{pattern_figure_path.relative_to(ROOT)}, {figure_path.relative_to(ROOT)}"
)
