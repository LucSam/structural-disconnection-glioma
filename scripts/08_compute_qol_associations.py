#!/usr/bin/env python3
"""Compute exploratory QoL associations and report their correction families."""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from _shared import generated_outcomes, generated_regional_chaco, load_nemo_edge_matrices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/qol"
FIGURE_OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.mplconfig"))

import matplotlib  # type: ignore[import-untyped]  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: E402


NEURO = {
    "AAT_mean_T_score": "Mean AAT T-score",
    "AAT_total": "AAT total",
    "Token Test | T Score (p. 133)": "Token Test",
    "repetition | average T Score across both tests": "Repetition",
    "naming | T Score (p. 135)": "Naming",
    "comprehension | T Score (p. 136)": "Comprehension",
    "DemTect_global": "DemTect global",
    "word list | first pass - out of 20": "Word List",
    "word list | delayed recall - out of 10": "Delayed Recall",
    "number conversion | out of 4": "Number Conversion",
    "verbal fluency | out of 20": "Verbal Fluency",
    "digit span backwards | out of 6": "Digit Span Backwards",
}
QOL = {
    "EORTC_QLQ_C30_global_health": ("C30 Global Health", True, True),
    "EORTC_QLQ_C30_physical_functioning": ("C30 Physical", True, False),
    "EORTC_QLQ_C30_role_functioning": ("C30 Role", True, False),
    "EORTC_QLQ_C30_emotional_functioning": ("C30 Emotional", True, False),
    "EORTC_QLQ_C30_cognitive_functioning": ("C30 Cognitive", True, True),
    "EORTC_QLQ_C30_social_functioning": ("C30 Social", True, False),
    "EORTC_QLQ_C30_fatigue": ("C30 Fatigue", False, False),
    "EORTC_QLQ_C30_nausea_vomiting": ("C30 Nausea/Vomiting", False, False),
    "EORTC_QLQ_C30_pain": ("C30 Pain", False, False),
    "EORTC_QLQ_BN20_future_uncertainty": ("BN20 Future Uncertainty", True, False),
    "EORTC_QLQ_BN20_visual_disorder": ("BN20 Visual Disorder", True, False),
    "EORTC_QLQ_BN20_motor_dysfunction": ("BN20 Motor Dysfunction", True, False),
    "EORTC_QLQ_BN20_communication_deficit": ("BN20 Communication", True, True),
    "EORTC_QLQ_BN20_headaches": ("BN20 Headaches", False, False),
    "EORTC_QLQ_BN20_seizures": ("BN20 Seizures", False, False),
    "EORTC_QLQ_BN20_drowsiness": ("BN20 Drowsiness", False, False),
    "EORTC_QLQ_BN20_hair_loss": ("BN20 Hair Loss", False, False),
    "EORTC_QLQ_BN20_itchy_skin": ("BN20 Itchy Skin", False, False),
    "EORTC_QLQ_BN20_bladder_control": ("BN20 Bladder Control", False, False),
}
FOCUS_QOL = {column: values for column, values in QOL.items() if values[2]}
IMPAIRMENT_GROUPS = {
    "AAT_mean_T_below_threshold": "Mean AAT T-score below 63.5",
    "DemTect_impaired_binary": "DemTect impaired",
    "Token Test_impaired": "Token Test impaired",
    "Repetition_impaired": "Repetition impaired",
    "Naming_impaired": "Naming impaired",
    "Comprehension_impaired": "Comprehension impaired",
}
# These summaries are calculated without QoL scores. "Fixed" describes their
# formulas; it does not imply that the QoL test family was prespecified.
FIXED_REGIONAL_LABELS = [
    "Mean regional NeMo ChaCo",
    "Top-decile regional NeMo ChaCo",
]
TFNBS_EDGE_LABELS = [
    "Mean ChaCoConn in AAT TFNBS edges",
    "Mean ChaCoConn in DemTect TFNBS edges",
]
GRAPH_SPECS = {
    "remaining_total_sc_weight": {
        "feature_label": "Total SC weight",
        "display_label": "Total SC weight",
        "alignment": 1.0,
        "display": False,
    },
    "remaining_edge_density": {
        "feature_label": "Edge density",
        "display_label": "Edge density",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_global_efficiency": {
        "feature_label": "Global efficiency",
        "display_label": "Global efficiency",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_characteristic_path_length": {
        "feature_label": "Characteristic path length",
        "display_label": "Characteristic path length",
        "alignment": -1.0,
        "display": False,
    },
    "remaining_local_efficiency": {
        "feature_label": "Local efficiency",
        "display_label": "Local efficiency",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_normalized_strength": {
        "feature_label": "Normalized strength",
        "display_label": "Normalized strength",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_clustering": {
        "feature_label": "Clustering",
        "display_label": "Clustering",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_transitivity": {
        "feature_label": "Transitivity",
        "display_label": "Transitivity",
        "alignment": 1.0,
        "display": True,
    },
    "remaining_assortativity": {
        "feature_label": "Assortativity",
        "display_label": "Assortativity",
        "alignment": 1.0,
        "display": False,
    },
    "remaining_modularity": {
        "feature_label": "Modularity",
        "display_label": "Modularity",
        "alignment": 1.0,
        "display": False,
    },
    "remaining_betweenness": {
        "feature_label": "Betweenness",
        "display_label": "Betweenness",
        "alignment": 1.0,
        "display": False,
    },
    "remaining_participation": {
        "feature_label": "Participation",
        "display_label": "Participation",
        "alignment": 1.0,
        "display": True,
    },
    "chacoconn_mean_all_edges": {
        "feature_label": "Mean ChaCoConn across all edges",
        "display_label": "Mean ChaCoConn across all edges",
        "alignment": -1.0,
        "display": True,
    },
    "chacoconn_affected_edge_fraction": {
        "feature_label": "ChaCoConn-affected edge fraction",
        "display_label": "ChaCoConn-affected edge fraction",
        "alignment": -1.0,
        "display": False,
    },
    "chacoconn_mean_affected_edges": {
        "feature_label": "Mean ChaCoConn among affected edges",
        "display_label": "Mean ChaCoConn among affected edges",
        "alignment": -1.0,
        "display": True,
    },
}
GRAPH_DISPLAY_METRICS = [
    "remaining_edge_density",
    "chacoconn_mean_all_edges",
    "chacoconn_mean_affected_edges",
    "remaining_global_efficiency",
    "remaining_local_efficiency",
    "remaining_normalized_strength",
    "remaining_clustering",
    "remaining_transitivity",
    "remaining_participation",
]
GRAPH_DISPLAY_LABELS = [
    str(GRAPH_SPECS[metric]["display_label"])
    for metric in GRAPH_DISPLAY_METRICS
]
STRUCTURAL_LABELS = [
    *FIXED_REGIONAL_LABELS,
    *GRAPH_DISPLAY_LABELS,
    *TFNBS_EDGE_LABELS,
]
GRAPH_GLOBAL_PATH = ROOT / "results/analysis/graph_topology/graph_global_metrics_from_nemo.csv"
TFNBS_QOL_MODEL = "clinical_lobe_hemisphere_deficit_only"
TFNBS_SELECTION_NOTE = (
    "Edge set came from the clinical+lobe+hemisphere TFNBS model and was selected for association "
    "with neuropsychological performance in the same cohort; the QoL correlation is exploratory "
    "and non-independent."
)


def bh_fdr(p: pd.Series) -> pd.Series:
    values = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    pv = values[valid]
    if pv.size == 0:
        return pd.Series(out, index=p.index)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(pv)
    tmp[order] = np.clip(q, 0, 1)
    out[valid] = tmp
    return pd.Series(out, index=p.index)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Return Cliff's delta for x relative to y, with ties contributing zero."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    differences = x[:, None] - y[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def safe_spearman(x: pd.Series, y: pd.Series, *, minimum_n: int = 20) -> tuple[int, float, float]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < minimum_n or x[mask].nunique() < 4 or y[mask].nunique() < 4:
        return n, np.nan, np.nan
    rho, p_value = stats.spearmanr(x[mask], y[mask])
    return n, float(rho), float(p_value)


def neuropsych_correlations(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for neuro_col, neuro_label in NEURO.items():
        x = pd.to_numeric(outcomes[neuro_col], errors="coerce")
        for qol_col, (qol_label, higher_better, focus) in QOL.items():
            y = pd.to_numeric(outcomes[qol_col], errors="coerce")
            n, rho, p_value = safe_spearman(x, y)
            rows.append(
                {
                    "neuropsych_measure": neuro_col,
                    "neuropsych_label": neuro_label,
                    "qol_measure": qol_col,
                    "qol_label": qol_label,
                    "qol_higher_better": higher_better,
                    "qol_focus": focus,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["q_fdr_all_neuropsych_qol_tests"] = bh_fdr(result["p_value"])
    result["q_fdr_36_focus_tests"] = np.nan
    focus = result["qol_focus"].astype(bool)
    result.loc[focus, "q_fdr_36_focus_tests"] = bh_fdr(result.loc[focus, "p_value"])
    result["fdr_all_significant"] = result["q_fdr_all_neuropsych_qol_tests"] < 0.05
    result["fdr_focus_significant"] = result["q_fdr_36_focus_tests"] < 0.05
    return result


def qol_completeness(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qol_col, (qol_label, higher_better, focus) in QOL.items():
        values = pd.to_numeric(outcomes[qol_col], errors="coerce")
        rows.append(
            {
                "qol_measure": qol_col,
                "qol_label": qol_label,
                "qol_higher_better": higher_better,
                "qol_focus": focus,
                "n_nonmissing": int(values.notna().sum()),
                "median": float(values.median(skipna=True)),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                "min": float(values.min(skipna=True)),
                "max": float(values.max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def impairment_group_differences(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col, group_label in IMPAIRMENT_GROUPS.items():
        group = pd.to_numeric(outcomes[group_col], errors="coerce")
        for qol_col, (qol_label, higher_better, focus) in QOL.items():
            qol = pd.to_numeric(outcomes[qol_col], errors="coerce")
            mask = group.isin([0, 1]) & qol.notna()
            impaired = qol[mask & group.eq(1)].to_numpy(dtype=float)
            unimpaired = qol[mask & group.eq(0)].to_numpy(dtype=float)
            n = int(len(impaired) + len(unimpaired))
            if n < 20 or len(impaired) < 5 or len(unimpaired) < 5:
                u_stat = p_value = raw_median_delta = raw_cliff = np.nan
            else:
                u_stat, p_value = stats.mannwhitneyu(impaired, unimpaired, alternative="two-sided")
                raw_median_delta = float(np.nanmedian(impaired) - np.nanmedian(unimpaired))
                raw_cliff = cliffs_delta(impaired, unimpaired)
            rows.append(
                {
                    "impairment_measure": group_col,
                    "impairment_label": group_label,
                    "qol_measure": qol_col,
                    "qol_label": qol_label,
                    "qol_higher_better": higher_better,
                    "qol_focus": focus,
                    "n": n,
                    "n_impaired": int(len(impaired)),
                    "n_unimpaired": int(len(unimpaired)),
                    "median_impaired": float(np.nanmedian(impaired)) if len(impaired) else np.nan,
                    "median_unimpaired": float(np.nanmedian(unimpaired)) if len(unimpaired) else np.nan,
                    "raw_median_delta_impaired_minus_unimpaired": raw_median_delta,
                    "worse_status_aligned_median_delta": (
                        (-raw_median_delta if higher_better else raw_median_delta)
                        if np.isfinite(raw_median_delta)
                        else np.nan
                    ),
                    "cliffs_delta_raw": raw_cliff,
                    "cliffs_delta_worse_status_aligned": (
                        (-raw_cliff if higher_better else raw_cliff) if np.isfinite(raw_cliff) else np.nan
                    ),
                    "mannwhitney_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["q_fdr_all_group_qol_tests"] = bh_fdr(result["p_value"])
    result["q_fdr_18_focus_group_tests"] = np.nan
    focus = result["qol_focus"].astype(bool)
    result.loc[focus, "q_fdr_18_focus_group_tests"] = bh_fdr(result.loc[focus, "p_value"])
    result["fdr_all_significant"] = result["q_fdr_all_group_qol_tests"] < 0.05
    result["fdr_focus_significant"] = result["q_fdr_18_focus_group_tests"] < 0.05
    return result


def structural_features() -> pd.DataFrame:
    roi = generated_regional_chaco()
    roi_cols = [col for col in roi.columns if col.startswith("chaco_roi_")]
    features = roi[["subject_id"]].copy()
    features["Mean regional NeMo ChaCo"] = roi[roi_cols].mean(axis=1)
    features["Top-decile regional NeMo ChaCo"] = roi[roi_cols].apply(
        lambda row: float(
            np.mean(
                np.sort(pd.to_numeric(row, errors="coerce").dropna().to_numpy())[
                    -max(1, int(np.ceil(row.notna().sum() * 0.10))) :
                ]
            )
        ),
        axis=1,
    )

    if not GRAPH_GLOBAL_PATH.exists():
        raise FileNotFoundError(
            f"Corrected nemoSC graph metrics are missing: {GRAPH_GLOBAL_PATH}"
        )
    graph = pd.read_csv(GRAPH_GLOBAL_PATH)[["subject_id", *GRAPH_SPECS]].rename(
        columns={metric: str(spec["feature_label"]) for metric, spec in GRAPH_SPECS.items()}
    )
    features = features.merge(graph, on="subject_id", how="left")

    edge_path = ROOT / "results/analysis/tfnbs/tfnbs_significant_and_top_edges.csv"
    if not edge_path.exists():
        return features
    edges = pd.read_csv(edge_path)
    if TFNBS_QOL_MODEL not in set(edges["model"].astype(str)):
        raise RuntimeError(
            f"Corrected TFNBS model {TFNBS_QOL_MODEL!r} is absent from {edge_path}; "
            "refusing to derive QoL TFNBS edge summaries from a legacy model."
        )
    _remaining_sc, chaco_edge = load_nemo_edge_matrices()
    tfnbs_rows = []
    for sid, matrix in chaco_edge.items():
        row: dict[str, object] = {"subject_id": sid}
        for battery, label in [
            ("AAT", "Mean ChaCoConn in AAT TFNBS edges"),
            ("DemTect", "Mean ChaCoConn in DemTect TFNBS edges"),
        ]:
            selected = edges[
                edges["battery"].eq(battery)
                & edges["model"].eq(TFNBS_QOL_MODEL)
                & edges["significant_fwe_edges"].astype(bool)
            ][["roi_i", "roi_j"]].drop_duplicates()
            if selected.empty:
                row[label] = np.nan
            else:
                idx_i = selected["roi_i"].to_numpy(dtype=int) - 1
                idx_j = selected["roi_j"].to_numpy(dtype=int) - 1
                row[label] = float(np.nanmean(matrix[idx_i, idx_j]))
        tfnbs_rows.append(row)
    return features.merge(pd.DataFrame(tfnbs_rows), on="subject_id", how="left")


def structural_correlations(outcomes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    data = outcomes.merge(features, on="subject_id", how="inner")
    rows = []
    graph_by_label = {str(spec["feature_label"]): spec for spec in GRAPH_SPECS.values()}
    for feature in [col for col in features.columns if col != "subject_id"]:
        graph_spec = graph_by_label.get(feature)
        is_graph = graph_spec is not None
        x = pd.to_numeric(data[feature], errors="coerce")
        for qol_col, (qol_label, _higher_better, _focus) in FOCUS_QOL.items():
            y = pd.to_numeric(data[qol_col], errors="coerce")
            n, rho, p_value = safe_spearman(x, y)
            is_tfnbs = "TFNBS" in feature
            inference_eligible = not is_tfnbs
            alignment = (
                float(cast(float, graph_spec["alignment"]))
                if graph_spec is not None
                else -1.0
            )
            rows.append(
                {
                    "structural_feature": feature,
                    "display_label": (
                        str(graph_spec["display_label"])
                        if graph_spec is not None
                        else feature
                    ),
                    "feature_family": (
                        "nemosc_graph_descriptor"
                        if is_graph
                        else "outcome_selected_tfnbs"
                        if is_tfnbs
                        else "fixed_regional_chaco"
                    ),
                    "display_in_figure": bool(graph_spec["display"]) if graph_spec is not None else True,
                    "qol_link": qol_label,
                    "n": n,
                    "rho_feature_vs_better_qol": rho,
                    "expected_direction_sign": alignment,
                    "inference_eligible": inference_eligible,
                    "p": p_value if inference_eligible else np.nan,
                    "same_sample_outcome_selected_feature": is_tfnbs,
                    "selection_note": (
                        TFNBS_SELECTION_NOTE
                        if is_tfnbs
                        else (
                            "nemoSC-derived graph descriptor; all 15 graph descriptors "
                            "were tested across all three focus QoL outcomes."
                            if is_graph
                            else "Fixed regional NeMo ChaCo summary; not selected from the QoL association results."
                        )
                    ),
                }
            )
    result = pd.DataFrame(rows)
    regional_mask = result["feature_family"].eq("fixed_regional_chaco")
    tfnbs_mask = result["feature_family"].eq("outcome_selected_tfnbs")
    graph_mask = result["feature_family"].eq("nemosc_graph_descriptor")
    result["q_fdr_6_fixed_regional_qol_tests"] = np.nan
    result.loc[regional_mask, "q_fdr_6_fixed_regional_qol_tests"] = bh_fdr(
        result.loc[regional_mask, "p"]
    )
    result["q_fdr_45_graph_qol_tests"] = np.nan
    result.loc[graph_mask, "q_fdr_45_graph_qol_tests"] = bh_fdr(result.loc[graph_mask, "p"])
    result["q_fdr_family"] = result["q_fdr_6_fixed_regional_qol_tests"].fillna(
        result["q_fdr_45_graph_qol_tests"]
    )
    result["fdr_family"] = ""
    result.loc[regional_mask, "fdr_family"] = (
        "6 tests: 2 fixed regional ChaCo summaries x 3 focus QoL outcomes"
    )
    result.loc[graph_mask, "fdr_family"] = (
        "45 tests: 15 corrected graph descriptors x 3 focus QoL outcomes"
    )
    result.loc[tfnbs_mask, "fdr_family"] = (
        "No p/q inference: descriptive same-cohort outcome-selected TFNBS edges"
    )
    result["fdr_significant"] = result["q_fdr_family"] < 0.05
    return result


def joint_structural_sensitivity(structural: pd.DataFrame) -> pd.DataFrame:
    """Joint BH correction across the six regional and 45 graph QoL tests.

    Keep the original correction families intact. Outcome-selected TFNBS edge
    sets are descriptive and do not belong to this sensitivity family.
    """
    selected = structural["feature_family"].isin(
        ["fixed_regional_chaco", "nemosc_graph_descriptor"]
    )
    result = structural.loc[selected].copy()
    if len(result) != 51 or result["p"].isna().any():
        raise ValueError("Joint structural QoL sensitivity requires 51 estimable tests")
    result["q_fdr_joint_51"] = bh_fdr(result["p"])
    result["significant_joint_51"] = result["q_fdr_joint_51"] < 0.05
    return result


def heatmap_source(neuro: pd.DataFrame, structural: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in neuro[neuro["qol_focus"].astype(bool)].itertuples(index=False):
        rows.append(
            {
                "block": "Neuropsychology",
                "row_label": row.neuropsych_label,
                "qol_link": row.qol_label,
                "plot_rho": row.spearman_rho,
                "n": row.n,
                "inference_eligible": True,
                "fdr_significant": bool(row.fdr_all_significant),
                "focus_fdr_significant": bool(row.fdr_focus_significant),
                "fdr_family": "all available neuropsychology-QoL tests",
                "selection_note": "Fixed focus QoL display; not selected from the displayed correlations.",
            }
        )
    for row in structural[structural["display_in_figure"].astype(bool)].itertuples(index=False):
        rows.append(
            {
                "block": "Structural disconnection",
                "row_label": row.display_label,
                "qol_link": row.qol_link,
                "plot_rho": row.rho_feature_vs_better_qol,
                "n": row.n,
                "inference_eligible": bool(row.inference_eligible),
                "fdr_significant": bool(row.fdr_significant),
                "focus_fdr_significant": bool(row.fdr_significant),
                "fdr_family": row.fdr_family,
                "selection_note": row.selection_note,
            }
        )
    return pd.DataFrame(rows)


def make_figure_07(source: pd.DataFrame) -> None:
    """Separate performance and structural panels without changing inference."""
    import textwrap

    col_order = [values[0] for values in FOCUS_QOL.values()]
    matrix = source.pivot(index="row_label", columns="qol_link", values="plot_rho")[col_order]
    significant = source.pivot(index="row_label", columns="qol_link", values="fdr_significant")[col_order].fillna(False).astype(bool)
    plt.rcParams.update({"font.size": 11, "font.weight": "normal", "axes.titleweight": "bold", "axes.labelweight": "normal"})
    fig = plt.figure(figsize=(10.5, 14.2))
    grid = fig.add_gridspec(2, 1, left=0.43, right=0.965, bottom=0.12, top=0.93,
                          height_ratios=[12, 16], hspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    structural_grid = grid[1, 0].subgridspec(3, 1, height_ratios=[2, 9, 2], hspace=0.70)
    structural_axes = [fig.add_subplot(structural_grid[i, 0]) for i in range(3)]

    def draw(axis, rows, title, *, top_labels=False, bottom_labels=False):
        data = matrix.reindex(rows)
        image = axis.imshow(data.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        axis.set_yticks(np.arange(len(rows)))
        axis.set_yticklabels([textwrap.fill(r.replace("Normalized", "Normalised"), width=40, break_long_words=False) for r in rows], fontsize=11.5)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(["C30 global\nhealth", "C30 cognitive\nfunctioning", "BN20\ncommunication"], fontsize=11)
        axis.tick_params(length=0, labeltop=top_labels, top=top_labels, labelbottom=bottom_labels, pad=5)
        axis.set_title(title, loc="left", fontsize=11.5, pad=12)
        for i, row in enumerate(rows):
            for j, qol in enumerate(col_order):
                value = data.loc[row, qol]
                if np.isfinite(value):
                    mark = "*" if significant.loc[row, qol] else ""
                    axis.text(j, i, f"{value:.2f}{mark}", ha="center", va="center", fontsize=11.5,
                              color="white" if abs(value) > 0.43 else "#151515")
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)
        return image

    image = draw(ax_a, list(NEURO.values()), "FDR across 222 score–QoL tests", bottom_labels=True)
    draw(structural_axes[0], FIXED_REGIONAL_LABELS, "Regional ChaCo · FDR across 6 tests")
    draw(structural_axes[1], GRAPH_DISPLAY_LABELS, "Network descriptors · FDR across 45 tests")
    draw(structural_axes[2], TFNBS_EDGE_LABELS, "TFNBS edge summaries · descriptive", bottom_labels=True)
    fig.text(0.03, 0.97, "A  Neuropsychological performance", fontsize=16, weight="bold", ha="left")
    fig.text(0.03, structural_axes[0].get_position().y1 + 0.045, "B  Structural measures", fontsize=16, weight="bold", ha="left")
    color_axis = fig.add_axes([0.50, 0.055, 0.36, 0.012])
    bar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    bar.set_label("Spearman rho", fontsize=11)
    bar.ax.xaxis.set_label_position("top")
    fig.text(0.03, 0.011, "* q < 0.05 within the indicated family.\nNo structural associations survived the joint 51-test sensitivity analysis.", fontsize=11)
    for path in [OUT / "figure_07_qol_associations.png", FIGURE_OUT / "figure_07_qol_associations.png"]:
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    outcomes = generated_outcomes()
    completeness = qol_completeness(outcomes)
    neuro = neuropsych_correlations(outcomes)
    groups = impairment_group_differences(outcomes)
    features = structural_features()
    structural = structural_correlations(outcomes, features)
    source = heatmap_source(neuro, structural)

    completeness.to_csv(OUT / "qol_completeness.csv", index=False)
    neuro.to_csv(OUT / "qol_neuropsych_correlations.csv", index=False)
    groups.to_csv(OUT / "qol_impairment_group_differences.csv", index=False)
    structural.to_csv(OUT / "qol_structural_disconnection_correlations.csv", index=False)
    joint_structural_sensitivity(structural).to_csv(
        OUT / "qol_structural_joint_51_sensitivity.csv", index=False
    )
    source.to_csv(OUT / "qol_heatmap_source.csv", index=False)
    make_figure_07(source)
    print("OK: computed exploratory QoL associations, group contrasts, and Figure 7 from source data")


if __name__ == "__main__":
    main()
