#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.mplconfig"))

import bct  # type: ignore[import-untyped]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]  # noqa: E402
from scipy import stats  # type: ignore[import-untyped]
from scipy.sparse.csgraph import shortest_path  # type: ignore[import-untyped]

from _shared import (
    generated_cohort,
    generated_outcomes,
    load_fs191_labels,
    load_nemo_edge_matrices,
)

OUT = ROOT / "results/analysis/graph_topology"
OUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUT = ROOT / "outputs/figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
N_ROIS = 191
N_EDGES = N_ROIS * (N_ROIS - 1) // 2
N_WORKERS = max(1, int(os.environ.get("ICONS_GRAPH_NWORKERS", "8")))
N_BINARY_PERMUTATIONS = int(os.environ.get("ICONS_GRAPH_BINARY_PERMUTATIONS", "5000"))

# These are graph descriptors computed from NeMo-predicted fs191 nemoSC matrices.
GRAPH_METRICS = {
    "remaining_total_sc_weight": "Total SC weight",
    "remaining_edge_density": "Edge density",
    "remaining_global_efficiency": "Global efficiency",
    "remaining_characteristic_path_length": "Characteristic path length",
    "remaining_local_efficiency": "Local efficiency",
    "remaining_normalized_strength": "Normalized strength",
    "remaining_clustering": "Clustering",
    "remaining_transitivity": "Transitivity",
    "remaining_assortativity": "Assortativity",
    "remaining_modularity": "Modularity",
    "remaining_betweenness": "Betweenness",
    "remaining_participation": "Participation",
    "chacoconn_mean_all_edges": "Mean ChaCoConn across all edges",
    "chacoconn_affected_edge_fraction": "ChaCoConn-affected edge fraction",
    "chacoconn_mean_affected_edges": "Mean ChaCoConn among affected edges",
}
# Fixed Figure 5 display set used to retain the established 12-row manuscript layout.
# The selection covers continuous disconnection descriptors and the principal
# integration, segregation, mixing, community, and centrality constructs. It
# is fixed independently of the observed p-values. Raw total weight is omitted
# because it is matrix-scale dependent, path length because it is redundant
# with efficiency, and affected-edge fraction because it depends on the
# positive-edge definition and is redundant with the two continuous ChaCoConn
# summaries. Inference and FDR correction still cover all 15 metrics above.
FIGURE_METRICS = [
    "remaining_edge_density",
    "chacoconn_mean_all_edges",
    "chacoconn_mean_affected_edges",
    "remaining_global_efficiency",
    "remaining_local_efficiency",
    "remaining_normalized_strength",
    "remaining_clustering",
    "remaining_transitivity",
    "remaining_assortativity",
    "remaining_modularity",
    "remaining_betweenness",
    "remaining_participation",
]
OUTCOME_LABELS = {
    "AAT_total": "AAT total",
    "DemTect_global": "DemTect global",
    "Token Test | T Score (p. 133)": "Token Test",
    "repetition | average T Score across both tests": "Repetition",
    "naming | T Score (p. 135)": "Naming",
    "comprehension | T Score (p. 136)": "Comprehension",
    "word list | first pass - out of 20": "Word List",
    "word list | delayed recall - out of 10": "Delayed Recall",
    "number conversion | out of 4": "Number Conversion",
    "verbal fluency | out of 20": "Verbal Fluency",
    "digit span backwards | out of 6": "Digit Span Backwards",
}
FIGURE_06_OUTCOMES = [
    "AAT_total",
    "Token Test | T Score (p. 133)",
    "repetition | average T Score across both tests",
    "naming | T Score (p. 135)",
    "comprehension | T Score (p. 136)",
    "DemTect_global",
    "word list | first pass - out of 20",
    "word list | delayed recall - out of 10",
    "number conversion | out of 4",
    "verbal fluency | out of 20",
    "digit span backwards | out of 6",
]
FIGURE_06_OUTCOME_LABELS = {
    "AAT_total": "AAT\ntotal",
    "Token Test | T Score (p. 133)": "Token\nTest",
    "repetition | average T Score across both tests": "Repetition",
    "naming | T Score (p. 135)": "Naming",
    "comprehension | T Score (p. 136)": "Compre-\nhension",
    "DemTect_global": "DemTect\nglobal",
    "word list | first pass - out of 20": "Word\nList",
    "word list | delayed recall - out of 10": "Delayed\nRecall",
    "number conversion | out of 4": "Number\nConversion",
    "verbal fluency | out of 20": "Verbal\nFluency",
    "digit span backwards | out of 6": "Digit Span\nBackwards",
}
FIGURE_06_METRIC_LABELS = {
    "remaining_total_sc_weight": "Total SC\nweight",
    "remaining_edge_density": "Edge\ndensity",
    "remaining_global_efficiency": "Global\nefficiency",
    "remaining_characteristic_path_length": "Characteristic\npath length",
    "remaining_local_efficiency": "Local\nefficiency",
    "remaining_normalized_strength": "Normalized\nstrength",
    "remaining_clustering": "Clustering",
    "remaining_transitivity": "Transitivity",
    "remaining_assortativity": "Assortativity",
    "remaining_modularity": "Modularity",
    "remaining_betweenness": "Betweenness",
    "remaining_participation": "Participation",
    "chacoconn_mean_all_edges": "Mean ChaCoConn\nall edges",
    "chacoconn_affected_edge_fraction": "ChaCoConn-affected\nedge fraction",
    "chacoconn_mean_affected_edges": "Mean ChaCoConn\naffected edges",
}
COVARIATES = ["age_at_inclusion", "grade", "tumor_volume_ml", "hemisphere_R"]
LOG_COVARIATES = ["age_at_inclusion", "grade", "log_tumor_volume_ml", "hemisphere_R"]
NODAL_METRICS = {
    "remaining_normalized_strength": "Normalized strength",
    "remaining_clustering": "Clustering",
    "remaining_betweenness": "Betweenness",
    "remaining_eigenvector_centrality": "Eigenvector centrality",
    "remaining_nodal_efficiency": "Nodal efficiency",
    "remaining_participation_coef": "Participation",
    "remaining_local_efficiency": "Local efficiency",
}
RETIRED_INVALID_OUTPUTS = [
    "graph_nodal_deltas_from_nemo.csv",
    "graph_nodal_post_lesion_from_nemo.csv",
    "graph_nodal_signature_counts_conservative_all_metrics.csv",
    "figure_05_post_lesion_graph_topology_source.csv",
    "figure_06_post_lesion_graph_partial_correlations_source.csv",
    "graph_metric_binary_endpoint_rank_freedman_lane_sensitivity.csv",
    "hvs_partial_correlations.csv",
]


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
    restored = np.empty_like(pv)
    restored[order] = np.clip(q, 0, 1)
    out[valid] = restored
    return pd.Series(out, index=p.index)


def rank_residuals(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    cols = [x_col, y_col, *covariates]
    sub = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(covariates) + 5:
        return np.array([]), np.array([]), int(len(sub))
    ranked = sub.rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked[covariates].to_numpy(dtype=float)])
    x = ranked[x_col].to_numpy(dtype=float)
    y = ranked[y_col].to_numpy(dtype=float)
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return x_resid, y_resid, int(len(sub))


def partial_spearman(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: list[str],
) -> tuple[float, float, int]:
    """Pearson correlation of residualized ranks with covariate-aware df."""
    x_resid, y_resid, n = rank_residuals(data, x_col, y_col, covariates)
    if x_resid.size == 0 or np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, np.nan, n
    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    df = n - len(covariates) - 2
    if df <= 0 or not np.isfinite(rho):
        return rho, np.nan, n
    rho_for_test = np.clip(rho, -1.0 + 1e-15, 1.0 - 1e-15)
    t_value = rho_for_test * np.sqrt(df / (1.0 - rho_for_test**2))
    return rho, float(2.0 * stats.t.sf(abs(t_value), df)), n


def rank_hc3_group_test(
    data: pd.DataFrame,
    metric: str,
    group_col: str,
) -> tuple[float, float, float, float, float, int, int]:
    """Rank-ANCOVA group test with an HC3 sandwich covariance estimate."""
    cols = [metric, group_col, *LOG_COVARIATES]
    sub = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(LOG_COVARIATES) + 8 or sub[group_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(len(sub)), 0
    ranked = sub.rank(method="average")
    y = ranked[metric].to_numpy(dtype=float)
    full = np.column_stack([np.ones(len(sub)), ranked[[group_col, *LOG_COVARIATES]].to_numpy(dtype=float)])
    reduced = np.column_stack([np.ones(len(sub)), ranked[LOG_COVARIATES].to_numpy(dtype=float)])
    rank_full = int(np.linalg.matrix_rank(full))
    df_den = int(len(sub) - rank_full)
    if df_den <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(len(sub)), df_den
    xtx_inverse = np.linalg.pinv(full.T @ full)
    beta_full = xtx_inverse @ full.T @ y
    beta_reduced = np.linalg.lstsq(reduced, y, rcond=None)[0]
    residuals = y - full @ beta_full
    rss_full = float(np.sum(residuals**2))
    rss_reduced = float(np.sum((y - reduced @ beta_reduced) ** 2))
    leverage = np.sum(full * (full @ xtx_inverse), axis=1)
    hc3_scale = residuals / np.clip(1.0 - leverage, 1e-8, None)
    weighted_design = full * hc3_scale[:, None]
    covariance_hc3 = xtx_inverse @ (weighted_design.T @ weighted_design) @ xtx_inverse
    robust_se = float(np.sqrt(max(float(covariance_hc3[1, 1]), 0.0)))
    group_beta = float(beta_full[1])
    if robust_se <= 0 or not np.isfinite(robust_se):
        return np.nan, np.nan, np.nan, group_beta, robust_se, int(len(sub)), df_den
    t_value = group_beta / robust_se
    f_value = t_value**2
    p_value = float(stats.f.sf(f_value, 1, df_den))
    ss_effect = max(rss_reduced - rss_full, 0.0)
    partial_eta2 = ss_effect / (ss_effect + rss_full) if (ss_effect + rss_full) else np.nan
    return float(t_value**2), p_value, float(partial_eta2), group_beta, robust_se, int(len(sub)), df_den


def rank_freedman_lane_group_test(
    data: pd.DataFrame,
    metric: str,
    group_col: str,
) -> tuple[float, float, int, int, int]:
    """Freedman-Lane permutation check for the ranked graph descriptor."""
    cols = [metric, group_col, *LOG_COVARIATES]
    sub = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(sub))
    if n < len(LOG_COVARIATES) + 8 or sub[group_col].nunique() < 2:
        return np.nan, np.nan, n, 0, 0
    ranked = sub.rank(method="average")
    y = ranked[metric].to_numpy(dtype=float)
    nuisance = np.column_stack([np.ones(n), ranked[LOG_COVARIATES].to_numpy(dtype=float)])
    group = ranked[group_col].to_numpy(dtype=float)
    nuisance_inverse = np.linalg.pinv(nuisance.T @ nuisance)
    fitted_reduced = nuisance @ nuisance_inverse @ nuisance.T @ y
    reduced_residuals = y - fitted_reduced
    group_residual = group - nuisance @ nuisance_inverse @ nuisance.T @ group
    group_ss = float(group_residual @ group_residual)
    full_rank = int(np.linalg.matrix_rank(np.column_stack([nuisance, group])))
    df_den = n - full_rank
    if group_ss <= 0 or df_den <= 0:
        return np.nan, np.nan, n, df_den, 0
    effect_ss = float((group_residual @ reduced_residuals) ** 2 / group_ss)
    rss_full = max(float(reduced_residuals @ reduced_residuals) - effect_ss, 0.0)
    observed_f = effect_ss / (rss_full / df_den) if rss_full > 0 else np.inf

    seed_material = f"{SEED}|{group_col}|{metric}|rank_freedman_lane"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    exceedances = 0
    batch_size = 500
    for start in range(0, N_BINARY_PERMUTATIONS, batch_size):
        batch = min(batch_size, N_BINARY_PERMUTATIONS - start)
        permutations = np.vstack([rng.permutation(n) for _ in range(batch)])
        residual_permutations = reduced_residuals[permutations]
        nuisance_scores = residual_permutations @ nuisance
        projected_ss = np.einsum(
            "bi,ij,bj->b", nuisance_scores, nuisance_inverse, nuisance_scores, optimize=True
        )
        reduced_ss = np.einsum(
            "bi,bi->b", residual_permutations, residual_permutations, optimize=True
        ) - projected_ss
        permutation_effect_ss = (residual_permutations @ group_residual) ** 2 / group_ss
        permutation_rss = np.maximum(reduced_ss - permutation_effect_ss, np.finfo(float).eps)
        permutation_f = permutation_effect_ss / (permutation_rss / df_den)
        exceedances += int(np.count_nonzero(permutation_f >= observed_f - 1e-12))
    p_value = (exceedances + 1.0) / (N_BINARY_PERMUTATIONS + 1.0)
    return float(observed_f), float(p_value), n, df_den, seed


def cuberoot(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** (1.0 / 3.0)


def invert_weights(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    mask = matrix != 0
    out[mask] = 1.0 / matrix[mask]
    return out


def local_efficiency_wei_unthresholded(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    length_cuberoot = cuberoot(invert_weights(matrix))
    weight_cuberoot = cuberoot(matrix)
    adjacency = (matrix != 0).astype(int)
    values = np.zeros(n, dtype=float)
    for node in range(n):
        neighbors = np.where(np.logical_or(matrix[node, :], matrix[:, node].T))[0]
        if len(neighbors) < 2:
            continue
        distances = shortest_path(length_cuberoot[np.ix_(neighbors, neighbors)], directed=False, method="D")
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_distances = np.where(np.isfinite(distances) & (distances != 0), 1.0 / distances, 0.0)
        sym_eff = inv_distances + inv_distances.T
        sym_weights = weight_cuberoot[node, neighbors] + weight_cuberoot[neighbors, node].T
        numerator = np.sum(np.outer(sym_weights.T, sym_weights) * sym_eff) / 2.0
        sym_adjacency = adjacency[node, neighbors] + adjacency[neighbors, node].T
        denominator = np.sum(sym_adjacency) ** 2 - np.sum(sym_adjacency * sym_adjacency)
        values[node] = numerator / denominator if denominator > 0 else 0.0
    return values


def compute_graph_metrics(matrix: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    weights = bct.weight_conversion(matrix, "normalize")
    lengths = bct.weight_conversion(weights, "lengths")
    distances = shortest_path(lengths, directed=False, unweighted=False, method="D")
    cpl, ge, _, _, _ = bct.charpath(distances, include_diagonal=False, include_infinite=True)
    clustering = bct.clustering_coef_wu(weights)
    transitivity = bct.transitivity_wu(weights)
    assortativity = bct.assortativity_wei(weights)
    community, modularity = bct.community_louvain(weights, seed=SEED)
    strength = bct.strengths_und(weights)
    betweenness = bct.betweenness_wei(lengths)
    eigenvector = bct.eigenvector_centrality_und(weights)
    participation = bct.participation_coef(weights, community)
    local_efficiency = local_efficiency_wei_unthresholded(weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_distances = np.where(np.isfinite(distances) & (distances != 0), 1.0 / distances, 0.0)
    nodal_efficiency = inv_distances.sum(axis=1) / max(weights.shape[0] - 1, 1)
    return (
        {
            "global_efficiency": float(ge),
            "characteristic_path_length": float(cpl),
            "local_efficiency": float(np.nanmean(local_efficiency)),
            "normalized_strength": float(np.nanmean(strength)),
            "clustering": float(np.nanmean(clustering)),
            "transitivity": float(transitivity),
            "assortativity": float(assortativity),
            "modularity": float(modularity),
            "betweenness": float(np.nanmean(betweenness)),
            "participation": float(np.nanmean(participation)),
        },
        {
            "normalized_strength": np.asarray(strength, dtype=float),
            "clustering": np.asarray(clustering, dtype=float),
            "betweenness": np.asarray(betweenness, dtype=float),
            "eigenvector_centrality": np.asarray(eigenvector, dtype=float),
            "nodal_efficiency": np.asarray(nodal_efficiency, dtype=float),
            "participation_coef": np.asarray(participation, dtype=float),
            "local_efficiency": np.asarray(local_efficiency, dtype=float),
        },
    )


def process_subject_graph(
    args: tuple[str, np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sid, remaining_sc, chaco_edge = args
    global_metrics, nodal_metrics = compute_graph_metrics(remaining_sc)
    upper = np.triu_indices_from(remaining_sc, k=1)
    sc_edges = remaining_sc[upper]
    cc_edges = chaco_edge[upper]
    affected = cc_edges > 0
    row: dict[str, Any] = {
        "subject_id": sid,
        "remaining_total_sc_weight": float(sc_edges.sum()),
        "remaining_edge_density": float(np.count_nonzero(sc_edges) / N_EDGES),
        "chacoconn_mean_all_edges": float(cc_edges.mean()),
        "chacoconn_affected_edge_fraction": float(affected.mean()),
        "chacoconn_mean_affected_edges": float(cc_edges[affected].mean()) if affected.any() else 0.0,
    }
    row.update({f"remaining_{key}": value for key, value in global_metrics.items()})
    nodal_rows = []
    for roi_index in range(1, N_ROIS + 1):
        roi_idx = roi_index - 1
        nodal_row: dict[str, Any] = {"subject_id": sid, "roi_index": roi_index}
        nodal_row.update(
            {f"remaining_{key}": float(values[roi_idx]) for key, values in nodal_metrics.items()}
        )
        nodal_rows.append(nodal_row)
    return row, nodal_rows


def compute_graph_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining_sc, chaco_edge = load_nemo_edge_matrices()
    subjects = sorted(set(remaining_sc).intersection(chaco_edge))
    if len(subjects) != 163:
        raise RuntimeError(f"Expected 163 NeMo matrices, observed {len(subjects)}")
    args = [(sid, remaining_sc[sid], chaco_edge[sid]) for sid in subjects]
    rows: list[dict[str, Any]] = []
    nodal_rows: list[dict[str, Any]] = []
    if N_WORKERS > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=N_WORKERS) as pool:
            iterator = pool.imap_unordered(process_subject_graph, args)
            for idx, (row, subject_nodal_rows) in enumerate(iterator, start=1):
                rows.append(row)
                nodal_rows.extend(subject_nodal_rows)
                if idx % 20 == 0:
                    print(
                        f"  nemoSC graph metrics: {idx}/{len(subjects)} subjects",
                        flush=True,
                    )
    else:
        for idx, item in enumerate(args, start=1):
            row, subject_nodal_rows = process_subject_graph(item)
            rows.append(row)
            nodal_rows.extend(subject_nodal_rows)
            if idx % 20 == 0:
                print(
                    f"  nemoSC graph metrics: {idx}/{len(subjects)} subjects",
                    flush=True,
                )
    graph = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    nodal = pd.DataFrame(nodal_rows).sort_values(["subject_id", "roi_index"]).reset_index(drop=True)
    graph.to_csv(OUT / "graph_global_metrics_from_nemo.csv", index=False)
    nodal.to_csv(OUT / "graph_nodal_remaining_connectivity_from_nemo.csv", index=False)
    return graph, nodal


def build_global_inference(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary_rows = []
    adjusted_rows = []
    permutation_rows = []
    for endpoint, label in [
        ("AAT_mean_T_below_threshold", "Mean AAT T-score group"),
        ("DemTect_impaired_binary", "DemTect global impairment"),
    ]:
        group = pd.to_numeric(data[endpoint], errors="coerce")
        for metric, metric_label in GRAPH_METRICS.items():
            x = pd.to_numeric(data[metric], errors="coerce")
            mask = group.isin([0, 1]) & x.notna()
            impaired = x[mask & group.eq(1)]
            reference = x[mask & group.eq(0)]
            if len(impaired) >= 5 and len(reference) >= 5:
                t_stat, p_value = stats.ttest_ind(impaired, reference, equal_var=False)
            else:
                t_stat = p_value = np.nan
            pooled = np.sqrt((impaired.std(ddof=1) ** 2 + reference.std(ddof=1) ** 2) / 2.0)
            d = (impaired.mean() - reference.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
            binary_rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": int(mask.sum()),
                    "n_reference": int(len(reference)),
                    "n_impaired": int(len(impaired)),
                    "reference_mean": float(reference.mean()) if len(reference) else np.nan,
                    "impaired_mean": float(impaired.mean()) if len(impaired) else np.nan,
                    "cohens_d": d,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )
            f_value, p_adjusted, eta2, group_beta, robust_se, n, df_den = rank_hc3_group_test(
                data, metric, endpoint
            )
            adjusted_rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": n,
                    "f_value": f_value,
                    "p_value": p_adjusted,
                    "partial_eta2": eta2,
                    "rank_group_beta": group_beta,
                    "rank_group_beta_hc3_se": robust_se,
                    "df_residual": df_den,
                    "cohens_d_raw": d,
                    "method": "fully_ranked_ancova_hc3_group_plus_age_grade_log_volume_hemisphere",
                }
            )
            permutation_f, permutation_p, permutation_n, permutation_df, permutation_seed = (
                rank_freedman_lane_group_test(data, metric, endpoint)
            )
            permutation_rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": permutation_n,
                    "f_value": permutation_f,
                    "p_value": permutation_p,
                    "df_residual": permutation_df,
                    "n_permutations": N_BINARY_PERMUTATIONS,
                    "permutation_seed": permutation_seed,
                    "method": "fully_ranked_freedman_lane_group_plus_age_grade_log_volume_hemisphere",
                }
            )
    binary = pd.DataFrame(binary_rows)
    binary["q_fdr"] = binary.groupby("endpoint", group_keys=False)["p_value"].apply(bh_fdr)
    binary["fdr_family"] = f"all_{len(GRAPH_METRICS)}_global_metrics_within_endpoint"
    adjusted = pd.DataFrame(adjusted_rows)
    adjusted["q_fdr"] = adjusted.groupby("endpoint", group_keys=False)["p_value"].apply(bh_fdr)
    adjusted["significant_fdr"] = adjusted["q_fdr"] < 0.05
    adjusted["fdr_family"] = f"all_{len(GRAPH_METRICS)}_global_metrics_within_endpoint"
    permutation = pd.DataFrame(permutation_rows)
    permutation["q_fdr"] = permutation.groupby("endpoint", group_keys=False)["p_value"].apply(bh_fdr)
    permutation["significant_fdr"] = permutation["q_fdr"] < 0.05
    permutation["fdr_family"] = f"all_{len(GRAPH_METRICS)}_global_metrics_within_endpoint"

    corr_rows = []
    partial_rows = []
    for outcome, outcome_label in OUTCOME_LABELS.items():
        y = pd.to_numeric(data[outcome], errors="coerce")
        for metric, metric_label in GRAPH_METRICS.items():
            x = pd.to_numeric(data[metric], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) >= 20 and x[mask].nunique() >= 3 and y[mask].nunique() >= 3:
                rho, p_value = stats.spearmanr(x[mask], y[mask])
            else:
                rho = p_value = np.nan
            corr_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": int(mask.sum()),
                    "rho": rho,
                    "p_value": p_value,
                }
            )
            rho_partial, p_partial, n_partial = partial_spearman(data, metric, outcome, COVARIATES)
            partial_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": n_partial,
                    "rho_partial": rho_partial,
                    "p_partial": p_partial,
                    "method": "pearson_correlation_of_residualized_ranks_df_n_minus_k_minus_2",
                    "covariates": "+".join(COVARIATES),
                }
            )
    corr = pd.DataFrame(corr_rows)
    corr["q_fdr"] = corr.groupby("outcome", group_keys=False)["p_value"].apply(bh_fdr)
    corr["fdr_family"] = f"all_{len(GRAPH_METRICS)}_global_metrics_within_outcome"
    partial = pd.DataFrame(partial_rows)
    partial["q_partial"] = partial.groupby("outcome", group_keys=False)["p_partial"].apply(bh_fdr)
    partial["significant_partial_fdr"] = partial["q_partial"] < 0.05
    partial["fdr_family"] = f"all_{len(GRAPH_METRICS)}_global_metrics_within_outcome"
    partial["q_partial_all_165_tests"] = bh_fdr(partial["p_partial"])
    partial["significant_partial_all_165_tests"] = (
        partial["q_partial_all_165_tests"] < 0.05
    )
    partial["fdr_family_all_outcomes_check"] = (
        f"all_{len(OUTCOME_LABELS)}_outcomes_x_{len(GRAPH_METRICS)}_global_metrics"
    )
    return binary, adjusted, permutation, corr, partial


def build_nodal_inference(
    nodal: pd.DataFrame,
    outcomes: pd.DataFrame,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = load_fs191_labels()[["roi_index", "roi_name", "anatomical_group", "yeo7_name"]]
    nodal_data = nodal.merge(outcomes, on="subject_id", how="left").merge(cohort, on="subject_id", how="left")
    rows = []
    for outcome, outcome_label in OUTCOME_LABELS.items():
        for metric_col, metric_label in NODAL_METRICS.items():
            for roi_index in range(1, N_ROIS + 1):
                roi_subset = nodal_data[nodal_data["roi_index"].eq(roi_index)]
                rho, p_value, n = partial_spearman(roi_subset, metric_col, outcome, COVARIATES)
                rows.append(
                    {
                        "roi_index": roi_index,
                        "metric": metric_col.removeprefix("remaining_"),
                        "metric_label": metric_label,
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "rho_partial": rho,
                        "p_partial": p_value,
                        "n": n,
                        "method": "pearson_correlation_of_residualized_ranks_df_n_minus_k_minus_2",
                        "covariates": "+".join(COVARIATES),
                    }
                )
    correlations = pd.DataFrame(rows)
    correlations["q_partial_all_metrics"] = correlations.groupby("outcome", group_keys=False)["p_partial"].apply(bh_fdr)
    correlations["significant_all_metrics"] = correlations["q_partial_all_metrics"] < 0.05
    correlations["fdr_family"] = f"all_{len(NODAL_METRICS)}_metrics_x_{N_ROIS}_parcels_within_outcome"
    correlations = correlations.merge(labels, on="roi_index", how="left")
    all_outcomes = pd.DataFrame(
        {"outcome": list(OUTCOME_LABELS), "outcome_label": list(OUTCOME_LABELS.values())}
    )
    counts = (
        correlations[correlations["significant_all_metrics"]]
        .groupby(["outcome", "outcome_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_sig_parcel_metric"})
    )
    counts = all_outcomes.merge(counts, on=["outcome", "outcome_label"], how="left")
    counts["n_sig_parcel_metric"] = counts["n_sig_parcel_metric"].fillna(0).astype(int)
    counts["fdr_family"] = f"all_{len(NODAL_METRICS)}_metrics_x_{N_ROIS}_parcels_within_outcome"
    return correlations, counts


def write_inventory_and_provenance() -> None:
    inventory = []
    for metric, label in GRAPH_METRICS.items():
        if metric.startswith("chacoconn_"):
            family = "pairwise_disconnection"
            scaling = "ChaCoConn ratio on correctly mirrored upper triangle"
        elif metric in {"remaining_total_sc_weight", "remaining_edge_density"}:
            family = "predicted_remaining_connectivity"
            scaling = (
                "raw correctly mirrored NeMo-predicted remaining SIFT2 matrix"
            )
        else:
            family = "predicted_remaining_topology"
            scaling = (
                "within-matrix maximum-normalized NeMo-predicted remaining "
                "SIFT2 matrix"
            )
        inventory.append(
            {
                "metric": metric,
                "metric_label": label,
                "family": family,
                "scaling": scaling,
                "analysis_role": (
                    "primary_15_metric_inference_family"
                ),
            }
        )
    pd.DataFrame(inventory).to_csv(OUT / "graph_metric_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "source": "data/nemo/ifod2act_fs191/*_nemoSC_sift2_mean.mat",
                "matrix_interpretation": "NeMo-predicted fs191 nemoSC structural connectivity",
                "symmetrization": "upper triangle mirrored as A + A.T; diagonal set to zero",
                "reference_matrix": "not used by manuscript graph metrics",
                "chacoconn_use": "reported separately as pairwise disconnection; not reapplied to nemoSC",
                "topology_scaling": (
                    "each NeMo-predicted fs191 nemoSC matrix "
                    "maximum-normalized before topology computation"
                ),
                "adjusted_covariates": "age, WHO grade, tumor volume, and lesion hemisphere",
                "binary_primary_inference": (
                    "HC3 rank ANCOVA; descriptor, group, age, grade, log tumor volume, and hemisphere "
                    "rank-transformed before fitting; group tested with finite-sample F(1, residual df)"
                ),
                "binary_permutation_check": (
                    f"Freedman-Lane residual permutation on the same ranked design; {N_BINARY_PERMUTATIONS} permutations"
                ),
                "path_length_note": (
                    "characteristic path length was computed from reciprocal edge lengths after matrix normalization"
                ),
            }
        ]
    ).to_csv(OUT / "graph_metric_computation_mode.csv", index=False)


def plot_binary_endpoints(data: pd.DataFrame, adjusted: pd.DataFrame) -> Path:
    rows = []
    endpoint_specs = [
        (
            "AAT_mean_T_below_threshold",
            "Mean AAT T-score group",
            {0: "At/above 63.5", 1: "Below 63.5"},
        ),
        ("DemTect_impaired_binary", "DemTect global impairment", {0: "Unimpaired", 1: "Impaired"}),
    ]
    for endpoint, endpoint_label, group_labels in endpoint_specs:
        for metric in FIGURE_METRICS:
            values = pd.to_numeric(data[metric], errors="coerce")
            std = values.std(ddof=1)
            z = (values - values.mean()) / std if std and np.isfinite(std) else np.nan
            rows.append(
                pd.DataFrame(
                    {
                        "subject_id": data["subject_id"],
                        "endpoint": endpoint_label,
                        "endpoint_variable": endpoint,
                        "metric": metric,
                        "metric_label": GRAPH_METRICS[metric],
                        "metric_z": z,
                        "group": pd.to_numeric(data[endpoint], errors="coerce").map(group_labels),
                        "display_selection": "fixed_12_descriptor_construct_coverage_not_selected_by_p_value",
                    }
                ).dropna()
            )
    source = pd.concat(rows, ignore_index=True)
    source.to_csv(OUT / "figure_05_remaining_connectivity_graph_topology_source.csv", index=False)
    pd.DataFrame(
        [
            {
                "display_order": index,
                "metric": metric,
                "metric_label": GRAPH_METRICS[metric],
                "selection_rule": "fixed construct-coverage set; selected independently of results",
                "inferential_fdr_family": f"all {len(GRAPH_METRICS)} global metrics within each endpoint",
            }
            for index, metric in enumerate(FIGURE_METRICS, start=1)
        ]
    ).to_csv(OUT / "figure_05_metric_selection.csv", index=False)
    sns.set_theme(style="whitegrid", font="Arial")
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 11.1), sharey=True)
    axes_flat = np.asarray(axes, dtype=object).ravel()
    configs = [
        (
            "Mean AAT T-score group",
            ["At/above 63.5", "Below 63.5"],
            {"At/above 63.5": "#9dbbd1", "Below 63.5": "#2f6f9f"},
        ),
        (
            "DemTect global impairment",
            ["Unimpaired", "Impaired"],
            {"Unimpaired": "#d8a6a6", "Impaired": "#b84a4a"},
        ),
    ]
    metric_order = [GRAPH_METRICS[metric] for metric in FIGURE_METRICS]
    for panel_index, (ax, (endpoint_label, group_order, palette)) in enumerate(
        zip(axes_flat, configs)
    ):
        sub = source[source["endpoint"].eq(endpoint_label)]
        sns.boxplot(
            data=sub,
            x="metric_z",
            y="metric_label",
            hue="group",
            order=metric_order,
            hue_order=group_order,
            palette=palette,
            linewidth=1.0,
            fliersize=0,
            ax=ax,
        )
        # Seaborn's categorical jitter uses NumPy's legacy global RNG. Reset it
        # per panel so the manuscript raster is byte-reproducible across runs.
        np.random.seed(SEED + panel_index)
        sns.stripplot(
            data=sub,
            x="metric_z",
            y="metric_label",
            hue="group",
            order=metric_order,
            hue_order=group_order,
            dodge=True,
            palette={group: "#202020" for group in group_order},
            alpha=0.24,
            size=1.8,
            ax=ax,
            legend=False,
        )
        ax.axvline(0, color="#777777", linewidth=0.9)
        ax.set_xlim(-2.8, 3.25)
        ax.set_title(endpoint_label, fontsize=13.0, fontweight="bold")
        ax.set_xlabel("Z-scored metric value", fontsize=13.0, fontweight="bold")
        ax.set_ylabel("")
        endpoint_variable = (
            "AAT_mean_T_below_threshold"
            if endpoint_label == "Mean AAT T-score group"
            else "DemTect_impaired_binary"
        )
        for row_idx, metric in enumerate(FIGURE_METRICS):
            result = adjusted[
                adjusted["endpoint"].eq(endpoint_variable) & adjusted["metric"].eq(metric)
            ]
            if not result.empty and bool(result.iloc[0]["significant_fdr"]):
                ax.text(
                    3.12,
                    row_idx,
                    f"q={result.iloc[0]['q_fdr']:.2g}*",
                    ha="right",
                    va="center",
                    fontsize=11.0,
                    fontweight="bold",
                )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, loc="upper left", fontsize=11.0)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(11.6)
        sns.despine(ax=ax)
    fig.suptitle("Graph metrics across binary clinical endpoints", fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = FIGURE_OUT / "figure_05_graph_topological_network_impact.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_partial_associations(partial: pd.DataFrame) -> Path:
    """Plot the complete fixed descriptor-by-outcome association matrix."""
    metric_order = list(GRAPH_METRICS)
    expected = pd.MultiIndex.from_product(
        [metric_order, FIGURE_06_OUTCOMES], names=["metric", "outcome"]
    )
    source = partial[
        partial["metric"].isin(metric_order) & partial["outcome"].isin(FIGURE_06_OUTCOMES)
    ].copy()
    if source.duplicated(["metric", "outcome"]).any():
        raise RuntimeError("Figure 6 source contains duplicate descriptor-outcome rows")
    observed = pd.MultiIndex.from_frame(source[["metric", "outcome"]])
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(missing) or len(extra) or len(source) != len(expected):
        raise RuntimeError(
            f"Figure 6 requires the complete 15 x 11 matrix; missing={list(missing)}, extra={list(extra)}"
        )

    source["metric_display_order"] = source["metric"].map(
        {metric: index for index, metric in enumerate(metric_order, start=1)}
    )
    source["outcome_display_order"] = source["outcome"].map(
        {outcome: index for index, outcome in enumerate(FIGURE_06_OUTCOMES, start=1)}
    )
    source["battery"] = np.where(source["outcome_display_order"] <= 5, "AAT", "DemTect")
    source["metric_display_label"] = source["metric"].map(FIGURE_06_METRIC_LABELS).str.replace(
        "\n", " ", regex=False
    )
    source["outcome_display_label"] = source["outcome"].map(OUTCOME_LABELS)
    source["significance_marker"] = np.where(source["q_partial"] < 0.05, "*", "")
    source["display_selection"] = (
        "complete fixed 15-descriptor by 11-outcome matrix; no result-based row or panel selection"
    )
    source = source.sort_values(["metric_display_order", "outcome_display_order"])
    source.to_csv(OUT / "figure_06_remaining_connectivity_graph_partial_correlations_source.csv", index=False)

    inventory = pd.DataFrame(
        [
            {
                "display_order": index,
                "metric": metric,
                "metric_label": GRAPH_METRICS[metric],
                "metric_display_label": FIGURE_06_METRIC_LABELS[metric].replace("\n", " "),
                "selection_rule": "all 15 inferred descriptors displayed; fixed independently of results",
                "inferential_fdr_family": f"all {len(GRAPH_METRICS)} global metrics within each outcome",
                "analysis_role": "primary_15_metric_inference_family",
            }
            for index, metric in enumerate(metric_order, start=1)
        ]
    )
    inventory.to_csv(OUT / "figure_06_panel_selection.csv", index=False)

    rho = source.pivot(index="metric", columns="outcome", values="rho_partial").reindex(
        index=metric_order, columns=FIGURE_06_OUTCOMES
    )
    q_value = source.pivot(index="metric", columns="outcome", values="q_partial").reindex(
        index=metric_order, columns=FIGURE_06_OUTCOMES
    )
    if rho.isna().any().any() or q_value.isna().any().any():
        raise RuntimeError("Figure 6 source contains missing rho or q values")

    max_abs = float(np.nanmax(np.abs(rho.to_numpy(dtype=float))))
    color_limit = max(0.10, np.ceil(max_abs * 20.0) / 20.0)
    sns.set_theme(style="white", font="Arial")
    fig, ax = plt.subplots(figsize=(16.4, 11.8))
    heatmap = sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        center=0,
        linewidths=0.8,
        linecolor="white",
        square=False,
        cbar_kws={"label": "Adjusted partial rho", "shrink": 0.72, "pad": 0.025},
    )
    ax.set_xticklabels(
        [FIGURE_06_OUTCOME_LABELS[outcome] for outcome in FIGURE_06_OUTCOMES],
        rotation=0,
        ha="center",
        fontsize=10.6,
        fontweight="bold",
    )
    ax.set_yticklabels(
        [FIGURE_06_METRIC_LABELS[metric] for metric in metric_order],
        rotation=0,
        fontsize=10.7,
        fontweight="bold",
    )
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axvline(5, color="#343434", linewidth=2.4)
    ax.add_patch(
        Rectangle(
            (0, 0),
            len(FIGURE_06_OUTCOMES),
            len(metric_order),
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            clip_on=False,
            zorder=6,
        )
    )

    for row_index, metric in enumerate(metric_order):
        for column_index, outcome in enumerate(FIGURE_06_OUTCOMES):
            value = float(rho.loc[metric, outcome])
            significant = bool(float(q_value.loc[metric, outcome]) < 0.05)
            text_color = "white" if abs(value) >= color_limit * 0.62 else "#202020"
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:.2f}{'*' if significant else ''}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10.0,
                fontweight="bold" if significant else "normal",
            )

    ax.text(
        2.5,
        1.14,
        "AAT",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=12.3,
        fontweight="bold",
        color="#202020",
        clip_on=False,
    )
    ax.text(
        8.0,
        1.14,
        "DemTect",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=12.3,
        fontweight="bold",
        color="#202020",
        clip_on=False,
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.set_label("Adjusted partial rho", fontsize=11.5, fontweight="bold")
    ax.set_title(
        "Graph descriptors and continuous performance",
        fontsize=14.0,
        fontweight="bold",
        pad=106,
    )
    fig.subplots_adjust(left=0.30, right=0.96, top=0.78, bottom=0.04)
    path = FIGURE_OUT / "figure_06_graph_metric_subtest_associations.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    for filename in RETIRED_INVALID_OUTPUTS:
        (OUT / filename).unlink(missing_ok=True)
    graph, nodal = compute_graph_tables()
    outcomes = generated_outcomes()
    cohort = generated_cohort()[
        ["subject_id", "age_at_inclusion", "grade", "tumor_volume_ml", "hemisphere"]
    ].copy()
    cohort["log_tumor_volume_ml"] = np.log1p(pd.to_numeric(cohort["tumor_volume_ml"], errors="coerce"))
    cohort["hemisphere_R"] = cohort["hemisphere"].astype(str).str.upper().eq("R").astype(float)
    data = outcomes.merge(cohort, on="subject_id", how="left").merge(graph, on="subject_id", how="inner")

    binary, adjusted, permutation, corr, partial = build_global_inference(data)
    binary.to_csv(OUT / "graph_metric_binary_endpoint_tests.csv", index=False)
    adjusted.to_csv(OUT / "graph_metric_binary_endpoint_covariate_adjusted.csv", index=False)
    permutation.to_csv(OUT / "graph_metric_binary_endpoint_rank_freedman_lane_permutation.csv", index=False)
    corr.to_csv(OUT / "graph_metric_subtest_score_correlations.csv", index=False)
    partial.to_csv(OUT / "graph_metric_subtest_score_partial_correlations.csv", index=False)

    nodal_corr, nodal_counts = build_nodal_inference(nodal, outcomes, cohort)
    nodal_corr.to_csv(OUT / "graph_nodal_outcome_correlations.csv", index=False)
    nodal_counts.to_csv(OUT / "graph_nodal_signature_counts.csv", index=False)
    by_metric = (
        nodal_corr[nodal_corr["significant_all_metrics"]]
        .pivot_table(index="outcome_label", columns="metric_label", values="roi_index", aggfunc="count", fill_value=0)
        .reset_index()
    )
    by_metric.to_csv(OUT / "graph_nodal_signature_counts_by_metric.csv", index=False)
    write_inventory_and_provenance()
    figure_5 = plot_binary_endpoints(data, adjusted)
    figure_6 = plot_partial_associations(partial)
    print(
        "OK: computed NeMo-predicted fs191 nemoSC graph statistics "
        f"({len(graph)} subjects)"
    )
    print(f"  figure: {figure_5}")
    print(f"  figure: {figure_6}")


if __name__ == "__main__":
    main()
