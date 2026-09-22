#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from _shared import (
    generated_cohort,
    generated_outcomes,
    generated_regional_chaco,
    load_fs191_labels,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/multivariate"
OUT.mkdir(parents=True, exist_ok=True)

PERMUTATION_SEED = 42
RNG = np.random.default_rng(PERMUTATION_SEED)
N_PERM = 5000
PERMUTATION_METHOD = (
    "Freedman-Lane outcome-residual permutation under reduced covariate model"
)
PCA_MAX_COMPONENTS = 20
PCA_COMPONENTS_USED = 8
PCA_SVD_SOLVER = "full"
PCA_IMPUTATION = "median"
PCA_SCALING = "standard_z"
DEMTECT_ANATOMICAL_REFERENCE_GROUP = "Subcortical/limbic"

AAT_SUBTESTS = {
    "token_test": "Token Test | T Score (p. 133)",
    "repetition": "repetition | average T Score across both tests",
    "naming": "naming | T Score (p. 135)",
    "comprehension": "comprehension | T Score (p. 136)",
}
DEMTECT_SUBTESTS = {
    "word_list": "word list | first pass - out of 20",
    "delayed_recall": "word list | delayed recall - out of 10",
    "number_conversion": "number conversion | out of 4",
    "verbal_fluency": "verbal fluency | out of 20",
    "digit_span_backwards": "digit span backwards | out of 6",
}


def bh_fdr(values: pd.Series) -> pd.Series:
    q = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    if not mask.any():
        return q
    p = values.loc[mask].to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    q.loc[mask] = out
    return q


def rank_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        x = pd.to_numeric(out[col], errors="coerce")
        ok = x.notna()
        y = pd.Series(np.nan, index=x.index, dtype=float)
        if ok.sum() > 1:
            ranks = stats.rankdata(x.loc[ok].to_numpy(dtype=float), method="average")
            probs = (ranks - 0.5) / ok.sum()
            y.loc[ok] = stats.norm.ppf(probs)
        out[col] = y
    return out


def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return out


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    if covariates.size == 0:
        return values - np.nanmean(values, axis=0, keepdims=True)
    c = np.column_stack([np.ones(covariates.shape[0]), covariates])
    beta = np.linalg.lstsq(c, values, rcond=None)[0]
    return values - c @ beta


def pillai_trace(x: np.ndarray, y: np.ndarray) -> float:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    resid = y - fitted
    h = fitted.T @ fitted
    e = resid.T @ resid
    return float(np.trace(h @ np.linalg.pinv(h + e)))


def partial_r2(x: np.ndarray, y: np.ndarray) -> float:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    resid = y - fitted
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_error = float(np.sum(resid**2))
    return 1.0 - ss_error / ss_total if ss_total > 0 else np.nan


def permutation_test(
    y: pd.DataFrame | pd.Series, x: pd.DataFrame, covariates: pd.DataFrame
) -> tuple[float, float, int]:
    merged = pd.concat(
        [
            pd.DataFrame(y),
            x.reset_index(drop=True).add_prefix("x_"),
            covariates.reset_index(drop=True).add_prefix("c_"),
        ],
        axis=1,
    ).dropna()
    n_y_cols = pd.DataFrame(y).shape[1]
    if len(merged) < 20:
        return np.nan, np.nan, len(merged)
    y_frame = rank_normalize(merged.iloc[:, :n_y_cols].reset_index(drop=True))
    y_mat = y_frame.to_numpy(dtype=float)
    x_mat = merged.iloc[:, n_y_cols : n_y_cols + x.shape[1]].to_numpy(dtype=float)
    c_mat = merged.iloc[:, n_y_cols + x.shape[1] :].to_numpy(dtype=float)
    covariate_design = np.column_stack([np.ones(len(c_mat)), c_mat])
    reduced_beta = np.linalg.lstsq(covariate_design, y_mat, rcond=None)[0]
    reduced_fitted = covariate_design @ reduced_beta
    y_res = y_mat - reduced_fitted
    x_res = residualize(x_mat, c_mat)
    x_rank = int(np.linalg.matrix_rank(x_res))
    if x_rank < x_res.shape[1]:
        raise ValueError(
            f"Residualized structural block is rank deficient: "
            f"rank {x_rank} for {x_res.shape[1]} predictors"
        )
    observed = (
        pillai_trace(x_res, y_res)
        if y_res.shape[1] > 1
        else partial_r2(x_res, y_res[:, 0])
    )
    exceed = 0
    for _ in range(N_PERM):
        perm = RNG.permutation(len(x_res))
        y_permuted = reduced_fitted + y_res[perm]
        y_permuted_res = residualize(y_permuted, c_mat)
        stat = (
            pillai_trace(x_res, y_permuted_res)
            if y_permuted_res.shape[1] > 1
            else partial_r2(x_res, y_permuted_res[:, 0])
        )
        if stat >= observed:
            exceed += 1
    return float(observed), float((exceed + 1) / (N_PERM + 1)), len(merged)


cohort = generated_cohort()
outcomes = generated_outcomes()
roi = generated_regional_chaco()
groups = load_fs191_labels()[["roi_index", "roi_name", "anatomical_group", "yeo7_name"]]

cohort = cohort.rename(
    columns={
        "age_at_inclusion": "age",
        "grade": "grade_int",
        "tumor_volume_ml": "volume_ml",
    }
)
roi_cols = [col for col in roi.columns if col.startswith("chaco_roi_")]
data = (
    cohort[["subject_id", "age", "grade_int", "volume_ml", "hemisphere"]]
    .merge(
        outcomes[["subject_id", *AAT_SUBTESTS.values(), *DEMTECT_SUBTESTS.values()]],
        on="subject_id",
    )
    .merge(roi[["subject_id", *roi_cols]], on="subject_id")
)
data["log_volume_ml"] = np.log1p(pd.to_numeric(data["volume_ml"], errors="coerce"))
data["hemisphere_R"] = data["hemisphere"].astype(str).str.upper().eq("R").astype(float)
data = numeric(data, ["age", "grade_int", "log_volume_ml", "hemisphere_R", *roi_cols])

roi_matrix = SimpleImputer(strategy=PCA_IMPUTATION).fit_transform(data[roi_cols])
roi_z = StandardScaler().fit_transform(roi_matrix)
pca = PCA(
    n_components=min(PCA_MAX_COMPONENTS, roi_z.shape[1]), svd_solver=PCA_SVD_SOLVER
).fit(roi_z)
pc_scores = pca.transform(roi_z)[:, :PCA_COMPONENTS_USED]
pc_cols = [f"fs191_pc{i}" for i in range(1, PCA_COMPONENTS_USED + 1)]
pcs = pd.DataFrame(pc_scores, columns=pc_cols, index=data.index)

left_roi_cols = []
for row in groups.itertuples(index=False):
    if "(L)" in str(row.roi_name):
        col = f"chaco_roi_{int(row.roi_index)}"
        if col in data.columns:
            left_roi_cols.append(col)
derived_cols = {
    "left_mean_chaco": data[left_roi_cols].mean(axis=1),
    "global_mean_chaco": data[roi_cols].mean(axis=1),
}
for group_name, sub in groups.groupby("anatomical_group"):
    cols = [
        f"chaco_roi_{int(idx)}"
        for idx in sub["roi_index"]
        if f"chaco_roi_{int(idx)}" in data.columns
    ]
    derived_cols[f"anat_{group_name.lower().replace('/', '_').replace(' ', '_')}"] = (
        data[cols].mean(axis=1)
    )
data = pd.concat([data, pd.DataFrame(derived_cols, index=data.index)], axis=1)
anat_cols = [col for col in data.columns if col.startswith("anat_")]
demtect_anatomical_reference_col = (
    "anat_"
    + DEMTECT_ANATOMICAL_REFERENCE_GROUP.lower()
    .replace("/", "_")
    .replace(" ", "_")
)
if demtect_anatomical_reference_col not in anat_cols:
    raise ValueError(
        "Missing DemTect anatomical reference group: "
        f"{DEMTECT_ANATOMICAL_REFERENCE_GROUP}"
    )
demtect_anat_cols = [
    col for col in anat_cols if col != demtect_anatomical_reference_col
]

pca_summary = pd.DataFrame(
    [
        {
            "n_fs191_parcels": len(roi_cols),
            "n_components_80pct": int(
                np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
            ),
            "variance_explained_pc1_to_pc8": float(
                np.sum(pca.explained_variance_ratio_[:PCA_COMPONENTS_USED])
            ),
            "n_left_labelled_parcels": len(left_roi_cols),
            "pca_max_components_fit": PCA_MAX_COMPONENTS,
            "pca_components_used_in_models": PCA_COMPONENTS_USED,
            "pca_imputation": PCA_IMPUTATION,
            "pca_scaling": PCA_SCALING,
            "pca_svd_solver": PCA_SVD_SOLVER,
        }
    ]
)
pca_summary.to_csv(OUT / "regional_chaco_pca_variance.csv", index=False)

aat_y = data[list(AAT_SUBTESTS.values())]
dem_y = data[list(DEMTECT_SUBTESTS.values())]
base_cov = data[["age", "grade_int", "log_volume_ml", "hemisphere_R"]]
base_covariate_description = "age+grade_int+log_volume_ml+hemisphere_R"

joint_specs = [
    (
        "AAT_subtests",
        "fs191_pc8_given_left_chaco",
        aat_y,
        pcs,
        pd.concat([base_cov, data[["left_mean_chaco"]]], axis=1),
        f"{base_covariate_description}+left_mean_chaco",
    ),
    (
        "AAT_subtests",
        "fs191_anatomical_given_left_chaco",
        aat_y,
        data[anat_cols],
        pd.concat([base_cov, data[["left_mean_chaco"]]], axis=1),
        f"{base_covariate_description}+left_mean_chaco",
    ),
    (
        "DemTect_subtests",
        "fs191_pc8_given_global_chaco",
        dem_y,
        pcs,
        pd.concat([base_cov, data[["global_mean_chaco"]]], axis=1),
        f"{base_covariate_description}+global_mean_chaco",
    ),
    (
        "DemTect_subtests",
        "fs191_anatomical_7df_given_global_chaco",
        dem_y,
        data[demtect_anat_cols],
        pd.concat([base_cov, data[["global_mean_chaco"]]], axis=1),
        f"{base_covariate_description}+global_mean_chaco",
    ),
]
joint_rows = []
for domain, model, y, x, cov, covariate_description in joint_specs:
    stat, p_value, n = permutation_test(y, x, cov)
    joint_rows.append(
        {
            "domain": domain,
            "model": model,
            "n": n,
            "n_outcomes": y.shape[1],
            "n_predictors": x.shape[1],
            "predictors": ";".join(x.columns),
            "pillai_permutation": stat,
            "p_permutation": p_value,
            "covariates": covariate_description,
            "permutation_method": PERMUTATION_METHOD,
            "n_permutations": N_PERM,
            "permutation_seed": PERMUTATION_SEED,
        }
    )
joint = pd.DataFrame(joint_rows)
joint["q_all_structural_conditional_tests"] = bh_fdr(joint["p_permutation"])
joint["significant_fdr"] = joint["q_all_structural_conditional_tests"] < 0.05
joint.to_csv(OUT / "conditional_component_tests.csv", index=False)

subtest_rows = []
for domain, mapping, y_block, mean_chaco_col in [
    ("AAT_subtests", AAT_SUBTESTS, aat_y, "left_mean_chaco"),
    ("DemTect_subtests", DEMTECT_SUBTESTS, dem_y, "global_mean_chaco"),
]:
    for short_name, source_col in mapping.items():
        y = y_block[[source_col]]
        stat, p_value, n = permutation_test(
            y,
            pcs,
            pd.concat([base_cov, data[[mean_chaco_col]]], axis=1),
        )
        subtest_rows.append(
            {
                "domain": domain,
                "subtest": short_name,
                "source_column": source_col,
                "model": "fs191_pc8_given_mean_chaco",
                "n": n,
                "n_predictors": pcs.shape[1],
                "partial_r2": stat,
                "p_permutation": p_value,
                "covariates": f"{base_covariate_description}+{mean_chaco_col}",
                "permutation_method": PERMUTATION_METHOD,
                "n_permutations": N_PERM,
                "permutation_seed": PERMUTATION_SEED,
            }
        )
subtests = pd.DataFrame(subtest_rows)
subtests["q_all_subtest_conditional_tests"] = bh_fdr(subtests["p_permutation"])
subtests["significant_fdr"] = subtests["q_all_subtest_conditional_tests"] < 0.05
subtests.to_csv(OUT / "endpoint_specific_conditional_tests.csv", index=False)

manuscript_rows = []
for row in joint.itertuples(index=False):
    if row.model in {"fs191_pc8_given_left_chaco", "fs191_pc8_given_global_chaco"}:
        manuscript_rows.append(
            {
                "result_level": "joint_profile",
                "domain": row.domain,
                "model": row.model,
                "n": row.n,
                "effect_statistic": row.pillai_permutation,
                "p_permutation": row.p_permutation,
                "q_value": row.q_all_structural_conditional_tests,
                "significant_fdr": row.significant_fdr,
                "subtest": np.nan,
                "permutation_method": row.permutation_method,
                "n_permutations": row.n_permutations,
                "permutation_seed": row.permutation_seed,
            }
        )
for row in subtests.itertuples(index=False):
    if bool(row.significant_fdr):
        manuscript_rows.append(
            {
                "result_level": "individual_subtest",
                "domain": row.domain,
                "model": row.model,
                "n": row.n,
                "effect_statistic": row.partial_r2,
                "p_permutation": row.p_permutation,
                "q_value": row.q_all_subtest_conditional_tests,
                "significant_fdr": row.significant_fdr,
                "subtest": row.subtest,
                "permutation_method": row.permutation_method,
                "n_permutations": row.n_permutations,
                "permutation_seed": row.permutation_seed,
            }
        )
pd.DataFrame(manuscript_rows).to_csv(
    OUT / "joint_subtest_profile_models.csv", index=False
)
print("OK: computed multivariate subtest statistics")
