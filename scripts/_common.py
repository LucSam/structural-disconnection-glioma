"""Shared numerical methods for the focused ICONS-GP analysis.

Linear outcome contrasts and outcome-side Freedman-Lane permutations.
Validated against the previous analysis and statsmodels general linear hypotheses.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

OUTCOMES = {
    "AAT": {
        "Token Test": "Token Test | T Score (p. 133)",
        "Repetition": "repetition | average T Score across both tests",
        "Naming": "naming | T Score (p. 135)",
        "Comprehension": "comprehension | T Score (p. 136)",
    },
    "DemTect": {
        "Word List": "word list | first pass - out of 20",
        "Delayed Recall": "word list | delayed recall - out of 10",
        "Number Conversion": "number conversion | out of 4",
        "Verbal Fluency": "verbal fluency | out of 20",
        "Digit Span Backwards": "digit span backwards | out of 6",
    },
}

SPECS = [
    ("AAT_pc8_rank", "AAT", "pc8", "rank_normal", "primary"),
    ("DemTect_pc8_rank", "DemTect", "pc8", "rank_normal", "primary"),
    ("AAT_anatomical_rank", "AAT", "anatomical", "rank_normal", "sensitivity"),
    ("DemTect_anatomical_rank", "DemTect", "anatomical", "rank_normal", "sensitivity"),
    ("AAT_pc8_T_scores", "AAT", "pc8", "original_T_scores", "scale_sensitivity"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def holm(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm correction requires all planned p-values")
    order = np.argsort(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result

def rank_normalise(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Rank normalisation requires a finite outcome matrix")
    ranks = stats.rankdata(y, method="average", axis=0)
    return stats.norm.ppf((ranks - 0.5) / len(y))

def profile_contrasts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return linear contrasts on the supplied, already chosen outcome scale."""
    h = linalg.helmert(y.shape[1], full=False)
    return y @ h.T, h

def orthogonal_design(x: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.column_stack([np.ones(len(x)), covariates])
    if np.linalg.matrix_rank(c) != c.shape[1]:
        raise ValueError("Covariate design is rank deficient")
    qc = np.linalg.qr(c, mode="reduced")[0]
    xr = x - qc @ (qc.T @ x)
    if np.linalg.matrix_rank(xr) != xr.shape[1]:
        raise ValueError("Residualised structural block is rank deficient")
    qx = np.linalg.qr(xr, mode="reduced")[0]
    return qc, qx

def pillai_from_residuals(y_res: np.ndarray, qx: np.ndarray) -> float:
    fitted_coordinates = qx.T @ y_res
    hypothesis = fitted_coordinates.T @ fitted_coordinates
    total = y_res.T @ y_res
    return float(np.trace(hypothesis @ np.linalg.pinv(total)))

def permutation_test(
    y: np.ndarray,
    x: np.ndarray,
    covariates: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Freedman-Lane with fixed predictors and whole residual-row permutations.

    Projecting permuted reduced-model residuals again is algebraically equal
    to adding reduced fitted values and refitting the covariates. QR avoids
    solving the identical predictor design thousands of times.
    """
    if n_permutations < 1:
        raise ValueError("At least one permutation is required")
    if y.ndim != 2 or not all(np.isfinite(a).all() for a in (y, x, covariates)):
        raise ValueError("All model arrays must be finite")
    qc, qx = orthogonal_design(x, covariates)
    y_res = y - qc @ (qc.T @ y)
    if np.linalg.matrix_rank(y_res) != y.shape[1]:
        raise ValueError("Residual outcome matrix is rank deficient")
    observed = pillai_from_residuals(y_res, qx)
    exceed = 0
    for _ in range(n_permutations):
        permuted = y_res[rng.permutation(len(y_res))]
        permuted -= qc @ (qc.T @ permuted)
        exceed += pillai_from_residuals(permuted, qx) >= observed
    interval = stats.binomtest(exceed, n_permutations).proportion_ci(method="exact")
    explained_ss = float(np.sum((qx.T @ y_res) ** 2))
    total_ss = float(np.sum(y_res ** 2))
    return {
        "pillai": observed,
        "p_permutation": float((exceed + 1) / (n_permutations + 1)),
        "n_exceedances": int(exceed),
        "mc_tail_probability_ci_low": float(interval.low),
        "mc_tail_probability_ci_high": float(interval.high),
        "partial_r2_trace_in_sample": explained_ss / total_ss,
        "n": len(y),
        "n_outcomes": y.shape[1],
        "n_predictors": x.shape[1],
        "residual_df": len(y) - qc.shape[1] - qx.shape[1],
        "n_permutations": n_permutations,
    }

def anatomical_group(value: object) -> str:
    """Same fixed name mapping as the existing multivariate analysis."""
    name = str(value).lower()
    token = name.replace("(", " ").replace(")", " ").split()[0] if name.strip() else ""
    terms = [
        ("Cingulate/medial", ["subcallosal", "pericallosal", "cingul", "precuneus"]),
        ("Subcortical/limbic", ["thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens", "ventraldc"]),
        ("Cerebellum", ["crus", "lobule"]),
        ("Perisylvian/central sulci", ["lat_fis", "s_central", "g_ins_lg", "cent_ins"]),
        ("Occipital/visual", ["occip", "cuneus", "calcarine", "lingual", "lunatus", "s_oc_"]),
        ("Temporal", ["temp", "heschl", "fusiform", "sts", "transverse", "collat"]),
        ("Parietal", ["pariet", "postcentral", "angular", "supramarg", "intrapariet", "jensen"]),
        ("Frontal/insula", ["front", "precentral", "paracentral", "subcentral", "insula", "orbital", "opercular", "rectus"]),
    ]
    for group, matches in terms:
        if group == "Cerebellum" and token in {"i_iv", "v", "vi", "viib", "viiia", "viiib", "ix", "x"}:
            return group
        if any(term in name for term in matches):
            return group
    return "Perisylvian/central sulci"

def load_inputs(root: Path) -> tuple[dict, dict, dict]:
    base = root
    files = {
        "cohort": base / "data/private/cohort.csv",
        "outcomes": base / "data/private/outcomes.csv",
        "chaco": base / "data/private/regional_chaco.csv",
        "labels": base / "data/resources/network_definitions.csv",
        "existing_joint": base / "data/reference/conditional_component_tests.csv",
        "existing_pca": base / "data/reference/regional_chaco_pca_variance.csv",
    }
    hashes = {str(p.relative_to(root)): sha256(p) for p in files.values()}
    cohort = pd.read_csv(files["cohort"], usecols=["subject_id", "age_at_inclusion", "grade", "tumor_volume_ml", "hemisphere"])
    outcome_cols = [col for battery in OUTCOMES.values() for col in battery.values()]
    outcomes = pd.read_csv(files["outcomes"], usecols=["subject_id", *outcome_cols])
    roi = pd.read_csv(files["chaco"])
    labels = pd.read_csv(files["labels"]).sort_values("roi_index").reset_index(drop=True)
    data = cohort.merge(outcomes, on="subject_id", validate="one_to_one", sort=False).merge(roi, on="subject_id", validate="one_to_one", sort=False)
    cols = [c for c in roi if c.startswith("chaco_roi_")]
    if len(cols) != 191 or len(data) != len(cohort):
        raise ValueError("Unexpected parcel count or dropped cohort rows")
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    matrix = data[cols].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Regional ChaCo has missing/nonfinite values: stop rather than change the input pipeline")
    if not data["hemisphere"].astype(str).str.upper().isin(["L", "R"]).all():
        raise ValueError("Unexpected hemisphere coding")
    z = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=20, svd_solver="full").fit(z)
    pcs = pca.transform(z)[:, :8]
    left = [f"chaco_roi_{int(r.roi_index)}" for r in labels.itertuples() if "(L)" in str(r.roi_name)]
    labels["anatomical_group"] = labels["roi_name"].map(anatomical_group)
    anatomy = {}
    for group, frame in labels.groupby("anatomical_group"):
        group_cols = [f"chaco_roi_{int(v)}" for v in frame["roi_index"]]
        anatomy[group] = data[group_cols].mean(axis=1).to_numpy()
    anatomical = pd.DataFrame(anatomy)
    numeric = data[["age_at_inclusion", "grade", "tumor_volume_ml"]].apply(pd.to_numeric, errors="coerce")
    cov = np.column_stack([
        numeric["age_at_inclusion"], numeric["grade"],
        np.log1p(numeric["tumor_volume_ml"]),
        data["hemisphere"].astype(str).str.upper().eq("R").astype(float),
    ])
    blocks = {}
    for battery, columns in OUTCOMES.items():
        y = data[list(columns.values())].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        burden = data[left if battery == "AAT" else cols].mean(axis=1).to_numpy()
        c = np.column_stack([cov, burden])
        keep = np.isfinite(y).all(axis=1) & np.isfinite(c).all(axis=1)
        a = anatomical if battery == "AAT" else anatomical.drop(columns="Subcortical/limbic")
        blocks[battery] = {
            "raw": y[keep], "rank": rank_normalise(y[keep]),
            "pc8": pcs[keep], "anatomical": a.to_numpy()[keep], "cov": c[keep],
            "subject_ids": data.loc[keep, "subject_id"].to_numpy(),
            "labels": list(columns), "anatomical_labels": list(a.columns),
            "burden": "left_mean_regional_chaco" if battery == "AAT" else "global_mean_regional_chaco",
        }
    diagnostics = {
        "pca_fit_n": len(data), "parcels": len(cols), "left_parcels": len(left),
        "pca_first_8_variance": float(pca.explained_variance_ratio_[:8].sum()),
        "missing_regional_values": 0,
        "complete_cases": {b: len(v["raw"]) for b, v in blocks.items()},
        "outcome_unique_values": {b: dict(zip(v["labels"], [len(np.unique(v["raw"][:, j])) for j in range(v["raw"].shape[1])])) for b, v in blocks.items()},
    }
    return blocks, {"files": files, "hashes": hashes}, diagnostics

def reproduce_existing(blocks: dict, existing: pd.DataFrame) -> pd.DataFrame:
    """Use the original seed, permutation count, and four-model RNG sequence."""
    rng = np.random.default_rng(42)
    rows = []
    order = [("AAT", "pc8"), ("AAT", "anatomical"), ("DemTect", "pc8"), ("DemTect", "anatomical")]
    for (battery, model), old in zip(order, existing.itertuples(index=False), strict=True):
        block = blocks[battery]
        r = permutation_test(block["rank"], block[model], block["cov"], 5000, rng)
        passed = r["n"] == old.n and np.isclose(r["pillai"], old.pillai_permutation, atol=1e-10, rtol=1e-10) and np.isclose(r["p_permutation"], old.p_permutation, atol=1e-12, rtol=0)
        rows.append({"battery": battery, "model": model, "n": r["n"], "pillai": r["pillai"], "reference_pillai": old.pillai_permutation, "p_permutation": r["p_permutation"], "reference_p": old.p_permutation, "passed": bool(passed)})
    result = pd.DataFrame(rows)
    if not result["passed"].all():
        raise ValueError("Existing joint-test reproduction failed: " + result.to_json(orient="records"))
    return result

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

def safe_spearman(x: pd.Series, y: pd.Series, *, minimum_n: int = 20) -> tuple[int, float, float]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < minimum_n or x[mask].nunique() < 4 or y[mask].nunique() < 4:
        return n, np.nan, np.nan
    rho, p_value = stats.spearmanr(x[mask], y[mask])
    return n, float(rho), float(p_value)
