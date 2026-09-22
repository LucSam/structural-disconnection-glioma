#!/usr/bin/env python3
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegressionCV, RidgeCV  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from _shared import generated_cohort, generated_outcomes, generated_regional_chaco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/prediction"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 3
BOOTSTRAP_RESAMPLES = 2_000
CLINICAL = ("log_volume_ml", "age", "grade_int")
LOBE = ("dominant_lobe_simple", "hemisphere")
BASELINE_FEATURE_SET = "clinical_lobe"
REGIONAL_FEATURE_SET = "clinical_lobe_roi_pca"
MEAN_PROFILE_BENCHMARK = "fold_mean_profile_benchmark"
OBSOLETE_OUTPUTS = (
    "binary_prediction_performance_from_cv_predictions.csv",
    "demtect_profile_prediction_performance.csv",
    "demtect_subtest_prediction_performance.csv",
)

GLOBAL_TARGETS = {
    "Mean AAT T-score": "AAT_mean_T_below_threshold",
    "DemTect global": "DemTect_impaired_binary",
}
AAT_TARGETS = {
    "AAT Token Test": "Token Test_impaired",
    "AAT Repetition": "Repetition_impaired",
    "AAT Naming": "Naming_impaired",
    "AAT Comprehension": "Comprehension_impaired",
}
DEMTECT_SUBTESTS = {
    "Word List 1st pass": "word list | first pass - out of 20",
    "Delayed Recall": "word list | delayed recall - out of 10",
    "Number Conversion": "number conversion | out of 4",
    "Verbal Fluency": "verbal fluency | out of 20",
    "Digit Span Backwards": "digit span backwards | out of 6",
}
DEMTECT_MAX = {
    "Word List 1st pass": 20.0,
    "Delayed Recall": 10.0,
    "Number Conversion": 4.0,
    "Verbal Fluency": 20.0,
    "Digit Span Backwards": 6.0,
}


@dataclass(frozen=True)
class ModelSpec:
    feature_set: str
    numeric_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...] = ()
    pca_cols: tuple[str, ...] = ()

    @property
    def feature_cols(self) -> list[str]:
        return list(
            dict.fromkeys([*self.numeric_cols, *self.categorical_cols, *self.pca_cols])
        )


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def lobe_simple(value: object) -> str:
    text = str(value).strip()
    if text in {"Frontal", "Temporal", "Parietal", "Insula"}:
        return text
    return "Other/sparse"


def safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return out


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    cohort = generated_cohort().rename(
        columns={
            "age_at_inclusion": "age",
            "grade": "grade_int",
            "tumor_volume_ml": "volume_ml",
        }
    )
    outcomes = generated_outcomes()
    roi = generated_regional_chaco()
    roi_cols = [col for col in roi.columns if col.startswith("chaco_roi_")]
    keep_outcomes = [
        "subject_id",
        *GLOBAL_TARGETS.values(),
        *AAT_TARGETS.values(),
        *DEMTECT_SUBTESTS.values(),
    ]
    data = (
        cohort[
            [
                "subject_id",
                "age",
                "grade_int",
                "volume_ml",
                "dominant_lobe",
                "hemisphere",
            ]
        ]
        .merge(outcomes[keep_outcomes], on="subject_id", how="left")
        .merge(roi, on="subject_id", how="left")
    )
    data["dominant_lobe_simple"] = data["dominant_lobe"].map(lobe_simple)
    data["hemisphere"] = data["hemisphere"].astype(str).str.upper().str.strip()
    data["log_volume_ml"] = np.log1p(
        pd.to_numeric(data["volume_ml"], errors="coerce").clip(lower=0)
    )
    numeric = [
        *CLINICAL,
        *GLOBAL_TARGETS.values(),
        *AAT_TARGETS.values(),
        *DEMTECT_SUBTESTS.values(),
        *roi_cols,
    ]
    return safe_numeric(data, numeric), roi_cols


def model_specs(roi_cols: list[str]) -> list[ModelSpec]:
    # The same fixed feature hierarchy is applied to every endpoint. No model is
    # selected because it happened to perform best for a particular outcome.
    return [
        ModelSpec("clinical", CLINICAL),
        ModelSpec(BASELINE_FEATURE_SET, CLINICAL, LOBE),
        ModelSpec(REGIONAL_FEATURE_SET, CLINICAL, LOBE, tuple(roi_cols)),
    ]


def input_description(spec: ModelSpec) -> str:
    if spec.feature_set == "clinical":
        return "log1p tumour volume + age + WHO grade"
    if spec.feature_set == BASELINE_FEATURE_SET:
        return "clinical + dominant lobe + lesion hemisphere"
    if spec.feature_set == REGIONAL_FEATURE_SET:
        return "clinical + dominant lobe + lesion hemisphere + fold-contained PCA of 191 regional NeMo ChaCo values"
    return spec.feature_set


def make_preprocessor(spec: ModelSpec) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(spec.numeric_cols),
            )
        )
    if spec.categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot()),
                    ]
                ),
                list(spec.categorical_cols),
            )
        )
    if spec.pca_cols:
        transformers.append(
            (
                "pca",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=0.90, svd_solver="full")),
                    ]
                ),
                list(spec.pca_cols),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(spec: ModelSpec) -> Pipeline:
    clf = LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 5),
        cv=StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
        refit=True,
    )
    return Pipeline([("preprocess", make_preprocessor(spec)), ("clf", clf)])


def model_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    return expit(np.asarray(model.decision_function(x_test), dtype=float))


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probability))
        if len(np.unique(y_true)) == 2
        else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def paired_binary_bootstrap(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, float]:
    paired = current[["subject_id", "y_true", "probability"]].merge(
        baseline[["subject_id", "y_true", "probability"]],
        on="subject_id",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if not np.array_equal(
        paired["y_true"].to_numpy(), paired["y_true_baseline"].to_numpy()
    ):
        raise RuntimeError("Binary bootstrap models do not contain identical outcomes")
    y = paired["y_true"].to_numpy(dtype=int)
    p = paired["probability"].to_numpy(dtype=float)
    p_base = paired["probability_baseline"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    ba_values: list[float] = []
    delta_auc_values: list[float] = []
    delta_ba_values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(y), size=len(y))
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        current_metrics = classification_metrics(y_b, p[idx])
        baseline_metrics = classification_metrics(y_b, p_base[idx])
        auc_values.append(float(current_metrics["auc"]))
        ba_values.append(float(current_metrics["balanced_accuracy"]))
        delta_auc_values.append(
            float(current_metrics["auc"]) - float(baseline_metrics["auc"])
        )
        delta_ba_values.append(
            float(current_metrics["balanced_accuracy"])
            - float(baseline_metrics["balanced_accuracy"])
        )
    auc_low, auc_high = percentile_interval(auc_values)
    ba_low, ba_high = percentile_interval(ba_values)
    delta_auc_low, delta_auc_high = percentile_interval(delta_auc_values)
    delta_ba_low, delta_ba_high = percentile_interval(delta_ba_values)
    return {
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high,
        "delta_auc_vs_clinical_lobe_ci_low": delta_auc_low,
        "delta_auc_vs_clinical_lobe_ci_high": delta_auc_high,
        "delta_balanced_accuracy_vs_clinical_lobe_ci_low": delta_ba_low,
        "delta_balanced_accuracy_vs_clinical_lobe_ci_high": delta_ba_high,
    }


def repeated_classifier_predictions(
    data: pd.DataFrame,
    target_col: str,
    spec: ModelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = (
        data[["subject_id", target_col, *spec.feature_cols]]
        .dropna(subset=[target_col])
        .copy()
    )
    model_data = model_data[model_data[target_col].isin([0, 1])].copy()
    y = model_data[target_col].astype(int).to_numpy()
    x = model_data[spec.feature_cols].replace([np.inf, -np.inf], np.nan)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{target_col} has fewer than two classes")
    if int(np.bincount(y).min()) < OUTER_SPLITS:
        raise RuntimeError(
            f"{target_col} has too few events for {OUTER_SPLITS} stratified folds"
        )
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED
    )
    pred_sum = np.zeros(len(model_data), dtype=float)
    pred_n = np.zeros(len(model_data), dtype=float)
    folds = []
    base_model = make_classifier(spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(x, y), start=1
        ):
            model = clone(base_model)
            model.fit(x.iloc[train_idx], y[train_idx])
            probability = model_scores(model, x.iloc[test_idx])
            pred_sum[test_idx] += probability
            pred_n[test_idx] += 1
            fold_y = y[test_idx]
            folds.append(
                {
                    "repeat": (fold_index - 1) // OUTER_SPLITS + 1,
                    "fold": (fold_index - 1) % OUTER_SPLITS + 1,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_events": int(y[train_idx].sum()),
                    "test_events": int(fold_y.sum()),
                    "train_has_both_classes": bool(len(np.unique(y[train_idx])) == 2),
                    "test_has_both_classes": bool(len(np.unique(fold_y)) == 2),
                    **classification_metrics(fold_y, probability),
                }
            )
    pred = pd.DataFrame(
        {
            "subject_id": model_data["subject_id"].to_numpy(),
            "y_true": y,
            "probability": pred_sum / np.maximum(pred_n, 1),
            "n_predictions": pred_n.astype(int),
        }
    )
    return pred, pd.DataFrame(folds)


def run_binary_models(data: pd.DataFrame, roi_cols: list[str]) -> pd.DataFrame:
    rows = []
    pred_rows = []
    fold_rows = []
    for target_label, target_col in {**GLOBAL_TARGETS, **AAT_TARGETS}.items():
        for spec in model_specs(roi_cols):
            pred, folds = repeated_classifier_predictions(data, target_col, spec)
            metrics = classification_metrics(
                pred["y_true"].to_numpy(dtype=int),
                pred["probability"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "target_label": target_label,
                    "target_column": target_col,
                    "feature_set": spec.feature_set,
                    "model": "Balanced logistic CV",
                    "inputs": input_description(spec),
                    "n": int(len(pred)),
                    "events": int(pred["y_true"].sum()),
                    **metrics,
                    "numeric_cols": ";".join(spec.numeric_cols),
                    "categorical_cols": ";".join(spec.categorical_cols),
                    "pca_cols_n": len(spec.pca_cols),
                    "pca_cols": ";".join(spec.pca_cols),
                    "model_role": {
                        "clinical": "clinical-only sensitivity",
                        BASELINE_FEATURE_SET: "reference baseline",
                        REGIONAL_FEATURE_SET: "regional disconnection extension",
                    }[spec.feature_set],
                    "selection_rule": "uniform feature-set comparison; no endpoint-specific performance selection",
                    "exploratory": True,
                    "validation": (
                        "5 x 5 repeated stratified CV; inner 3-fold C tuning; class weights; "
                        "2,000 patient-level bootstrap resamples of averaged out-of-fold predictions"
                    ),
                }
            )
            pred_rows.extend(
                pred.assign(
                    target_label=target_label,
                    target_column=target_col,
                    feature_set=spec.feature_set,
                ).to_dict(orient="records")
            )
            fold_rows.extend(
                folds.assign(
                    target_label=target_label,
                    target_column=target_col,
                    feature_set=spec.feature_set,
                ).to_dict(orient="records")
            )

    model_screen = pd.DataFrame(rows)
    predictions = pd.DataFrame(pred_rows)
    baseline_metrics = model_screen[
        model_screen["feature_set"].eq(BASELINE_FEATURE_SET)
    ][["target_label", "auc", "balanced_accuracy"]].rename(
        columns={
            "auc": "clinical_lobe_auc",
            "balanced_accuracy": "clinical_lobe_balanced_accuracy",
        }
    )
    model_screen = model_screen.merge(
        baseline_metrics, on="target_label", how="left", validate="many_to_one"
    )
    model_screen["delta_auc_vs_clinical_lobe"] = (
        model_screen["auc"] - model_screen["clinical_lobe_auc"]
    )
    model_screen["delta_balanced_accuracy_vs_clinical_lobe"] = (
        model_screen["balanced_accuracy"]
        - model_screen["clinical_lobe_balanced_accuracy"]
    )

    interval_rows = []
    target_order = list({**GLOBAL_TARGETS, **AAT_TARGETS})
    feature_order = [spec.feature_set for spec in model_specs(roi_cols)]
    for target_index, target_label in enumerate(target_order):
        baseline = predictions[
            predictions["target_label"].eq(target_label)
            & predictions["feature_set"].eq(BASELINE_FEATURE_SET)
        ]
        for feature_index, feature_set in enumerate(feature_order):
            current = predictions[
                predictions["target_label"].eq(target_label)
                & predictions["feature_set"].eq(feature_set)
            ]
            intervals = paired_binary_bootstrap(
                current,
                baseline,
                seed=SEED + 100 * target_index + feature_index,
            )
            interval_rows.append(
                {"target_label": target_label, "feature_set": feature_set, **intervals}
            )
    model_screen = model_screen.merge(
        pd.DataFrame(interval_rows),
        on=["target_label", "feature_set"],
        how="left",
        validate="one_to_one",
    )
    model_screen = model_screen.sort_values(["target_label", "feature_set"])
    model_screen.to_csv(OUT / "binary_model_screen_results.csv", index=False)
    predictions.to_csv(OUT / "binary_cv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "binary_cv_folds.csv", index=False)
    reported = model_screen[model_screen["feature_set"].eq(REGIONAL_FEATURE_SET)].copy()
    reported.to_csv(OUT / "reported_binary_model_details.csv", index=False)
    return model_screen


def make_regressor(spec: ModelSpec) -> Pipeline:
    estimator = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return Pipeline([("preprocess", make_preprocessor(spec)), ("model", estimator)])


@lru_cache(maxsize=None)
def pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def continuous_c_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    i, j = pair_indices(len(y_true))
    dy = y_true[i] - y_true[j]
    dp = y_pred[i] - y_pred[j]
    valid = np.isfinite(dy) & np.isfinite(dp) & (dy != 0)
    if not np.any(valid):
        return np.nan
    dy = dy[valid]
    dp = dp[valid]
    concordance = (dy * dp > 0).astype(float) + 0.5 * (dp == 0)
    return float(np.mean(concordance))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.nanmean(x, axis=1, keepdims=True)
    y_centered = y - np.nanmean(y, axis=1, keepdims=True)
    denom = np.sqrt(np.nansum(x_centered**2, axis=1) * np.nansum(y_centered**2, axis=1))
    out = np.full(x.shape[0], np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0)
    out[valid] = np.nansum(x_centered[valid] * y_centered[valid], axis=1) / denom[valid]
    return out


def profile_metrics(
    y: np.ndarray, pred: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    profile_r = rowwise_pearson(y / upper, pred / upper)
    total_y = y.sum(axis=1)
    total_pred = pred.sum(axis=1)
    sub_r2 = [float(r2_score(y[:, j], pred[:, j])) for j in range(y.shape[1])]
    return {
        "median_profile_r": float(np.nanmedian(profile_r)),
        "summed_raw_component_score_c_index": continuous_c_index(
            total_y, total_pred
        ),
        "summed_raw_component_score_spearman": safe_spearman(total_y, total_pred),
        "summed_raw_component_score_mae": float(
            mean_absolute_error(total_y, total_pred)
        ),
        "mean_score_r2": float(np.mean(sub_r2)),
        "mean_score_rmse": float(
            np.mean(np.sqrt(mean_squared_error(y, pred, multioutput="raw_values")))
        ),
}


def fold_mean_profile_predictions(
    y: np.ndarray, y_strat: np.ndarray
) -> np.ndarray:
    """Predict each held-out profile with its corresponding training-fold mean."""
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED
    )
    pred_sum = np.zeros_like(y, dtype=float)
    pred_n = np.zeros(y.shape[0], dtype=float)
    placeholder_x = np.zeros((y.shape[0], 1), dtype=float)
    for train_idx, test_idx in splitter.split(placeholder_x, y_strat):
        pred_sum[test_idx] += np.mean(y[train_idx], axis=0)
        pred_n[test_idx] += 1
    return pred_sum / np.maximum(pred_n[:, None], 1)


def profile_bootstrap_intervals(
    y: np.ndarray,
    pred: np.ndarray,
    upper: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(y), size=len(y))
        values.append(profile_metrics(y[idx], pred[idx], upper)["median_profile_r"])
    low, high = percentile_interval(values)
    return {
        "fold_mean_profile_benchmark_median_profile_r_ci_low": low,
        "fold_mean_profile_benchmark_median_profile_r_ci_high": high,
    }


def subtest_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": safe_spearman(y, pred),
        "c_index": continuous_c_index(y, pred),
    }


def paired_profile_bootstrap(
    y: np.ndarray,
    current: np.ndarray,
    baseline: np.ndarray,
    mean_profile_benchmark: np.ndarray,
    upper: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    profile_values: dict[str, list[float]] = {
        key: []
        for key in [
            "median_profile_r",
            "summed_raw_component_score_c_index",
            "summed_raw_component_score_spearman",
            "mean_score_r2",
        ]
    }
    profile_delta: dict[str, list[float]] = {
        "delta_median_profile_r_vs_clinical_lobe": [],
        "delta_summed_raw_component_score_c_index_vs_clinical_lobe": [],
        "delta_median_profile_r_vs_fold_mean_profile_benchmark": [],
    }
    subtest_values: list[dict[str, list[float]]] = [
        dict(c_index=[], spearman=[], r2=[]) for _ in range(y.shape[1])
    ]
    subtest_delta: list[dict[str, list[float]]] = [
        dict(c_index=[]) for _ in range(y.shape[1])
    ]
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(y), size=len(y))
        current_profile = profile_metrics(y[idx], current[idx], upper)
        baseline_profile = profile_metrics(y[idx], baseline[idx], upper)
        benchmark_profile = profile_metrics(
            y[idx], mean_profile_benchmark[idx], upper
        )
        for key in profile_values:
            profile_values[key].append(current_profile[key])
        profile_delta["delta_median_profile_r_vs_clinical_lobe"].append(
            current_profile["median_profile_r"]
            - baseline_profile["median_profile_r"]
        )
        profile_delta[
            "delta_summed_raw_component_score_c_index_vs_clinical_lobe"
        ].append(
            current_profile["summed_raw_component_score_c_index"]
            - baseline_profile["summed_raw_component_score_c_index"]
        )
        profile_delta[
            "delta_median_profile_r_vs_fold_mean_profile_benchmark"
        ].append(
            current_profile["median_profile_r"]
            - benchmark_profile["median_profile_r"]
        )
        for j in range(y.shape[1]):
            current_subtest = subtest_metrics(y[idx, j], current[idx, j])
            baseline_subtest = subtest_metrics(y[idx, j], baseline[idx, j])
            for key in subtest_values[j]:
                subtest_values[j][key].append(current_subtest[key])
            subtest_delta[j]["c_index"].append(
                current_subtest["c_index"] - baseline_subtest["c_index"]
            )

    profile_intervals: dict[str, float] = {}
    for key, values in profile_values.items():
        low, high = percentile_interval(values)
        profile_intervals[f"{key}_ci_low"] = low
        profile_intervals[f"{key}_ci_high"] = high
    for column, values in profile_delta.items():
        low, high = percentile_interval(values)
        profile_intervals[f"{column}_ci_low"] = low
        profile_intervals[f"{column}_ci_high"] = high

    subtest_intervals: list[dict[str, float]] = []
    for j in range(y.shape[1]):
        result: dict[str, float] = {}
        for key, values in subtest_values[j].items():
            low, high = percentile_interval(values)
            result[f"{key}_ci_low"] = low
            result[f"{key}_ci_high"] = high
        low, high = percentile_interval(subtest_delta[j]["c_index"])
        result["delta_c_index_vs_clinical_lobe_ci_low"] = low
        result["delta_c_index_vs_clinical_lobe_ci_high"] = high
        subtest_intervals.append(result)
    return profile_intervals, subtest_intervals


def run_demtect_profile_models(
    data: pd.DataFrame, roi_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_labels = list(DEMTECT_SUBTESTS)
    target_cols = [DEMTECT_SUBTESTS[label] for label in target_labels]
    upper = np.asarray([DEMTECT_MAX[label] for label in target_labels], dtype=float)
    model_data = data[["subject_id", "DemTect_impaired_binary", *target_cols]].dropna(
        subset=target_cols
    )
    model_data = model_data[model_data["DemTect_impaired_binary"].isin([0, 1])].copy()
    y = model_data[target_cols].to_numpy(dtype=float)
    y_strat = model_data["DemTect_impaired_binary"].astype(int).to_numpy()
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED
    )
    predictions_by_feature: dict[str, np.ndarray] = {}
    prediction_rows = []

    mean_profile_benchmark = np.clip(
        fold_mean_profile_predictions(y, y_strat), 0.0, upper
    )
    benchmark_profile = profile_metrics(y, mean_profile_benchmark, upper)
    benchmark_intervals = profile_bootstrap_intervals(
        y,
        mean_profile_benchmark,
        upper,
        seed=SEED + 900,
    )
    for i, sid in enumerate(model_data["subject_id"]):
        for j, label in enumerate(target_labels):
            prediction_rows.append(
                {
                    "subject_id": sid,
                    "subtest": label,
                    "model": "Fold-contained training-set mean profile",
                    "feature_set": MEAN_PROFILE_BENCHMARK,
                    "observed": y[i, j],
                    "predicted": mean_profile_benchmark[i, j],
                }
            )

    for spec in model_specs(roi_cols):
        x = data.loc[model_data.index, spec.feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        pred_sum = np.zeros_like(y, dtype=float)
        pred_n = np.zeros(y.shape[0], dtype=float)
        base_model = make_regressor(spec)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for train_idx, test_idx in splitter.split(x, y_strat):
                model = clone(base_model)
                model.fit(x.iloc[train_idx], y[train_idx])
                pred_sum[test_idx] += np.asarray(
                    model.predict(x.iloc[test_idx]), dtype=float
                )
                pred_n[test_idx] += 1
        pred_mean = np.clip(pred_sum / np.maximum(pred_n[:, None], 1), 0.0, upper)
        predictions_by_feature[spec.feature_set] = pred_mean
        for i, sid in enumerate(model_data["subject_id"]):
            for j, label in enumerate(target_labels):
                prediction_rows.append(
                    {
                        "subject_id": sid,
                        "subtest": label,
                        "model": "Multi-output RidgeCV",
                        "feature_set": spec.feature_set,
                        "observed": y[i, j],
                        "predicted": pred_mean[i, j],
                    }
                )

    baseline = predictions_by_feature[BASELINE_FEATURE_SET]
    profile_rows = []
    subtest_rows = []
    for feature_index, spec in enumerate(model_specs(roi_cols)):
        pred_mean = predictions_by_feature[spec.feature_set]
        point_profile = profile_metrics(y, pred_mean, upper)
        baseline_profile = profile_metrics(y, baseline, upper)
        profile_intervals, subtest_intervals = paired_profile_bootstrap(
            y,
            pred_mean,
            baseline,
            mean_profile_benchmark,
            upper,
            seed=SEED + 1_000 + feature_index,
        )
        profile_rows.append(
            {
                "battery": "DemTect",
                "model": "Multi-output RidgeCV",
                "feature_set": spec.feature_set,
                "inputs": input_description(spec),
                "n": int(y.shape[0]),
                **point_profile,
                "fold_mean_profile_benchmark_median_profile_r": (
                    benchmark_profile["median_profile_r"]
                ),
                **benchmark_intervals,
                "delta_median_profile_r_vs_clinical_lobe": (
                    point_profile["median_profile_r"]
                    - baseline_profile["median_profile_r"]
                ),
                "delta_median_profile_r_vs_fold_mean_profile_benchmark": (
                    point_profile["median_profile_r"]
                    - benchmark_profile["median_profile_r"]
                ),
                "delta_summed_raw_component_score_c_index_vs_clinical_lobe": (
                    point_profile["summed_raw_component_score_c_index"]
                    - baseline_profile["summed_raw_component_score_c_index"]
                ),
                **profile_intervals,
                "selection_rule": "uniform feature-set comparison with a fixed RidgeCV estimator",
                "exploratory": True,
            }
        )
        for j, label in enumerate(target_labels):
            point = subtest_metrics(y[:, j], pred_mean[:, j])
            base_point = subtest_metrics(y[:, j], baseline[:, j])
            subtest_rows.append(
                {
                    "battery": "DemTect",
                    "subtest": label,
                    "model": "Multi-output RidgeCV",
                    "feature_set": spec.feature_set,
                    "inputs": input_description(spec),
                    "n": int(y.shape[0]),
                    **point,
                    "delta_c_index_vs_clinical_lobe": point["c_index"]
                    - base_point["c_index"],
                    **subtest_intervals[j],
                    "selection_rule": "same fixed model used for all five subtests",
                    "exploratory": True,
                }
            )

    profile = pd.DataFrame(profile_rows).sort_values("feature_set")
    subtests = pd.DataFrame(subtest_rows).sort_values(["subtest", "feature_set"])
    predictions = pd.DataFrame(prediction_rows)
    profile.to_csv(OUT / "demtect_profile_model_summary.csv", index=False)
    subtests.to_csv(OUT / "demtect_subtest_model_summary.csv", index=False)
    predictions.to_csv(OUT / "demtect_profile_cv_predictions_long.csv", index=False)
    reported_subtests = subtests[
        subtests["feature_set"].eq(REGIONAL_FEATURE_SET)
    ].copy()
    reported_subtests.to_csv(
        OUT / "reported_demtect_subtest_model_details.csv", index=False
    )
    return profile, reported_subtests


def target_text(row: pd.Series, label: str) -> str:
    if label == "Mean AAT T-score":
        return f"<63.5; {int(row.events)}/{int(row.n)} below threshold"
    if label == "DemTect global":
        return f"global score < 13; {int(row.events)}/{int(row.n)} impaired"
    return f"{int(row.events)}/{int(row.n)} impaired"


def interval_text(value: float, low: float, high: float) -> str:
    return f"{value:.3f} (95% CI {low:.3f} to {high:.3f})"


def write_table_03(
    model_screen: pd.DataFrame, profile: pd.DataFrame, demtect_subtests: pd.DataFrame
) -> None:
    """Export one continuous table with the same columns for every outcome."""
    columns = ["Target", "Metric", "Clinical only", "Clinical + location",
               "Regional extension", "Difference vs clinical + location"]
    rows = []

    def interval(value: float, low: float, high: float) -> str:
        return f"{value:.3f}\n({low:.3f} to {high:.3f})"

    def estimate(row: pd.Series, field: str) -> str:
        return interval(row[field], row[f"{field}_ci_low"], row[f"{field}_ci_high"])

    def result_row(target: str, metric: str, **values: str) -> dict[str, str]:
        return {**dict.fromkeys(columns, "—"), "Target": target, "Metric": metric, **values}

    for target_label in [*GLOBAL_TARGETS, *AAT_TARGETS]:
        selected = model_screen[model_screen["target_label"].eq(target_label)].set_index("feature_set")
        if not selected.index.is_unique or not {"clinical", BASELINE_FEATURE_SET, REGIONAL_FEATURE_SET}.issubset(selected.index):
            raise ValueError(f"Expected all three feature sets for {target_label}")
        clinical, baseline, regional = (selected.loc[key] for key in ["clinical", BASELINE_FEATURE_SET, REGIONAL_FEATURE_SET])
        target = f"{target_label}: {target_text(regional, target_label)}"
        for metric, field in [("AUC", "auc"), ("Balanced accuracy", "balanced_accuracy")]:
            rows.append(result_row(target, metric, **{
                "Clinical only": estimate(clinical, field) if metric == "AUC" else "—",
                "Clinical + location": estimate(baseline, field),
                "Regional extension": estimate(regional, field),
                "Difference vs clinical + location": estimate(regional, f"delta_{field}_vs_clinical_lobe"),
            }))

    regional = profile[profile["feature_set"].eq(REGIONAL_FEATURE_SET)].iloc[0]
    baseline = profile[profile["feature_set"].eq(BASELINE_FEATURE_SET)].iloc[0]
    for metric, field, delta in [
        ("Median within-patient profile r", "median_profile_r", "delta_median_profile_r_vs_clinical_lobe"),
        ("Summed raw subtest-score C-index", "summed_raw_component_score_c_index", "delta_summed_raw_component_score_c_index_vs_clinical_lobe"),
    ]:
        row = result_row(f"DemTect profile; n = {int(regional.n)}", metric, **{
            "Clinical + location": estimate(baseline, field),
            "Regional extension": estimate(regional, field),
            "Difference vs clinical + location": estimate(regional, delta),
        })
        rows.append(row)

    for metric, field in [("C-index", "c_index"), ("Spearman rho", "spearman"), ("R²", "r2")]:
        row = result_row(f"DemTect individual subtests (range); n = {int(demtect_subtests.n.iloc[0])}", metric, **{
            "Regional extension": f"{demtect_subtests[field].min():.3f} to {demtect_subtests[field].max():.3f}",
        })
        if field == "c_index":
            delta = demtect_subtests["delta_c_index_vs_clinical_lobe"]
            row["Difference vs clinical + location"] = f"{delta.min():.3f} to {delta.max():.3f}"
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(OUT / "table_03_prediction_models.csv", index=False)

def main() -> None:
    for filename in OBSOLETE_OUTPUTS:
        (OUT / filename).unlink(missing_ok=True)
    data, roi_columns = prepare_data()
    binary_model_screen = run_binary_models(data, roi_columns)
    profile_summary, reported_demtect_subtests = run_demtect_profile_models(
        data, roi_columns
    )
    write_table_03(binary_model_screen, profile_summary, reported_demtect_subtests)
    print(
        "OK: computed graph-free exploratory prediction models from cohort and regional NeMo ChaCo"
    )


if __name__ == "__main__":
    main()
