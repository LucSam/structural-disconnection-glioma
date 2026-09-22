#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch
from scipy import stats  # type: ignore[import-untyped]

from _table_exports import write_table_02
from _shared import ANATOMICAL_GROUP_ORDER, generated_cohort, generated_outcomes, load_fs191_labels, load_nemo_edge_matrices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/tfnbs"
MRTRIX_OUT = OUT / "mrtrix_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MRTRIX_OUT.mkdir(parents=True, exist_ok=True)

CONNECTOMESTATS = os.environ.get("CONNECTOMESTATS", "connectomestats")
N_SHUFFLES = int(os.environ.get("ICONS_TFNBS_NSHUFFLES", "5000"))
TFCE_DH = float(os.environ.get("ICONS_TFNBS_DH", "0.1"))
TFCE_E = float(os.environ.get("ICONS_TFNBS_E", "0.4"))
TFCE_H = float(os.environ.get("ICONS_TFNBS_H", "3"))
NTHREADS = os.environ.get("ICONS_TFNBS_NTHREADS", "2")
REUSE = os.environ.get("ICONS_TFNBS_REUSE", "0") == "1"
PREPARE_ONLY = os.environ.get("ICONS_TFNBS_PREPARE_ONLY", "0") == "1"
PARALLEL_RUNS = int(os.environ.get("ICONS_TFNBS_PARALLEL_RUNS", "1"))
RERUN_FAMILIES = {
    family.strip()
    for family in os.environ.get("ICONS_TFNBS_RERUN_FAMILIES", "clinical,lobe").split(",")
    if family.strip()
}
PERMUTATION_BASE_SEED = int(os.environ.get("ICONS_TFNBS_PERMUTATION_SEED", "20260710"))
P_FLOOR = 1.0 / max(N_SHUFFLES, 1)
MAX_DISPLAY_EDGES = int(os.environ.get("ICONS_TFNBS_MAX_DISPLAY_EDGES", "160"))

MODELS = [
    {
        "family": "clinical",
        "dir": "clinical_covariates_deficit_only",
        "covariates": ["age_years", "grade_int", "log_tumor_volume_ml"],
    },
    {
        "family": "lobe",
        "dir": "clinical_lobe_hemisphere_deficit_only",
        "covariates": [
            "age_years",
            "grade_int",
            "log_tumor_volume_ml",
            "hemisphere_R",
            "lobe_pct_frontal",
            "lobe_pct_temporal",
            "lobe_pct_parietal",
            "lobe_pct_occipital",
            "lobe_pct_insular",
            "lobe_pct_thalamic",
        ],
    },
]
SUBTESTS = [
    ("AAT", "Token Test", "Token Test | T Score (p. 133)", "aat_token_test", 4),
    ("AAT", "Repetition", "repetition | average T Score across both tests", "aat_repetition", 4),
    ("AAT", "Naming", "naming | T Score (p. 135)", "aat_naming", 4),
    ("AAT", "Comprehension", "comprehension | T Score (p. 136)", "aat_comprehension", 4),
    ("DemTect", "Word List", "word list | first pass - out of 20", "demtect_word_list", 5),
    ("DemTect", "Delayed Recall", "word list | delayed recall - out of 10", "demtect_delayed_recall", 5),
    ("DemTect", "Number Conversion", "number conversion | out of 4", "demtect_number_conversion", 5),
    ("DemTect", "Verbal Fluency", "verbal fluency | out of 20", "demtect_verbal_fluency", 5),
    ("DemTect", "Digit Span Backwards", "digit span backwards | out of 6", "demtect_digit_span_backwards", 5),
]
ANATOMICAL_ORDER = [group for group, _code in sorted(ANATOMICAL_GROUP_ORDER.items(), key=lambda item: item[1])]
ANATOMICAL_COLORS = {
    "Subcortical/limbic": "#1f77b4",
    "Cerebellum": "#8c564b",
    "Frontal/insula": "#d62728",
    "Perisylvian/central sulci": "#bcbd22",
    "Temporal": "#9467bd",
    "Parietal": "#2ca02c",
    "Occipital/visual": "#17becf",
    "Cingulate/medial": "#ff7f0e",
}
FIGURE_SUBTESTS = [
    ("AAT", "Token Test"),
    ("AAT", "Repetition"),
    ("AAT", "Naming"),
    ("DemTect", "Delayed Recall"),
    ("DemTect", "Number Conversion"),
    ("DemTect", "Verbal Fluency"),
]


def rank_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = stats.rankdata(values, method="average")
    sd = float(ranks.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros(len(values), dtype=np.float32)
    return ((ranks - float(ranks.mean())) / sd).astype(np.float32)


def rank_normalize_edges(edges: np.ndarray) -> np.ndarray:
    out = np.empty_like(edges, dtype=np.float32)
    for idx in range(edges.shape[1]):
        out[:, idx] = rank_z(edges[:, idx])
    return out


def write_connectome_inputs(run_dir: Path, subject_ids: np.ndarray, rank_edges: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> Path:
    mat_dir = run_dir / "mats"
    if mat_dir.exists():
        shutil.rmtree(mat_dir)
    mat_dir.mkdir(parents=True)
    n_nodes = 191
    rel_paths = []
    for row_idx, sid in enumerate(subject_ids):
        matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        matrix[edge_i, edge_j] = rank_edges[row_idx]
        matrix[edge_j, edge_i] = rank_edges[row_idx]
        path = mat_dir / f"{sid}.csv"
        np.savetxt(path, matrix, delimiter=",", fmt="%.7g")
        rel_paths.append(f"mats/{path.name}")
    input_path = run_dir / "input_connectomes.txt"
    input_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
    return input_path


def write_design(run_dir: Path, scores: np.ndarray, covariates: pd.DataFrame, covariate_names: list[str]) -> tuple[Path, list[str]]:
    columns = [np.ones(len(scores), dtype=np.float32), rank_z(scores)]
    names = ["intercept", "subtest_score_rank_z"]
    for name in covariate_names:
        values = pd.to_numeric(covariates[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Missing values remain in TFNBS covariate {name}")
        columns.append(rank_z(values))
        names.append(f"{name}_rank_z")
    design = np.column_stack(columns).astype(np.float32)
    design_path = run_dir / "design.txt"
    np.savetxt(design_path, design, fmt="%.7g")
    (run_dir / "design_columns.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return design_path, names


def write_contrast(run_dir: Path, n_columns: int) -> Path:
    contrast = np.zeros((1, n_columns), dtype=np.float32)
    contrast[0, 1] = -1.0
    contrast_path = run_dir / "contrast.txt"
    np.savetxt(contrast_path, contrast, fmt="%.7g")
    return contrast_path


def write_permutation_seed(run_dir: Path, subject_ids: np.ndarray, run_key: str) -> tuple[Path, int, str]:
    seed_material = f"{PERMUTATION_BASE_SEED}|{run_key}|" + "|".join(subject_ids.astype(str))
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "little") % (2**32)
    specification_checksum = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    metadata = {
        "generator": "MRtrix internal shuffler controlled by MRTRIX_RNG_SEED",
        "base_seed": PERMUTATION_BASE_SEED,
        "derived_seed": derived_seed,
        "n_subjects": len(subject_ids),
        "n_permutations": N_SHUFFLES,
        "nthreads": int(NTHREADS),
        "seed_specification_sha256": specification_checksum,
    }
    path = run_dir / "permutation_seed.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path, derived_seed, specification_checksum


def run_connectomestats(
    input_path: Path,
    design_path: Path,
    contrast_path: Path,
    permutation_seed: int,
    output_prefix: Path,
) -> str:
    command = [
        CONNECTOMESTATS,
        "-force",
        "-nshuffles",
        str(N_SHUFFLES),
        "-tfce_dh",
        str(TFCE_DH),
        "-tfce_e",
        str(TFCE_E),
        "-tfce_h",
        str(TFCE_H),
    ]
    if NTHREADS:
        command.extend(["-nthreads", str(NTHREADS)])
    command.extend([str(input_path), "tfnbs", str(design_path), str(contrast_path), str(output_prefix)])
    environment = os.environ.copy()
    environment["MRTRIX_RNG_SEED"] = str(permutation_seed)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=output_prefix.parent,
        env=environment,
    )
    return (result.stdout or "") + (result.stderr or "")


def run_prepared_tfnbs() -> None:
    prepared = []
    for model in MODELS:
        if str(model["family"]) not in RERUN_FAMILIES:
            continue
        for _battery, _label, _score_col, slug, _n_battery in SUBTESTS:
            run_dir = MRTRIX_OUT / str(model["dir"]) / slug
            metadata = json.loads((run_dir / "permutation_seed.json").read_text(encoding="utf-8"))
            prepared.append(
                (
                    run_dir / "input_connectomes.txt",
                    run_dir / "design.txt",
                    run_dir / "contrast.txt",
                    int(metadata["derived_seed"]),
                    run_dir / "tfnbs_",
                )
            )

    def execute(paths: tuple[Path, Path, Path, int, Path]) -> Path:
        input_path, design_path, contrast_path, seed, output_prefix = paths
        log = run_connectomestats(input_path, design_path, contrast_path, seed, output_prefix)
        (output_prefix.parent / "connectomestats.log").write_text(log, encoding="utf-8")
        return output_prefix.parent

    with ThreadPoolExecutor(max_workers=PARALLEL_RUNS) as executor:
        futures = [executor.submit(execute, paths) for paths in prepared]
        for future in as_completed(futures):
            print(f"Completed TFNBS: {future.result().relative_to(ROOT)}", flush=True)


def anatomical_pair(group_a: object, group_b: object) -> str:
    groups = sorted([str(group_a), str(group_b)], key=lambda group: ANATOMICAL_GROUP_ORDER.get(group, 999))
    return " - ".join(groups)


def load_edge_output(prefix: Path, suffix: str, edge_i: np.ndarray, edge_j: np.ndarray) -> np.ndarray:
    matrix = np.loadtxt(Path(str(prefix) + suffix), delimiter=",")
    return matrix[edge_i, edge_j]


def circle_layout(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float | str]]]:
    meta = labels[["roi_index", "roi_name", "anatomical_group"]].copy()
    meta["group_code"] = meta["anatomical_group"].map(ANATOMICAL_GROUP_ORDER).fillna(999).astype(int)
    meta["hemisphere_code"] = (
        meta["roi_name"].str.extract(r"\(([LRV])\)", expand=False).map({"L": 0, "V": 1, "R": 2}).fillna(1)
    )
    meta = meta.sort_values(["group_code", "hemisphere_code", "roi_name", "roi_index"]).reset_index(drop=True)
    group_gap = 0.075
    span = 2 * np.pi - group_gap * len(ANATOMICAL_ORDER)
    theta = np.zeros(len(meta), dtype=float)
    bounds: list[dict[str, float | str]] = []
    cursor = np.pi / 2
    for group in ANATOMICAL_ORDER:
        index = meta.index[meta["anatomical_group"].eq(group)].to_numpy()
        if index.size == 0:
            continue
        group_span = span * index.size / len(meta)
        values = np.linspace(cursor, cursor - group_span, index.size, endpoint=False)
        values -= group_span / index.size / 2
        theta[index] = values
        bounds.append({"group": group, "start": cursor, "end": cursor - group_span})
        cursor -= group_span + group_gap
    meta["x"] = np.cos(theta)
    meta["y"] = np.sin(theta)
    return meta, bounds


def draw_edge_panel(
    ax: plt.Axes,
    selected: pd.DataFrame,
    layout: pd.DataFrame,
    bounds: list[dict[str, float | str]],
    title: str,
    n_fwe: int,
    n_battery: int,
    vmax: float,
) -> None:
    positions = layout.set_index("roi_index")[["x", "y", "anatomical_group"]].to_dict("index")
    strength = {int(roi): 0.0 for roi in layout["roi_index"]}
    cmap = plt.get_cmap("magma_r")
    norm = mcolors.Normalize(0.0, max(vmax, 1e-8))
    for row in selected.sort_values("abs_t").itertuples(index=False):
        roi_i, roi_j, value = int(row.roi_i), int(row.roi_j), float(row.abs_t)
        if roi_i not in positions or roi_j not in positions:
            continue
        p0 = (float(positions[roi_i]["x"]), float(positions[roi_i]["y"]))
        p1 = (float(positions[roi_j]["x"]), float(positions[roi_j]["y"]))
        strength[roi_i] += value
        strength[roi_j] += value
        path = MplPath([p0, (0.0, 0.0), p1], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(
                PathPatch(path, facecolor="none", edgecolor=cmap(norm(value)), lw=0.25 + 1.75 * value / max(vmax, 1e-8), alpha=0.30)
        )
    max_strength = max(strength.values(), default=1.0) or 1.0
    for row in layout.itertuples(index=False):
        size = 4.0 + 28.0 * strength[int(row.roi_index)] / max_strength
        ax.scatter(row.x, row.y, s=size, c=ANATOMICAL_COLORS.get(row.anatomical_group, "#7f7f7f"), edgecolors="white", linewidths=0.18, zorder=3)
    for bound in bounds:
        theta = np.linspace(float(bound["start"]), float(bound["end"]), 80)
        ax.plot(1.05 * np.cos(theta), 1.05 * np.sin(theta), color=ANATOMICAL_COLORS[str(bound["group"])], lw=2.0)
    shown = len(selected)
    display = f"FWE={n_fwe:,}; battery={n_battery:,}; shown={shown:,}"
    if n_fwe > shown:
        display = f"FWE={n_fwe:,}; battery={n_battery:,}; top {shown:,}"
    ax.set_title(f"{title}\n{display}", fontsize=10.0, fontweight="bold", pad=7)
    if selected.empty:
        ax.text(0, 0, "No FWE\nedges", ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")
    ax.set(xlim=(-1.16, 1.16), ylim=(-1.16, 1.16), aspect="equal")
    ax.axis("off")


def generate_tfnbs_figure(edges: pd.DataFrame, counts: pd.DataFrame, labels: pd.DataFrame) -> Path:
    layout, bounds = circle_layout(labels)
    models = [("clinical", "Clinical covariates"), ("lobe", "Clinical + lobe location + hemisphere")]
    selected: dict[tuple[str, str, str], pd.DataFrame] = {}
    vmax = 1e-8
    for model_family, _model_label in models:
        for battery, subtest in FIGURE_SUBTESTS:
            subset = edges[
                edges["model_family"].eq(model_family)
                & edges["battery"].eq(battery)
                & edges["subtest_label"].eq(subtest)
                & edges["significant_fwe_edges"].astype(bool)
            ].copy()
            subset["abs_t"] = pd.to_numeric(subset["t_value"], errors="coerce").abs()
            subset = subset.sort_values(["p_fwe_edges", "abs_t"], ascending=[True, False]).head(MAX_DISPLAY_EDGES)
            selected[(model_family, battery, subtest)] = subset
            if not subset.empty:
                vmax = max(vmax, float(subset["abs_t"].max()))
    fig, axes = plt.subplots(2, len(FIGURE_SUBTESTS), figsize=(18.2, 7.7))
    axes_grid = np.asarray(axes, dtype=object)
    count_index = counts.set_index(["model_family", "battery", "subtest_label"])
    for row_idx, (model_family, model_label) in enumerate(models):
        for col_idx, (battery, subtest) in enumerate(FIGURE_SUBTESTS):
            key = (model_family, battery, subtest)
            summary = count_index.loc[key]
            draw_edge_panel(
                axes_grid[row_idx, col_idx],
                selected[key],
                layout,
                bounds,
                subtest if row_idx == 0 else "",
                int(summary["n_fwe05_edges"]),
                int(summary["n_battery_bonf05_edges"]),
                vmax,
            )
            if col_idx == 0:
                axes_grid[row_idx, col_idx].text(-1.40, 0, model_label, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
    legend = [Patch(facecolor=ANATOMICAL_COLORS[group], label=group) for group in ANATOMICAL_ORDER]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, prop={"size": 9.8, "weight": "bold"})
    cbar_ax = fig.add_axes((0.962, 0.24, 0.016, 0.54))
    scalar = ScalarMappable(norm=mcolors.Normalize(0, vmax), cmap="magma_r")
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.set_label("displayed edge |t|", fontsize=15.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=15.0, width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.suptitle("Deficit-only continuous subtest TFNBS edge patterns", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.08, 0.955, 0.925))
    path = ROOT / "outputs/figures/figure_04_subtest_tfnbs_edge_configurations.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if not REUSE and not PREPARE_ONLY:
    environment = os.environ.copy()
    environment["ICONS_TFNBS_PREPARE_ONLY"] = "1"
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True, env=environment)
    run_prepared_tfnbs()
    REUSE = True


outcomes = generated_outcomes()
cohort = generated_cohort()
cohort_vars = cohort[
    [
        "subject_id",
        "age_at_inclusion",
        "grade",
        "tumor_volume_ml",
        "hemisphere",
        "tumour_location_frontal_%",
        "tumour_location_temporal_%",
        "tumour_location_parietal_%",
        "tumour_location_occipital_%",
        "tumour_location_insular_%",
        "tumour_location_thalamic_%",
    ]
].rename(
    columns={
        "age_at_inclusion": "age_years",
        "grade": "grade_int",
        "tumour_location_frontal_%": "lobe_pct_frontal",
        "tumour_location_temporal_%": "lobe_pct_temporal",
        "tumour_location_parietal_%": "lobe_pct_parietal",
        "tumour_location_occipital_%": "lobe_pct_occipital",
        "tumour_location_insular_%": "lobe_pct_insular",
        "tumour_location_thalamic_%": "lobe_pct_thalamic",
    }
)
cohort_vars["log_tumor_volume_ml"] = np.log1p(pd.to_numeric(cohort_vars["tumor_volume_ml"], errors="coerce").clip(lower=0))
cohort_vars["hemisphere_R"] = cohort_vars["hemisphere"].astype(str).str.upper().eq("R").astype(float)
cohort_vars = cohort_vars.drop(columns=["hemisphere"])
for col in cohort_vars.columns:
    if col != "subject_id":
        cohort_vars[col] = pd.to_numeric(cohort_vars[col], errors="coerce")
analysis = outcomes.merge(cohort_vars, on="subject_id", how="inner")

_remaining_sc, chaco_edge = load_nemo_edge_matrices()
subject_ids = sorted(set(analysis["subject_id"]).intersection(chaco_edge))
upper = np.triu_indices(191, k=1)
edge_i, edge_j = upper[0], upper[1]
edge_matrix = np.vstack([chaco_edge[sid][upper] for sid in subject_ids]).astype(np.float32)
aligned = analysis.set_index("subject_id").loc[subject_ids].reset_index()

labels = load_fs191_labels().set_index("roi_index")
roi_names = labels["roi_name"].to_dict()
anatomical_groups = labels["anatomical_group"].to_dict()

design_rows = []
count_rows = []
edge_rows = []
for model in MODELS:
    if PREPARE_ONLY and str(model["family"]) not in RERUN_FAMILIES:
        continue
    for battery, label, score_col, slug, n_battery in SUBTESTS:
        covariates = list(model["covariates"])
        include = aligned[[score_col, *covariates]].notna().all(axis=1).to_numpy()
        subjects = aligned.loc[include].reset_index(drop=True)
        scores = pd.to_numeric(subjects[score_col], errors="coerce").to_numpy(dtype=float)
        rank_edges = rank_normalize_edges(edge_matrix[include])
        run_dir = MRTRIX_OUT / str(model["dir"]) / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = run_dir / "tfnbs_"
        permutation_metadata_path = run_dir / "permutation_seed.json"
        permutation_seed: int | None = None
        permutation_checksum = ""
        if not REUSE or not Path(str(output_prefix) + "fwe_1mpvalue.csv").exists():
            for stale_output in run_dir.glob("tfnbs_*"):
                if stale_output.is_file():
                    stale_output.unlink()
            for obsolete_name in ["permutations.txt", "permutations.json"]:
                obsolete_path = run_dir / obsolete_name
                if obsolete_path.exists():
                    obsolete_path.unlink()
            input_path = write_connectome_inputs(run_dir, subjects["subject_id"].to_numpy(dtype=str), rank_edges, edge_i, edge_j)
            design_path, design_columns = write_design(run_dir, scores, subjects, covariates)
            contrast_path = write_contrast(run_dir, len(design_columns))
            subjects[["subject_id", score_col, *covariates]].to_csv(run_dir / "subjects.csv", index=False)
            permutation_metadata_path, permutation_seed, permutation_checksum = write_permutation_seed(
                run_dir,
                subjects["subject_id"].to_numpy(dtype=str),
                f"{model['dir']}/{slug}",
            )
            if PREPARE_ONLY:
                continue
            log = run_connectomestats(input_path, design_path, contrast_path, permutation_seed, output_prefix)
            (run_dir / "connectomestats.log").write_text(log, encoding="utf-8")
        else:
            input_path = run_dir / "input_connectomes.txt"
            design_path = run_dir / "design.txt"
            contrast_path = run_dir / "contrast.txt"
            design_columns = [line.strip() for line in (run_dir / "design_columns.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            if permutation_metadata_path.exists():
                permutation_metadata = json.loads(permutation_metadata_path.read_text(encoding="utf-8"))
                permutation_seed = int(permutation_metadata["derived_seed"])
                permutation_checksum = str(permutation_metadata["seed_specification_sha256"])
        design = pd.read_csv(design_path, sep=r"\s+", header=None)
        contrast = pd.read_csv(contrast_path, sep=r"\s+", header=None)
        if design.shape[0] != len(subjects) or contrast.shape[1] != design.shape[1]:
            raise RuntimeError(f"TFNBS design mismatch for {model['dir']}/{slug}")
        design_rows.append(
            {
                "model_family": model["family"],
                "model_dir": model["dir"],
                "battery": battery,
                "subtest_label": label,
                "subtest_slug": slug,
                "n_subjects": len(subjects),
                "n_design_columns": design.shape[1],
                "design_columns": ";".join(design_columns),
                "contrast": " ".join(str(x) for x in contrast.iloc[0].tolist()),
                "input_connectomes": str(input_path.relative_to(ROOT)),
                "design_file": str(design_path.relative_to(ROOT)),
                "contrast_file": str(contrast_path.relative_to(ROOT)),
                "subjects_file": str((run_dir / "subjects.csv").relative_to(ROOT)),
                "permutation_seed_metadata_file": (
                    str(permutation_metadata_path.relative_to(ROOT)) if permutation_metadata_path.exists() else ""
                ),
                "permutation_seed": permutation_seed,
                "permutation_seed_specification_sha256": permutation_checksum,
            }
        )
        fwe_1mp = load_edge_output(output_prefix, "fwe_1mpvalue.csv", edge_i, edge_j)
        p_fwe = np.maximum(np.clip(1.0 - fwe_1mp, 0.0, 1.0), P_FLOOR)
        t_values = load_edge_output(output_prefix, "tvalue.csv", edge_i, edge_j)
        enhanced = load_edge_output(output_prefix, "enhanced.csv", edge_i, edge_j)
        p_battery = np.minimum(p_fwe * n_battery, 1.0)
        rho_score = (rank_edges.T @ rank_z(scores)) / max(len(subjects) - 1, 1)
        n_fwe = int(np.sum(p_fwe < 0.05))
        n_battery_bonf = int(np.sum(p_battery < 0.05))
        count_rows.append(
            {
                "battery": battery,
                "subtest_label": label,
                "model_family": model["family"],
                "n_fwe05_edges": n_fwe,
                "n_battery_bonf05_edges": n_battery_bonf,
                "min_p_fwe": float(np.nanmin(p_fwe)),
            }
        )
        keep = (p_fwe < 0.05) | (p_battery < 0.05)
        if keep.sum() < 500:
            keep[np.argsort(p_fwe)[:500]] = True
        for edge_idx in np.flatnonzero(keep):
            roi_i = int(edge_i[edge_idx] + 1)
            roi_j = int(edge_j[edge_idx] + 1)
            group_i = anatomical_groups.get(roi_i, "unknown")
            group_j = anatomical_groups.get(roi_j, "unknown")
            edge_rows.append(
                {
                    "model": model["dir"],
                    "model_family": model["family"],
                    "battery": battery,
                    "subtest_label": label,
                    "roi_i": roi_i,
                    "roi_j": roi_j,
                    "roi_i_name": roi_names.get(roi_i, f"ROI {roi_i}"),
                    "roi_j_name": roi_names.get(roi_j, f"ROI {roi_j}"),
                    "roi_i_anatomical_group": group_i,
                    "roi_j_anatomical_group": group_j,
                    "class_pair": anatomical_pair(group_i, group_j),
                    "deficit_rho": float(-rho_score[edge_idx]),
                    "t_value": float(t_values[edge_idx]),
                    "tfnbs_enhanced": float(enhanced[edge_idx]),
                    "p_fwe_edges": float(p_fwe[edge_idx]),
                    "p_battery_bonferroni": float(p_battery[edge_idx]),
                    "significant_fwe_edges": bool(p_fwe[edge_idx] < 0.05),
                    "significant_battery_bonferroni": bool(p_battery[edge_idx] < 0.05),
                }
            )

if PREPARE_ONLY:
    print("OK: prepared deterministic TFNBS inputs and per-run seeds")
    raise SystemExit(0)

counts = pd.DataFrame(count_rows)
counts.to_csv(OUT / "tfnbs_edge_counts_from_mrtrix.csv", index=False)
pd.DataFrame(design_rows).to_csv(OUT / "tfnbs_design_contrast_inventory.csv", index=False)
edges = pd.DataFrame(edge_rows)
edges.to_csv(OUT / "tfnbs_significant_and_top_edges.csv", index=False)
pairs = (
    edges[edges["significant_fwe_edges"].astype(bool)]
    .groupby(["model", "model_family", "battery", "subtest_label", "class_pair"], as_index=False)
    .agg(
        n_edges=("class_pair", "size"),
        n_battery_bonferroni_edges=("significant_battery_bonferroni", "sum"),
        mean_deficit_rho=("deficit_rho", "mean"),
        min_p_fwe=("p_fwe_edges", "min"),
    )
    .sort_values(["model_family", "battery", "subtest_label", "n_edges"], ascending=[True, True, True, False])
)
pairs.to_csv(OUT / "tfnbs_anatomical_group_pairs.csv", index=False)

write_table_02(counts, pairs, OUT / "table_02_tfnbs_subtest_summary.csv")
figure_path = generate_tfnbs_figure(edges, counts, labels.reset_index())
parameters = {
    "method": "MRtrix3 connectomestats TFNBS",
    "n_permutations": N_SHUFFLES,
    "permutation_base_seed": PERMUTATION_BASE_SEED,
    "permutation_schedule": "MRtrix internal shuffler with a deterministic per-run MRTRIX_RNG_SEED",
    "permutation_nthreads": int(NTHREADS),
    "tfce_dh": TFCE_DH,
    "tfce_e": TFCE_E,
    "tfce_h": TFCE_H,
    "edge_input": "NeMo ChaCoConn upper triangle mirrored by addition, then rank-normalized edgewise",
    "figure_max_display_edges_per_panel": MAX_DISPLAY_EDGES,
}
(OUT / "tfnbs_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
print(f"OK: computed TFNBS summaries and {figure_path.relative_to(ROOT)}")
