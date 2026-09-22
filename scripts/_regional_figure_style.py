#!/usr/bin/env python3
"""Render the established manuscript regional-ChaCo figure compositions.

The layouts, anatomical palette, marker views, and heatmap styling reproduce
the curated journal variants. Statistical values are supplied by the canonical
manuscript-data pipeline rather than read from the retired analysis outputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.patches import Patch
from nilearn import plotting  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont
from scipy import stats  # type: ignore[import-untyped]
from _shared import REGIONAL_OUTCOME_LABELS


ANATOMICAL_ORDER = [
    "Subcortical/limbic",
    "Cerebellum",
    "Frontal/insula",
    "Perisylvian/central sulci",
    "Temporal",
    "Parietal",
    "Occipital/visual",
    "Cingulate/medial",
]
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
ANATOMICAL_CODE = {name: index for index, name in enumerate(ANATOMICAL_ORDER)}
ANATOMICAL_CMAP = mcolors.ListedColormap([ANATOMICAL_COLORS[name] for name in ANATOMICAL_ORDER])

FIGURE2_SUBTEST_SPECS = [
    ("AAT", "Token Test", "Token\nTest"),
    ("AAT", "Repetition", "Repetition"),
    ("AAT", "Naming", "Naming"),
    ("AAT", "Comprehension", "Comprehension"),
    ("DemTect", "Verbal Fluency", "Verbal\nFluency"),
    ("DemTect", "Word List", "Word\nList"),
    ("DemTect", "Digit Span Backwards", "Digit\nSpan"),
    ("DemTect", "Delayed Recall", "Delayed\nRecall"),
    ("DemTect", "Number Conversion", "Number\nConversion"),
]

HEATMAP_OUTCOME_ORDER = [
    ("Token Test", "Token Test"),
    ("Naming", "Naming"),
    ("Comprehension", "Comprehension"),
    ("Repetition", "Repetition"),
    ("Auditory Comprehension", "Auditory Comprehension"),
    ("Auditory Comprehension: sentences", "Auditory Comprehension: sentences"),
    ("Reading Comprehension", "Reading Comprehension"),
    ("Reading Comprehension: sentences", "Reading Comprehension: sentences"),
    ("Repetition: compound words", "Repetition: compound words"),
    ("Repetition: sentences", "Repetition: sentences"),
    ("Word List", "Word List immediate recall"),
    ("Delayed Recall", "Delayed Recall"),
    ("Number Conversion", "Number Conversion"),
    ("Verbal Fluency", "Verbal Fluency"),
    ("Digit Span Backwards", "Digit Span Backwards"),
]

def _bh_fdr(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    out[valid] = restored
    return out


def _compute_centroids(atlas_path: Path) -> pd.DataFrame:
    atlas_img = cast(nib.Nifti1Image, nib.load(atlas_path))
    atlas = np.asarray(atlas_img.dataobj, dtype=np.int16)
    rows = []
    for roi_index in sorted(int(value) for value in np.unique(atlas) if value > 0):
        voxels = np.array(np.where(atlas == roi_index))
        mni = nib.affines.apply_affine(atlas_img.affine, voxels.mean(axis=1))
        rows.append({"roi_index": roi_index, "x": mni[0], "y": mni[1], "z": mni[2]})
    return pd.DataFrame(rows)


def _ordered_meta(labels: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    meta = labels.merge(centroids, on="roi_index", how="inner").copy()
    meta["roi_col"] = "chaco_roi_" + meta["roi_index"].astype(int).astype(str)
    meta["anatomical_code"] = meta["anatomical_group"].map(ANATOMICAL_CODE)
    missing = meta.loc[meta["anatomical_code"].isna(), "anatomical_group"].drop_duplicates().tolist()
    if missing:
        raise ValueError(f"Unrecognized anatomical groups: {missing}")
    return meta.sort_values(["anatomical_code", "roi_index"]).reset_index(drop=True)


def _group_specs() -> list[dict[str, object]]:
    return [
        {
            "label": "Mean AAT T-score at/above 63.5",
            "grouping": "Mean AAT T-score",
            "column": "AAT_mean_T_below_threshold",
            "value": 0,
            "group": "At/above 63.5",
        },
        {
            "label": "Mean AAT T-score below 63.5",
            "grouping": "Mean AAT T-score",
            "column": "AAT_mean_T_below_threshold",
            "value": 1,
            "group": "Below 63.5",
        },
        {
            "label": "DemTect not impaired",
            "grouping": "DemTect binary",
            "column": "DemTect_impaired_binary",
            "value": 0,
            "group": "Unimpaired",
        },
        {
            "label": "DemTect impaired",
            "grouping": "DemTect binary",
            "column": "DemTect_impaired_binary",
            "value": 1,
            "group": "Impaired",
        },
    ]


def _group_chaco_summary(data: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in _group_specs():
        subset = data[data[str(spec["column"])].eq(spec["value"])]
        p_values = []
        start = len(rows)
        for roi in meta.itertuples(index=False):
            values = pd.to_numeric(subset[roi.roi_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) < 3 or np.allclose(values, 0):
                p_value = np.nan
            else:
                try:
                    p_value = float(stats.wilcoxon(values, alternative="greater", zero_method="zsplit").pvalue)
                except ValueError:
                    p_value = np.nan
            p_values.append(p_value)
            rows.append(
                {
                    "grouping": spec["grouping"],
                    "group": spec["group"],
                    "n": len(values),
                    "roi_index": int(roi.roi_index),
                    "roi_name": roi.roi_name,
                    "anatomical_group": roi.anatomical_group,
                    "mean_chaco": float(np.mean(values)) if len(values) else np.nan,
                    "median_chaco": float(np.median(values)) if len(values) else np.nan,
                    "wilcoxon_greater_zero_p": p_value,
                }
            )
        for offset, q_value in enumerate(_bh_fdr(p_values), start=start):
            rows[offset]["wilcoxon_greater_zero_q"] = q_value
            rows[offset]["significant_above_zero_fdr"] = bool(np.isfinite(q_value) and q_value < 0.05)
    return pd.DataFrame(rows)


def _node_sizes(
    values: np.ndarray,
    *,
    min_size: float,
    max_size: float,
    vmax: float,
) -> np.ndarray:
    absolute = np.abs(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0))
    scaled = np.clip(absolute / max(float(vmax), 1e-9), 0, 1)
    return min_size + (max_size - min_size) * scaled


def _bold_axis_text(ax: plt.Axes) -> None:
    for item in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        item.set_fontweight("bold")


def _draw_group_distribution(
    ax: plt.Axes,
    values_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    grouping: str,
    group: str,
) -> None:
    roi_cols = meta["roi_col"].tolist()
    distributions = [pd.to_numeric(values_df[col], errors="coerce").dropna().to_numpy() for col in roi_cols]
    positions = np.arange(1, len(roi_cols) + 1)
    boxes = ax.boxplot(
        distributions,
        positions=positions,
        patch_artist=True,
        showfliers=True,
        widths=0.55,
        flierprops={
            "marker": "o",
            "markersize": 0.9,
            "markerfacecolor": "#555555",
            "markeredgewidth": 0,
            "alpha": 0.22,
        },
    )
    significant = (
        group_summary[
            group_summary["grouping"].eq(grouping) & group_summary["group"].eq(group)
        ]
        .set_index("roi_index")["significant_above_zero_fdr"]
        .to_dict()
    )
    for patch, roi in zip(boxes["boxes"], meta.itertuples(index=False), strict=True):
        if bool(significant.get(int(roi.roi_index), False)):
            patch.set_facecolor(ANATOMICAL_COLORS.get(roi.anatomical_group, "#7f7f7f"))
            patch.set_alpha(0.72)
        else:
            patch.set_facecolor("#202020")
            patch.set_alpha(0.86)
        patch.set_edgecolor("#2d2d2d")
        patch.set_linewidth(0.22)
    for element in ["whiskers", "caps"]:
        for artist in boxes[element]:
            artist.set_color("#666666")
            artist.set_linewidth(0.25)
    for median in boxes["medians"]:
        median.set_color("white")
        median.set_linewidth(0.45)

    bounds = []
    for anatomical_group, group_meta in meta.groupby("anatomical_group", sort=False):
        bounds.append((anatomical_group, int(group_meta.index.min()) + 1, int(group_meta.index.max()) + 1))
    for _, _, last in bounds[:-1]:
        ax.axvline(last + 0.5, color="#d8d8d8", linewidth=0.45, zorder=0)
    ax.axhline(0, color="#222222", linewidth=0.55)
    ax.set_xlim(0.5, len(roi_cols) + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 1.0])
    ax.tick_params(axis="y", labelsize=12.0, length=2, pad=1)
    ax.set_xticks([])
    _bold_axis_text(ax)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def _subtest_profile_table(partial: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    meta_cols = meta[["roi_index", "anatomical_group"]]
    rows = []
    pooled = []
    available = set(partial["outcome_label"].astype(str))
    missing = [outcome_label for _, outcome_label, _ in FIGURE2_SUBTEST_SPECS if outcome_label not in available]
    if missing:
        raise ValueError(f"Missing canonical Figure 02 subtest profiles: {missing}")
    for domain, outcome_label, display_label in FIGURE2_SUBTEST_SPECS:
        selected = partial[partial["outcome_label"].eq(outcome_label)][
            ["roi_index", "deficit_rho_partial"]
        ]
        selected = meta_cols.merge(selected, on="roi_index", how="left")
        positive = np.clip(
            pd.to_numeric(selected["deficit_rho_partial"], errors="coerce").fillna(0).to_numpy(dtype=float),
            0,
            None,
        )
        pooled.append(positive)
        for roi, value in zip(selected.itertuples(index=False), positive, strict=True):
            rows.append(
                {
                    "domain": domain,
                    "subtest_short": outcome_label,
                    "display_label": display_label.replace("\n", " "),
                    "roi_index": int(roi.roi_index),
                    "anatomical_group": roi.anatomical_group,
                    "positive_deficit_rho": float(value),
                }
            )
    if not rows:
        raise ValueError("No canonical partial-correlation subtest profiles were available")
    nonzero = np.concatenate(pooled)
    nonzero = nonzero[nonzero > 0]
    vmax = float(np.percentile(nonzero, 98)) if nonzero.size else 1.0
    return pd.DataFrame(rows), vmax


def _draw_subtest_distribution(
    ax: plt.Axes,
    profile: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    domain: str,
    subtest: str,
    ymax: float,
) -> None:
    selected = (
        profile[profile["domain"].eq(domain) & profile["subtest_short"].eq(subtest)]
        .set_index("roi_index")
        .reindex(meta["roi_index"].astype(int))
    )
    values = pd.to_numeric(selected["positive_deficit_rho"], errors="coerce").fillna(0).to_numpy(dtype=float)
    positions = np.arange(1, len(values) + 1)
    colors = [ANATOMICAL_COLORS.get(group, "#7f7f7f") for group in meta["anatomical_group"]]
    ax.vlines(positions, 0, values, color=colors, linewidth=0.9, alpha=0.84)
    ax.scatter(positions, values, s=3.5, color=colors, alpha=0.86, linewidths=0)
    bounds = []
    for anatomical_group, group_meta in meta.groupby("anatomical_group", sort=False):
        bounds.append((anatomical_group, int(group_meta.index.min()) + 1, int(group_meta.index.max()) + 1))
    for _, _, last in bounds[:-1]:
        ax.axvline(last + 0.5, color="#d8d8d8", linewidth=0.45, zorder=0)
    ax.set_xlim(0.5, len(values) + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, round(ymax, 2)])
    ax.tick_params(axis="y", labelsize=12.0, length=2, pad=1)
    ax.set_xticks([])
    _bold_axis_text(ax)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def generate_figure_02(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    partial: pd.DataFrame,
    atlas_path: Path,
    output_path: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Generate the established four-group plus nine-subtest composite."""
    centroids = _compute_centroids(atlas_path)
    meta = _ordered_meta(labels, centroids)
    group_summary = _group_chaco_summary(data, meta)
    profile, subtest_vmax = _subtest_profile_table(partial, meta)
    coordinates = meta[["x", "y", "z"]].to_numpy()
    color_codes = meta["anatomical_code"].to_numpy()
    specs = _group_specs()

    subsets = [data[data[str(spec["column"])].eq(spec["value"])].copy() for spec in specs]
    group_means = [
        subset[meta["roi_col"].tolist()].mean(axis=0).to_numpy(dtype=float) for subset in subsets
    ]
    pooled_means = np.concatenate([np.nan_to_num(values, nan=0.0) for values in group_means])
    positive_means = pooled_means[pooled_means > 0]
    group_vmax = float(np.percentile(positive_means, 98)) if positive_means.size else 1.0

    fig = plt.figure(figsize=(22, 24.8))
    group_left, group_right = 0.030, 0.970
    group_gap = 0.018
    group_width = (group_right - group_left - group_gap * (len(specs) - 1)) / len(specs)
    group_map_y, group_map_height = 0.812, 0.120
    group_distribution_y, group_distribution_height = 0.700, 0.096
    for index, (spec, subset, means) in enumerate(zip(specs, subsets, group_means, strict=True)):
        left = group_left + index * (group_width + group_gap)
        plotting.plot_markers(
            color_codes,
            coordinates,
            node_size=_node_sizes(means, min_size=5, max_size=118, vmax=group_vmax),
            node_cmap=ANATOMICAL_CMAP,
            node_vmin=-0.5,
            node_vmax=len(ANATOMICAL_ORDER) - 0.5,
            alpha=0.78,
            display_mode="lzr",
            colorbar=False,
            figure=fig,
            axes=(left, group_map_y, group_width, group_map_height),
            title=None,
            node_kwargs={"linewidths": 0.12, "edgecolors": "#303030"},
        )
        fig.text(
            left + group_width / 2,
            group_map_y + group_map_height + 0.010,
            f"{spec['label']} (n={len(subset)})",
            ha="center",
            fontsize=17.0,
            fontweight="bold",
        )
        axis = fig.add_axes((left + 0.010, group_distribution_y, group_width - 0.020, group_distribution_height))
        _draw_group_distribution(
            axis,
            subset,
            group_summary,
            meta,
            grouping=str(spec["grouping"]),
            group=str(spec["group"]),
        )
        axis.set_title("Regional ChaCo distribution", fontsize=16.0, fontweight="bold", pad=8)
        if index == 0:
            axis.set_ylabel("ChaCo", fontsize=14.0, fontweight="bold")

    fig.text(0.5, 0.985, "Global-score groups", ha="center", fontsize=20.0, fontweight="bold", color="#222222")
    fig.text(
        0.5,
        0.635,
        "Subtest regional deficit maps and anatomical-order distributions",
        ha="center",
        fontsize=20.0,
        fontweight="bold",
        color="#222222",
    )

    available_specs = [
        spec
        for spec in FIGURE2_SUBTEST_SPECS
        if not profile[
            profile["domain"].eq(spec[0]) & profile["subtest_short"].eq(spec[1])
        ].empty
    ]
    sub_left, sub_right = 0.040, 0.960
    sub_gap_x = 0.025
    sub_width = (sub_right - sub_left - 2 * sub_gap_x) / 3
    sub_map_height = 0.108
    sub_distribution_height = 0.048
    sub_row_gap = 0.050
    sub_top = 0.598
    for index, (domain, subtest, display_label) in enumerate(available_specs):
        row = index // 3
        column = index % 3
        left = sub_left + column * (sub_width + sub_gap_x)
        cell_top = sub_top - row * (sub_map_height + sub_distribution_height + sub_row_gap)
        map_y = cell_top - sub_map_height
        distribution_y = map_y - sub_distribution_height - 0.006
        values = (
            profile[profile["domain"].eq(domain) & profile["subtest_short"].eq(subtest)]
            .set_index("roi_index")
            .reindex(meta["roi_index"].astype(int))["positive_deficit_rho"]
            .to_numpy(dtype=float)
        )
        plotting.plot_markers(
            color_codes,
            coordinates,
            node_size=_node_sizes(values, min_size=1.5, max_size=82, vmax=subtest_vmax),
            node_cmap=ANATOMICAL_CMAP,
            node_vmin=-0.5,
            node_vmax=len(ANATOMICAL_ORDER) - 0.5,
            alpha=0.75,
            display_mode="lzr",
            colorbar=False,
            figure=fig,
            axes=(left, map_y, sub_width, sub_map_height),
            title=None,
            node_kwargs={"linewidths": 0.08, "edgecolors": "#303030"},
        )
        prefix = "AAT" if domain == "AAT" else "DemTect"
        fig.text(
            left + sub_width / 2,
            map_y + sub_map_height + 0.006,
            f"{prefix}: {display_label.replace(chr(10), ' ')}",
            ha="center",
            fontsize=16.0,
            fontweight="bold",
        )
        axis = fig.add_axes((left + 0.012, distribution_y, sub_width - 0.024, sub_distribution_height))
        _draw_subtest_distribution(
            axis,
            profile,
            meta,
            domain=domain,
            subtest=subtest,
            ymax=max(subtest_vmax, 0.05),
        )
        if column == 0:
            axis.set_ylabel("Deficit\nrho", fontsize=13.2, fontweight="bold")

    handles = [Patch(facecolor=ANATOMICAL_COLORS[name], edgecolor="none", label=name) for name in ANATOMICAL_ORDER]
    handles.append(Patch(facecolor="#202020", edgecolor="none", label="Background fs191 parcels"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        prop={"size": 13.6, "weight": "bold"},
        bbox_to_anchor=(0.5, -0.030),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path, centroids, group_summary


def _plot_glass_brain_component(image: nib.Nifti1Image, output_path: Path, title: str) -> None:
    values = np.asarray(image.dataobj)
    nonzero = values[values != 0]
    vmax = float(np.nanmax(np.abs(nonzero))) if nonzero.size else 0.1
    fig, axis = plt.subplots(figsize=(14, 5), facecolor="white")
    axis.set_facecolor("white")
    plotting.plot_glass_brain(
        image,
        display_mode="lyrz",
        colorbar=True,
        cmap="RdBu_r",
        symmetric_cbar=True,
        vmax=vmax,
        axes=axis,
        title=None,
        plot_abs=False,
        black_bg=False,
    )
    for figure_axis in fig.axes:
        figure_axis.set_facecolor("white")
        if figure_axis is not axis:
            figure_axis.tick_params(labelsize=15.5, width=1.7)
            figure_axis.set_ylabel("Deficit association rho", fontsize=15.5, fontweight="bold")
            for tick_label in [*figure_axis.get_xticklabels(), *figure_axis.get_yticklabels()]:
                tick_label.set_fontweight("bold")
            figure_axis.yaxis.label.set_fontweight("bold")
    fig.suptitle(title, fontsize=16.5, fontweight="bold", y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pretty_roi(name: str, max_len: int = 80) -> str:
    text = str(name).replace("ctx-lh-", "").replace("ctx-rh-", "")
    text = text.replace("_and_", " & ").replace("_", " ")
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "..."


# Full anatomical names for the fixed display selection. Raw atlas identifiers
# remain in the statistical tables and the exported selection manifest.
# Nomenclature: Destrieux et al. (2010), Table 1; FreeSurfer DestrieuxAtlasChanges.
PARCEL_DISPLAY_NAMES = {
    "G_temp_sup-Plan_tempo": "Planum temporale",
    "G_temporal_inf": "Inferior temporal gyrus",
    "Lat_Fis-post": "Posterior segment of the lateral fissure",
    "S_temporal_inf": "Inferior temporal sulcus",
    "G_temporal_middle": "Middle temporal gyrus",
    "G_pariet_inf-Angular": "Angular gyrus",
    "S_interm_prim-Jensen": "Intermediate sulcus of Jensen",
    "S_temporal_transverse": "Transverse temporal sulcus",
    "S_intrapariet_and_P_trans": "Intraparietal and transverse parietal sulci",
    "G_temp_sup-Lateral": "Superior temporal gyrus, lateral aspect",
    "G_front_inf-Orbital": "Inferior frontal gyrus, orbital part",
    "Lat_Fis-ant-Horizont": "Lateral fissure, anterior horizontal ramus",
    "G_pariet_inf-Supramar": "Supramarginal gyrus",
    "S_occipital_ant": "Anterior occipital sulcus",
    "S_oc-temp_lat": "Lateral occipito-temporal sulcus",
    "S_temporal_sup": "Superior temporal sulcus",
    "G_temp_sup-G_T_transv": "Anterior transverse temporal gyrus (Heschl)",
    "G_orbital": "Orbital gyri",
    "G_and_S_subcentral": "Subcentral gyrus and sulci",
    "G_and_S_frontomargin": "Frontomarginal gyrus and sulcus",
    "S_collat_transv_ant": "Anterior transverse collateral sulcus",
    "G_occipital_middle": "Middle occipital gyrus",
    "S_orbital_lateral": "Lateral orbital sulcus",
    "G_oc-temp_lat-fusifor": "Fusiform gyrus",
}


def anatomical_parcel_label(roi_name: str) -> str:
    name, hemisphere = str(roi_name).rsplit(" (", 1)
    if name not in PARCEL_DISPLAY_NAMES:
        raise ValueError(f"No anatomical display name for {roi_name}")
    side = {"L)": "Left", "R)": "Right"}[hemisphere]
    return f"{side} {PARCEL_DISPLAY_NAMES[name][0].lower()}{PARCEL_DISPLAY_NAMES[name][1:]}"


def _heatmap_component(partial: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    partial = partial.copy()
    partial["outcome_label"] = partial["outcome"].map(REGIONAL_OUTCOME_LABELS)
    label_map = dict(HEATMAP_OUTCOME_ORDER)
    missing = sorted(set(label_map) - set(partial["outcome_label"].astype(str)))
    if missing:
        raise ValueError(f"Missing canonical Figure 03 heatmap outcomes: {missing}")
    subtests = partial[partial["outcome_label"].isin(label_map)].copy()
    subtests["display_label"] = subtests["outcome_label"].map(label_map)
    parcel_stats = (
        subtests.assign(abs_deficit=lambda frame: frame["deficit_rho_partial"].abs())
        .groupby(["roi_index", "roi_name", "yeo7_name"], as_index=False)
        .agg(
            n_fdr=("significant_partial_fdr", "sum"),
            mean_abs_deficit=("abs_deficit", "mean"),
            max_abs_deficit=("abs_deficit", "max"),
            sd_deficit=("deficit_rho_partial", "std"),
            range_deficit=("deficit_rho_partial", lambda values: float(np.nanmax(values) - np.nanmin(values))),
        )
    )
    parcel_stats["specificity_score"] = parcel_stats["mean_abs_deficit"] * parcel_stats["sd_deficit"]
    strongest = parcel_stats.sort_values(
        ["n_fdr", "mean_abs_deficit", "max_abs_deficit"], ascending=[False, False, False]
    ).head(12).assign(selection_reason="largest cross-subtest adjusted association")
    selective = parcel_stats.sort_values(
        ["specificity_score", "n_fdr", "max_abs_deficit"], ascending=[False, False, False]
    ).assign(selection_reason="largest subtest-specific adjusted association")
    selected = pd.concat([strongest, selective], ignore_index=True).drop_duplicates("roi_index").head(24)
    if len(selected) != 24:
        raise ValueError(f"Figure 03 requires 24 unique display parcels; observed {len(selected)}")
    roi_order = selected["roi_index"].astype(int).tolist()
    display_order = [display for _, display in HEATMAP_OUTCOME_ORDER]

    heat = (
        subtests.pivot(index="display_label", columns="roi_index", values="deficit_rho_partial")
        .reindex(index=display_order, columns=roi_order)
    )
    stars = (
        subtests.assign(marker=lambda frame: np.where(frame["significant_partial_fdr"], "*", ""))
        .pivot(index="display_label", columns="roi_index", values="marker")
        .reindex(index=display_order, columns=roi_order)
    )
    label_lookup = labels.set_index("roi_index")
    parcel_labels = [anatomical_parcel_label(label_lookup.loc[index, "roi_name"]) for index in roi_order]
    selected = selected.copy()
    selected["display_label"] = parcel_labels
    # Put long anatomical names on horizontal rows; preserve all selected cells,
    # their order, their effects and their within-outcome FDR markers.
    heat = heat.T
    stars = stars.T
    heat.index = parcel_labels
    stars.index = parcel_labels

    fig, axis = plt.subplots(figsize=(15.5, 13.0))
    image = axis.imshow(heat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    axis.set_title("Subtest-disconnection correlations", fontsize=18, fontweight="bold")
    axis.set_xticks(np.arange(len(heat.columns)))
    axis.set_xticklabels(heat.columns, rotation=55, ha="right", fontsize=14)
    axis.set_yticks(np.arange(len(heat.index)))
    axis.set_yticklabels(heat.index, fontsize=14)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.78)
    colorbar.set_label("Deficit association rho", fontsize=15.0, fontweight="bold")
    colorbar.ax.tick_params(labelsize=14.0, width=1.4)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    for row_index, subtest in enumerate(heat.index):
        for column_index, parcel in enumerate(heat.columns):
            if stars.loc[subtest, parcel] == "*":
                axis.text(
                    column_index,
                    row_index,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=21,
                    fontweight="bold",
                    color="#111111",
                    path_effects=[path_effects.withStroke(linewidth=1.4, foreground="white")],
                )
    axis.tick_params(axis="both", length=0)
    fig.subplots_adjust(left=0.42, right=0.96, top=0.94, bottom=0.29)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return selected


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size, index=1 if bold and candidate.endswith(".ttc") else 0)
            except Exception:
                continue
    return ImageFont.load_default()


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _crop_fraction(image: Image.Image, *, top: float, bottom: float) -> Image.Image:
    return image.crop((0, int(round(image.height * top)), image.width, int(round(image.height * bottom))))


def _label_only(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    draw.text((x, y), f"{label})", fill="#111111", font=_font(44, bold=True))


def _centered_text(draw: ImageDraw.ImageDraw, x_center: int, y: int, value: str) -> None:
    selected_font = _font(38, bold=True)
    bounds = draw.textbbox((0, 0), value, font=selected_font)
    draw.text((x_center - (bounds[2] - bounds[0]) // 2, y), value, fill="#111111", font=selected_font)


def generate_figure_03(
    partial: pd.DataFrame,
    labels: pd.DataFrame,
    atlas_path: Path,
    output_path: Path,
) -> tuple[Path, pd.DataFrame]:
    """Generate two global maps and a 24-parcel by 15-subtest heatmap."""
    atlas = cast(nib.Nifti1Image, nib.load(atlas_path))
    atlas_data = np.asarray(atlas.dataobj, dtype=np.int16)

    def image_for(outcome: str) -> nib.Nifti1Image:
        selected = partial[partial["outcome"].eq(outcome)].set_index("roi_index")["deficit_rho_partial"]
        values = np.zeros(atlas_data.shape, dtype=np.float32)
        for roi_index, value in selected.items():
            if np.isfinite(value):
                values[atlas_data == int(roi_index)] = float(value)
        return nib.Nifti1Image(values, atlas.affine, atlas.header)

    with tempfile.TemporaryDirectory(prefix="regional_fig03_") as directory:
        component_dir = Path(directory)
        aat_path = component_dir / "aat.png"
        demtect_path = component_dir / "demtect.png"
        heatmap_path = component_dir / "heatmap.png"
        _plot_glass_brain_component(image_for("AAT_total"), aat_path, "AAT total fs191 deficit rho")
        _plot_glass_brain_component(image_for("DemTect_global"), demtect_path, "DemTect global fs191 deficit rho")
        selected = _heatmap_component(partial, labels, heatmap_path)

        canvas_width = 3600
        margin = 80
        gutter = 60
        header_height = 96
        top_width = (canvas_width - 2 * margin - gutter) // 2
        top_images = [
            _resize_to_width(
                _crop_fraction(Image.open(path).convert("RGB"), top=0.10, bottom=0.98),
                top_width,
            )
            for path in [aat_path, demtect_path]
        ]
        bottom_image = _resize_to_width(Image.open(heatmap_path).convert("RGB"), canvas_width - 2 * margin)
        natural_height = (
            margin
            + header_height
            + max(image.height for image in top_images)
            + 90
            + header_height
            + bottom_image.height
            + margin
        )
        canvas_height = natural_height
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        y = margin
        for index, (label, title) in enumerate(
            [("A", "AAT total fs191 deficit rho"), ("B", "DemTect global fs191 deficit rho")]
        ):
            x = margin + index * (top_width + gutter)
            _label_only(draw, x, y, label)
            _centered_text(draw, x + top_width // 2, y + 8, title)
            canvas.paste(top_images[index], (x, y + header_height))
        y += header_height + max(image.height for image in top_images) + 90
        _label_only(draw, margin, y, "C")
        canvas.paste(bottom_image, (margin, y + header_height))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, optimize=True)
    return output_path, selected
