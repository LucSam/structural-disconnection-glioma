"""Established cohort displays with regional and connection coverage added."""
from io import BytesIO
from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from PIL import Image
from _common import ROOT
OUT_FIGURES = ROOT / 'outputs/figures'


AAT_SUBTESTS = [
    ("Token Test", "Token Test | T Score (p. 133)", 63.0),
    ("Repetition", "repetition | average T Score across both tests", 63.0),
    ("Naming", "naming | T Score (p. 135)", 64.0),
    ("Comprehension", "comprehension | T Score (p. 136)", 64.0),
]

DEMTECT_SUBTESTS = [
    ("Verbal Fluency", "verbal fluency | out of 20", 20.0),
    ("Word List 1st pass", "word list | first pass - out of 20", 20.0),
    ("Digit Span Backwards", "digit span backwards | out of 6", 6.0),
    ("Delayed Recall", "word list | delayed recall - out of 10", 10.0),
    ("Number Conversion", "number conversion | out of 4", 4.0),
]

SHORT_LABELS = {
    "Word List 1st pass": "Word List",
    "Digit Span Backwards": "Digit Span",
    "Number Conversion": "Number\nConversion",
    "Verbal Fluency": "Verbal Fluency",
    "Delayed Recall": "Delayed Recall",
    "Comprehension": "Comprehension",
    "Naming": "Naming",
    "Token Test": "Token Test",
    "Repetition": "Repetition",
}

def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.18) -> None:
    ax.text(x, y, f"{label})", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

def style_axis_text(ax: plt.Axes) -> None:
    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontweight('normal')
        item.set_fontsize(12)
    for item in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        item.set_fontweight('normal')
        item.set_fontsize(11.5)

def lesion_overlap_slice_rgb() -> tuple[np.ndarray, int]:
    """Render the original eight-cut anatomical lesion-overlap panel."""
    overlap_image = nib.load(ROOT / 'data/resources/cohort_lesion_overlap.nii.gz')
    maximum = int(np.asarray(overlap_image.dataobj).max())
    template = datasets.load_mni152_template()

    source_figure, source_axis = plt.subplots(figsize=(11.8, 3.4))
    plotting.plot_stat_map(
        overlap_image,
        bg_img=template,
        display_mode="z",
        cut_coords=[-28, -18, -8, 2, 12, 22, 32, 42],
        threshold=1,
        colorbar=False,
        annotate=False,
        black_bg=False,
        dim=0.35,
        cmap="inferno",
        vmax=maximum,
        axes=source_axis,
        figure=source_figure,
        title=None,
    )
    source_colorbar_axis = source_figure.add_axes((0.92, 0.18, 0.015, 0.62))
    source_colorbar = source_figure.colorbar(
        ScalarMappable(norm=mcolors.Normalize(vmin=1, vmax=maximum), cmap="inferno"),
        cax=source_colorbar_axis,
    )
    source_colorbar.set_label("Overlap count", fontsize=13.0, fontweight="black", labelpad=6)
    source_colorbar.ax.yaxis.label.set_fontweight("black")
    source_colorbar.ax.tick_params(labelsize=12.0, width=1.6)
    for label in source_colorbar.ax.get_yticklabels():
        label.set_fontweight("black")
    source_figure.subplots_adjust(left=0.01, right=0.90, top=0.98, bottom=0.02)

    buffer = BytesIO()
    source_figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(source_figure)
    buffer.seek(0)
    with Image.open(buffer) as source_image:
        source_image = source_image.convert("RGB")
        left = round(source_image.width * 0.10)
        top = round(source_image.height * 0.05)
        right = round(source_image.width * 0.915)
        bottom = round(source_image.height * 0.95)
        rendered = np.asarray(source_image.crop((left, top, right, bottom))).copy()
    # Remove the source renderer's white margins before placing the eight slices.
    ink = (rendered < 242).any(axis=2)
    yy, xx = np.where(ink)
    rendered = rendered[max(0, yy.min()-4):yy.max()+5, max(0, xx.min()-4):xx.max()+5]
    return rendered, maximum

def histogram_panel(
    ax: plt.Axes,
    values: pd.Series,
    *,
    threshold: float,
    bins: np.ndarray,
    color: str,
    xlabel: str,
    shade_label: str,
    ylabel: str = "Patients",
) -> None:
    x = pd.to_numeric(values, errors="coerce").dropna()
    ax.hist(x, bins=bins.tolist(), color=color, edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axvspan(bins.min(), threshold, color="#B23A48", alpha=0.10, lw=0)
    ax.axvline(threshold, color="#B23A48", linestyle="--", linewidth=1.6, label=shade_label)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="normal")
    ax.tick_params(labelsize=11.5)
    legend = ax.legend(
        frameon=False,
        prop={"size": 11.5, "weight": "normal"},
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0,
        handlelength=1.6,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
    style_axis_text(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)

def horizontal_distribution_panel(
    ax: plt.Axes,
    outcomes: pd.DataFrame,
    specs: list[tuple[str, str, float]],
    *,
    xlabel: str,
    thresholds_are_scores: bool,
) -> None:
    rng = np.random.default_rng(42)
    y_positions = np.arange(len(specs))
    values_for_box = []
    for _, column, scale_or_threshold in specs:
        values = pd.to_numeric(outcomes[column], errors="coerce").dropna().to_numpy(dtype=float)
        if not thresholds_are_scores:
            values = 100.0 * values / float(scale_or_threshold)
        values_for_box.append(values)
    ax.boxplot(
        values_for_box,
        vert=False,
        positions=y_positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.1},
        boxprops={"facecolor": "#F5F5F5", "edgecolor": "#555555", "linewidth": 0.7},
        whiskerprops={"color": "#555555", "linewidth": 0.7},
        capprops={"color": "#555555", "linewidth": 0.7},
    )
    for y, values in zip(y_positions, values_for_box):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(values, y + jitter, s=8, color="#555555", alpha=0.42, linewidths=0, zorder=3)
    if thresholds_are_scores:
        for y, (_, _, threshold) in zip(y_positions, specs):
            ax.vlines(threshold, y - 0.33, y + 0.33, color="#B23A48", linestyle="--", linewidth=1.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SHORT_LABELS.get(label, label) for label, _, _ in specs], fontsize=11.5, fontweight="normal")
    ax.tick_params(axis="y", labelsize=11.5, pad=2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12, fontweight="normal")
    ax.tick_params(axis="x", labelsize=11.5)
    style_axis_text(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45)

def make_figure_01() -> None:
    outcomes = pd.read_csv(ROOT / "data/private/outcomes.csv")
    overlap_image, overlap_maximum = lesion_overlap_slice_rgb()
    aat_threshold = 63.5
    demtect_threshold = 13.0

    fig = plt.figure(figsize=(16.1, 12.4), dpi=300)
    gs = fig.add_gridspec(
        2,
        8,
        height_ratios=[1.18, 1.0],
        width_ratios=[1.0, 0.42, 1.25, 1.25, 1.32, 0.88, 1.42, 1.42],
        left=0.045,
        right=0.99,
        top=0.97,
        bottom=0.08,
        hspace=0.34,
        wspace=0.25,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0:1])
    ax_c = fig.add_subplot(gs[1, 2:4])
    ax_d = fig.add_subplot(gs[1, 4:5])
    ax_e = fig.add_subplot(gs[1, 6:8])
    ax_a.set_position((0.020, 0.795, 0.845, 0.19))
    ax_b.set_position((0.055, 0.075, 0.085, 0.285))
    ax_c.set_position((0.300, 0.075, 0.210, 0.285))
    ax_d.set_position((0.570, 0.075, 0.095, 0.285))
    ax_e.set_position((0.825, 0.075, 0.170, 0.285))

    ax_a.imshow(overlap_image)
    ax_a.axis("off")
    fig.text(.038, .978, 'A)  Tumour distribution in the cohort (n = 163)',
             fontsize=16, fontweight='bold')
    cax = fig.add_axes((0.897, 0.805, 0.012, 0.16), zorder=21)
    overlap_cbar = fig.colorbar(
        ScalarMappable(norm=mcolors.Normalize(vmin=1, vmax=overlap_maximum), cmap="inferno"),
        cax=cax,
        ticks=np.arange(10, overlap_maximum + 1, 10),
    )
    overlap_cbar.set_label("Overlap count", fontsize=12, fontweight="normal", labelpad=6)
    overlap_cbar.ax.tick_params(labelsize=11.5, width=1.)

    histogram_panel(
        ax_b,
        outcomes["AAT_mean_T_score"],
        threshold=aat_threshold,
        bins=np.arange(35, 101, 5),
        color="#8B1E3F",
        xlabel="Mean AAT T-score",
        shade_label="Mean T-score < 63.5",
    )
    panel_label(ax_b, "D")

    horizontal_distribution_panel(
        ax_c,
        outcomes,
        AAT_SUBTESTS,
        xlabel="AAT T-score",
        thresholds_are_scores=True,
    )
    panel_label(ax_c, "E")

    histogram_panel(
        ax_d,
        outcomes["DemTect_global"],
        threshold=demtect_threshold,
        bins=np.arange(0, 20, 1),
        color="#2A6F97",
        xlabel="DemTect score",
        shade_label="DemTect < 13",
        ylabel="",
    )
    panel_label(ax_d, "F")

    horizontal_distribution_panel(
        ax_e,
        outcomes,
        DEMTECT_SUBTESTS,
        xlabel="DemTect subtest score\n(% maximum)",
        thresholds_are_scores=False,
    )
    panel_label(ax_e, "G")

    from _reader_figures import cohort_coverage_panels
    cohort_coverage_panels(fig)

    # Keep the added coverage row from making labels too small at manuscript width.
    fig.canvas.draw()
    for ax in fig.axes:
        style_axis_text(ax)
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize()*1.3)

    fig.savefig(OUT_FIGURES / "figure_01_cohort_coverage.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
