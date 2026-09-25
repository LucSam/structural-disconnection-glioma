#!/usr/bin/env python3
"""Reproduce the five planned profile contrasts and their pairwise follow-ups."""
import argparse
import itertools
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
import sklearn
from _common import ROOT, SPECS, load_inputs, reproduce_existing, permutation_test, profile_contrasts, holm, sha256
N_PERMUTATIONS = 19999
SEED = 20260923
DEFAULT_OUTPUT = ROOT / 'results/profile'

def analysis_plan(n_permutations: int, seed: int) -> dict:
    return {
        "status": "Exploratory extension; scope fixed before computing these results, not a historical preregistration",
        "null": "The entire structural predictor block has identical coefficients across subtests on the chosen outcome scale",
        "outcome_contrasts": "Orthonormal Helmert contrasts; 3 AAT / 4 DemTect; no second rank transformation",
        "models": [dict(zip(["test_id", "battery", "representation", "scale", "role"], spec)) for spec in SPECS],
        "primary_multiplicity": "Holm family-wise correction across all 5 profile tests, including sensitivities",
        "pairwise_followups": "All 6 AAT and 10 DemTect pair contrasts, rank-normal scale and PC8 only; Holm across all 16; interpreted only if corresponding PC8 omnibus passes Holm-5",
        "covariates": "Age, ordinal WHO grade, log1p tumour volume, lesion hemisphere, and mean regional ChaCo (left for AAT; whole brain for DemTect)",
        "permutation_method": "Freedman-Lane reduced-model outcome-residual permutation; complete patient vectors permuted together",
        "n_permutations": n_permutations, "seed_AAT": seed, "seed_DemTect": seed + 1,
        "reproduction": "All four original joint models, seed 42 sequentially, 5000 permutations; must match saved statistics and p-values",
        "limitations": ["Conditional residual exchangeability assumed; not an exact randomisation experiment", "Rank-normal differences do not equal original T-point differences", "No test of individual reserve, clinical prediction, or regional localisation of between-subtest differences", "In-sample explained variance is descriptive, not cross-validated"],
        "references": ["https://doi.org/10.1016/j.neuroimage.2016.02.053", "https://doi.org/10.1016/j.neuroimage.2014.01.060"],
    }

def write_report(output: Path, profiles: pd.DataFrame, pairs: pd.DataFrame, diagnostics: dict, plan: dict) -> None:
    lines = ["# Subtest profile contrast tests", "", "Exploratory extension of the existing multivariate models. Aggregate outputs only.", "", "## Omnibus results", "", "| Test | n | Pillai | Permutation p | Holm p (5 tests) |", "|---|---:|---:|---:|---:|"]
    for r in profiles.itertuples():
        lines.append(f"| {r.test_id} | {r.n} | {r.pillai:.6f} | {r.p_permutation:.6f} | {r.p_holm_5:.6f} |")
    lines += ["", "## Pairwise follow-ups", "", "All 16 planned contrasts were evaluated and corrected together. Interpretation additionally requires the battery's rank-normal PC8 omnibus to pass Holm-5.", "", "| Battery | Contrast | In-sample partial R² | p | Holm p (16) | Passes both conditions |", "|---|---|---:|---:|---:|---|"]
    for r in pairs.itertuples():
        lines.append(f"| {r.battery} | {r.contrast} | {r.partial_r2_trace_in_sample:.4f} | {r.p_permutation:.6f} | {r.p_holm_16:.6f} | {r.interpretable_after_gate} |")
    lines += ["", "## Interpretation limits", "", "A significant omnibus rejects equal structural-block coefficients across subtests on the tested scale. It does not locate a specific region or show that a global score loses clinically useful information. A nonsignificant result does not establish equal coefficients.", "", "Original AAT T-scores are a planned scale sensitivity; DemTect raw scores are not contrasted because their numerical scales differ.", "", "The Monte Carlo intervals in the CSVs describe uncertainty in permutation tail probabilities, not confidence intervals for effect sizes.", "", f"PCA used {diagnostics['pca_fit_n']} participants and {diagnostics['parcels']} regional measures. The first eight PCs explained {100 * diagnostics['pca_first_8_variance']:.3f}% of regional variance. The original four joint tests reproduced exactly at the recorded permutation p-value resolution.", "", "## Reproduction", "", "Run from the project root:", "", "```sh", "python3 manuscript-data-v2/scripts/04_compute_profile_checks.py", "```", "", "Inputs in manuscript-data-v2/data/private are read only; this script exports aggregate results only. See analysis_plan.json and run_metadata.json for scope, input hashes, versions, and permutation settings.", "", "## Methods references", ""]
    lines.extend(f"- {url}" for url in plan["references"])
    (output / "README.md").write_text("\n".join(lines) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(ROOT.parent / "manuscript-data"):
        raise ValueError("This exploratory script must not write into manuscript-data")
    output.mkdir(parents=True, exist_ok=True)
    plan = analysis_plan(args.permutations, args.seed)
    (output / "analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    blocks, inputs, diagnostics = load_inputs(root)
    old = pd.read_csv(inputs["files"]["existing_joint"])
    reproduced = reproduce_existing(blocks, old)
    reproduced.to_csv(output / "existing_joint_reproduction.csv", index=False)
    print("Reproduced all four existing joint tests", flush=True)
    profile_rows, contrast_rows = [], []
    for test_id, battery, representation, scale, role in SPECS:
        b = blocks[battery]
        y = b["rank"] if scale == "rank_normal" else b["raw"]
        transformed, h = profile_contrasts(y)
        seed = args.seed + int(battery == "DemTect")
        r = permutation_test(transformed, b[representation], b["cov"], args.permutations, np.random.default_rng(seed))
        profile_rows.append({"test_id": test_id, "battery": battery, "representation": representation, "scale": scale, "role": role, "burden_covariate": b["burden"], "seed": seed, **r})
        for j, weights in enumerate(h):
            for label, weight in zip(b["labels"], weights, strict=True):
                contrast_rows.append({"test_id": test_id, "contrast": f"Helmert_{j + 1}", "subtest": label, "weight": float(weight)})
        print(f"Computed {test_id}", flush=True)
    profiles = pd.DataFrame(profile_rows)
    profiles["p_holm_5"] = holm(profiles["p_permutation"])
    profiles["significant_holm_5"] = profiles["p_holm_5"].lt(0.05)
    profiles.to_csv(output / "profile_omnibus_tests.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "contrast_definitions.csv", index=False)
    pair_rows = []
    for battery, b in blocks.items():
        seed = args.seed + int(battery == "DemTect")
        parent = profiles.set_index("test_id").loc[f"{battery}_pc8_rank"]
        for first, second in itertools.combinations(range(len(b["labels"])), 2):
            y = b["rank"][:, [first]] - b["rank"][:, [second]]
            r = permutation_test(y, b["pc8"], b["cov"], args.permutations, np.random.default_rng(seed))
            pair_rows.append({"battery": battery, "contrast": f"{b['labels'][first]} - {b['labels'][second]}", "scale": "rank_normal", "parent_p_holm_5": float(parent["p_holm_5"]), "seed": seed, **r})
        print(f"Computed all {battery} pair contrasts", flush=True)
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm_16"] = holm(pairs["p_permutation"])
    pairs["interpretable_after_gate"] = pairs["p_holm_16"].lt(0.05) & pairs["parent_p_holm_5"].lt(0.05)
    pairs.to_csv(output / "pairwise_profile_tests.csv", index=False)
    hashes_after = {rel: sha256(root / rel) for rel in inputs["hashes"]}
    if hashes_after != inputs["hashes"]:
        raise RuntimeError("Input files changed during the run")
    metadata = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
        "input_sha256": inputs["hashes"], "inputs_unchanged": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "diagnostics": diagnostics, "analysis_plan": plan,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(output, profiles, pairs, diagnostics, plan)
    print(profiles[["test_id", "n", "pillai", "p_permutation", "p_holm_5"]].to_string(index=False), flush=True)
    print("Aggregate output directory:", output, flush=True)

if __name__ == "__main__":
    main()
