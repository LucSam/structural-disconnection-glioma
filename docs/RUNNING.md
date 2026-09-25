# Running the code

Run commands from the repository root. Version 2 uses relative paths. It starts after lesion segmentation, spatial normalisation and NeMo processing; these upstream steps are not included.

## Set up the environment

The reference Python version is 3.12.4:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic data and require no study inputs. They check profile contrasts, permutation calculations and multivariate statistics against independent calculations. They do not validate the complete study pipeline.

TFNBS additionally requires MRtrix3 `connectomestats` on `PATH`; the reference version is `3.0.4-153-g4040c17b`. This runner does not use the version 1 `CONNECTOMESTATS` or `ICONS_TFNBS_REUSE` environment options.

## Required private inputs

A public clone does not contain the following files. The study scripts retain cohort-specific checks and cannot be applied unchanged to an arbitrary dataset.

| Local directory | Contents used by the scripts |
| --- | --- |
| `data/private/` | Prepared cohort, outcome and regional ChaCo tables; patient-level analysis records; TFNBS and hemisphere-audit snapshots, including original subject order. |
| `data/resources/` | Atlas definitions, matching fs191 parcellation and centroids, and the aggregate lesion-overlap image. |
| `data/reference/` | Saved multivariate, regional, connection and QoL reference results from the earlier analysis. |
| `metadata/` | Private input manifests and provenance required for validation; additional metadata is generated locally. |
| `manuscript/` | Private `manuscript_template.md`, `supplement_template.md` and `reference_library.json`, required by the document builder. |

Table columns and outcome mappings are defined in [`_common.py`](../scripts/_common.py) and [`01_prepare_inputs.py`](../scripts/01_prepare_inputs.py). Join keys must agree across patient-level files. The `subject_id` field in source code is a column name, not a supplied identifier list. Locally generated records, manifests and raw MRtrix logs may contain identifiers or local paths; they are excluded from version control by the allowlist in `.gitignore`.

With an authorised, complete prepared version 2 input package, place those inputs in the directories above. Existing result tables are additionally required for a report-only rebuild. A prepared package must include all reference files and private audit/TFNBS snapshots, not just the three clinical/regional CSV files.

### Preparing again from the original study package

For a refresh, the scripts expect the complete original private package in a sibling directory named `manuscript-data`:

```text
workspace/
  manuscript-data/                  # original private data, scripts and completed results
  structural-disconnection-glioma/   # this repository plus authorised local inputs
```

```bash
python scripts/00_run_all.py --refresh-inputs
```

This imports derived inputs, recreates the lesion-overlap image and refreshes the TFNBS/audit snapshots. Some later stages also copy reference files from that sibling package when missing. The original package must include its completed version 1 results, masks and NeMo outputs. Checking out the public v1.0.0 tag alone does not supply those data or results. No NeMo estimation is rerun here.

## Run or rebuild

```bash
# Full focused workflow from complete prepared private inputs:
python scripts/00_run_all.py

# Rebuild descriptions, figures, tables and document from existing model results:
python scripts/00_run_all.py --report-only
```

The full run recomputes the central models and profile checks, which use 19,999 permutations. Nine clinical-plus-hemisphere TFNBS analyses use 5,000 shuffles each. TFNBS results are reused only when input, parameter, executable and output hashes match. If a previous run no longer matches, the script stops for review rather than overwriting it.

`--report-only` skips central-model, profile and TFNBS permutation runs. It still needs private patient-level inputs for descriptions, audits and figures. It is not a data-free document build. The two runner options cannot be combined.

Results are written to `results/`, PNG figures to `outputs/figures/`, an optional circle alternative to `outputs/alternatives/`, and editable tables to `outputs/tables/`. The document builder writes local Word and Markdown files under `manuscript/`. These generated files are not tracked.

To inspect or run an individual analysis, use the corresponding script below. Document generation and full package validation require the private manuscript sources; analysis scripts do not read manuscript text.

## Script map

| Script | Purpose |
| --- | --- |
| `00_run_all.py` | Execute the workflow in dependency order. |
| `01_prepare_inputs.py` | Import derived study inputs and reference results; create aggregate lesion overlap. |
| `02_compute_staged_models.py` | Compare clinical factors, added burden and added pattern; reproduce earlier joint models. |
| `03_compute_clinical_qol.py` | Describe deficit classifications and burden thirds; reproduce the full neuropsychological–QoL family. |
| `04_compute_profile_checks.py` | Test subtest contrasts and retain both complete Holm correction families. |
| `05_prepare_anatomy.py` | Verify earlier connection counts and prepare anatomical summaries. |
| `05b_prepare_displays.py` | Prepare individual-performance displays and model summaries. |
| `05c_audit_hemisphere_support.py` | Check atlas composition and lesion-side support for retained connections. |
| `05d_compute_hemisphere_tfnbs.py` | Run or verify clinical-plus-hemisphere TFNBS for all nine subtests. |
| `05e_prepare_group_anatomy.py` | Describe high-burden regional groups and prepare display data. |
| `05f_prepare_regional_displays.py` | Calculate hemisphere-adjusted regional coefficients and group differences. |
| `05g_prepare_network_context.py` | Summarise structural coverage, anatomical connection differences and structural–QoL coefficients. |
| `05h_describe_battery_overlap.py` | Describe shared cases, classifications and correlation of the two burden measures. |
| `06_create_figures.py` | Draw the main and supplementary PNG figures. |
| `07_build_manuscript.py` | Export editable tables and the local manuscript using private templates. |
| `08_validate_package.py` | Check input provenance, correction families, reported values and document/figure consistency. |
| `09_create_circle_alternative.py` | Draw the optional circle alternative, retaining all qualifying edges. |

`_common.py` contains the numerical methods and input definitions. The four figure helpers handle anatomical maps, the cohort figure, the NeMo schematic and patient-report displays. The letters on the `05` scripts group supporting anatomical and descriptive stages; they are not separate workflow versions.

## Verification and version 1

The reviewed working package passed 698 data-dependent checks before publication. These require the private study package and generated report. A fresh public clone can run the synthetic tests only. Matching source hashes establishes code identity, not equivalence of results under different inputs or software versions.

[`metadata/code_snapshot.json`](../metadata/code_snapshot.json) records the exact source/test checksums for v2.0.0. [`metadata/software_versions.json`](../metadata/software_versions.json) records the reference environment. Other local metadata is excluded.

The original broader workflow, its documentation and MIT licence remain available at [v1.0.0](https://github.com/LucSam/structural-disconnection-glioma/tree/v1.0.0). Use a separate checkout for that version; its script names, environment options and output layout differ from version 2.
