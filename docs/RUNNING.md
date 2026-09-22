# Running the code

Run all commands from the repository root. The scripts use paths relative to that directory; no personal filesystem path is required.

## 1. Set up Python

Use Python 3.12.4 to match the reference environment. Once that version is available as `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The activation command is for macOS/Linux. The full environment check requires the exact recorded versions. TFNBS also needs the separately installed MRtrix3 executable `connectomestats`, reference version `3.0.4-153-g4040c17b`.

## 2. Run the tests that need no patient data

```bash
python -m pytest -q tests/test_nemo_graph_loading.py -k 'upper_triangle or nontriangular'
```

These two tests verify that an upper-triangular NeMo matrix is mirrored without halving its weights and that an incorrectly encoded matrix is rejected. The third test in this file requires the study's extracted NeMo outputs. Synthetic tests alone do not validate the complete analysis.

## 3. Supply the study inputs

The full pipeline expects the following local layout. These inputs are **not distributed in this repository**.

| Location | Required inputs |
| --- | --- |
| `data/raw/clinical_outcome_metadata/` | `data_mr.csv`, `patients_mr.csv` and `tumor_volumes_ml.csv`: clinical characteristics, neuropsychological/QoL scores and tumour volumes. |
| `data/raw/lesion_metadata/` | `atlas_mask_coverage.csv`: atlas-derived lesion-location summaries. |
| `data/raw/lesion_masks_mni/` | The 163 MNI-space lesion masks used for cohort visualisation. |
| `data/nemo/ifod2act_fs191/` | Per-subject NeMo regional ChaCo, pairwise ChaCoConn and predicted remaining structural-connectivity outputs. |
| `data/nemo/upload_batches/` | The 18 archived input ZIP files counted by the environment check. |
| `data/nemo/downloaded_results/` | The 18 archived result ZIP files counted by the environment check. |
| `data/resources/` | `network_definitions.csv` and `nemo_fs191_parcellation.nii.gz`: parcel labels/group assignments and the matching atlas. |

The expected table columns and outcome mappings are defined in [`_shared.py`](../scripts/_shared.py) and the analysis scripts. Subject keys must agree across the input tables and NeMo directories. The environment check is specific to the original study package, including its archive counts.

Local data and generated outputs remain excluded by `.gitignore`. The cohort-building script retains columns from the supplied clinical tables, so local results may contain identifiers if those tables contain them. Keep those results within the authorised data environment.

## 4. Run the full workflow

If `connectomestats` is not on `PATH`, point to the installed executable:

```bash
export CONNECTOMESTATS=/path/to/connectomestats
```

Then run:

```bash
python scripts/01_check_environment.py
python scripts/00_run_all.py
```

The runner executes scripts **01–09, then 11, then 10**. Word export precedes verification because the verifier also inspects the generated tables. Statistical analyses use the inputs and computed intermediate results; the manuscript is not a pipeline input.

The full workflow includes 18 TFNBS models with 5,000 shuffles each, graph permutation tests, multivariate permutation tests and repeated cross-validation with bootstrap intervals. These computations can take substantial time. Results are written to:

- `results/analysis/`: cohort, regional, TFNBS, graph, multivariate, prediction and QoL tables, including individual-level outputs.
- `outputs/figures/`: Figures 1–7.
- `outputs/tables/`: Tables 1–3 in CSV and Word format.

For a rerun with complete TFNBS results already present:

```bash
ICONS_TFNBS_REUSE=1 python scripts/00_run_all.py
```

This reuses the existing MRtrix outputs; the other analysis stages still run. Exact software versions are checked by default. `ICONS_ALLOW_VERSION_MISMATCH=1` bypasses that version check for deliberate compatibility testing; it does not establish equivalence to the reference run.

## 5. Refresh existing manuscript exports

With the study data and completed analysis results available:

```bash
python scripts/12_refresh_manuscript_exports.py
```

This regenerates Tables 1–3, refreshes regional labels and Figures 3/7, and recalculates the joint 51-test QoL correction from saved test results. It uses the same export functions as the full workflow, without rerunning patient-level models or permutations.

## Verification and reference metadata

[`10_verify_outputs.py`](../scripts/10_verify_outputs.py) checks statistical identities, study-specific result counts, table contents, figure provenance and reference file hashes. It needs the study data and generated outputs. It is a reproducibility check for this cohort, not a general test suite for arbitrary datasets.

The four metadata files serve distinct purposes:

| File | Purpose |
| --- | --- |
| [`software_versions.json`](../metadata/software_versions.json) | Reference Python, package and external-tool versions. |
| [`table_export_metadata.json`](../metadata/table_export_metadata.json) | Table titles and notes used by the Word exporter. |
| [`figure_generation_provenance.csv`](../metadata/figure_generation_provenance.csv) | Source analysis for each figure; refreshed by the output writer. |
| [`output_checksums_sha256.csv`](../metadata/output_checksums_sha256.csv) | Reference hashes for the seven figure PNGs and three table CSVs. |

Numerical or rendering changes can cause verification to fail. Inspect the differences first. Only when intentionally accepting a new reference output set, update the hashes and verify again:

```bash
python scripts/update_release_checksums.py
python scripts/10_verify_outputs.py
```

The full runner never updates the reference hashes automatically. Matching hashes establish file identity; the separate statistical checks assess selected calculations and outputs.
