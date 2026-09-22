# Structural disconnection analysis in glioma patients

Python analysis scripts for investigating structural disconnection, language and cognitive performance, and quality of life in patients with glioma.

## The ICONS-GP project

**ICONS-GP** stands for *Individual-level COnnectomics for Neuro-oncological advanced Stratification in Glioma Patients*. The [official BMFTR project page](https://www.gesundheitsforschung-bmftr.de/de/icons-gp-stratifizierung-des-neuroonkologischen-und-neurokognitiven-risikos-bei-gliom-19047.php) describes the broader project and its aims.

## Study overview

These scripts accompany the manuscript **Structural disconnection patterns relate to preoperative language subtest profiles in glioma**.

This retrospective study examined how the anatomical distribution of estimated structural disconnection relates to language and cognitive performance before glioma surgery. The cohort comprised 163 patients. Lesion masks and normative tractography, processed with the Network Modification (NeMo) toolbox, provided disconnection estimates for 191 brain parcels and their connections. Language and cognition were assessed with the Aachen Aphasia Test (AAT) and DemTect.

The main analyses characterise regional disconnection, connections associated with individual subtests, network organisation and joint subtest profiles. The manuscript reports an association between regional disconnection patterns and AAT profiles beyond mean left-hemisphere disconnection, alongside Token Test and Naming edge associations that survived lesion-location adjustment and correction across subtests. Individual prediction and patient-reported quality of life are secondary, exploratory analyses.

## What this repository contains

- Python analysis scripts and shared helpers.
- Pinned Python dependencies and the reference software versions.
- Table captions, figure provenance and reference output checksums.
- Tests for the interpretation and loading of NeMo connectivity matrices.

**This is a code release. Patient data, lesion masks, individual NeMo outputs and generated results are not included.** A fresh clone supports code inspection and two synthetic matrix tests. Reproducing the study requires the separately held study inputs; the full pipeline will stop if they are absent.

The workflow starts after lesion segmentation, spatial normalisation and NeMo processing. These upstream steps are outside this repository.

## Script overview

The scripts follow the study from cohort preparation to statistical analyses and manuscript outputs. Generated numerical results go to `results/analysis/`; tables and figures go to `outputs/`.

| Script | Purpose |
| --- | --- |
| [00_run_all.py](scripts/00_run_all.py) | Run the complete downstream workflow in dependency order. |
| [01_check_environment.py](scripts/01_check_environment.py) | Check software versions, required files and study input counts. |
| [02_compute_cohort_and_outcomes.py](scripts/02_compute_cohort_and_outcomes.py) | Prepare cohort characteristics, outcome scores and group definitions. |
| [03_compute_regional_chaco_statistics.py](scripts/03_compute_regional_chaco_statistics.py) | Map regional disconnection and its associations with performance; generate Figures 2–3. |
| [04_compute_tfnbs_statistics.py](scripts/04_compute_tfnbs_statistics.py) | Test connection-level associations with individual subtests; generate Table 2 and Figure 4. |
| [05_compute_graph_topology_statistics.py](scripts/05_compute_graph_topology_statistics.py) | Describe estimated remaining network organisation and its associations with performance; generate Figures 5–6. |
| [06_compute_multivariate_statistics.py](scripts/06_compute_multivariate_statistics.py) | Test associations between regional disconnection patterns and joint subtest profiles, accounting for mean disconnection. |
| [07_compute_prediction_performance.py](scripts/07_compute_prediction_performance.py) | Compare clinical, location and regional-disconnection predictors using cross-validation; generate Table 3. |
| [08_compute_qol_associations.py](scripts/08_compute_qol_associations.py) | Examine quality-of-life associations with test performance and structural measures; generate Figure 7. |
| [09_generate_manuscript_outputs.py](scripts/09_generate_manuscript_outputs.py) | Generate the cohort/lesion-overlap Figure 1 and collect the manuscript tables and figures. |
| [10_verify_outputs.py](scripts/10_verify_outputs.py) | Check statistical calculations, study-specific reference results, exports and checksums. |
| [11_export_tables_to_docx.py](scripts/11_export_tables_to_docx.py) | Export Tables 1–3 as editable Word documents. |
| [12_refresh_manuscript_exports.py](scripts/12_refresh_manuscript_exports.py) | Refresh selected tables, figures, labels and the joint QoL correction from saved results without refitting models. |

Shared helpers handle data loading and outcome definitions ([`_shared.py`](scripts/_shared.py)), regional figure layouts ([`_regional_figure_style.py`](scripts/_regional_figure_style.py)) and Table 2 export ([`_table_exports.py`](scripts/_table_exports.py)). [`update_release_checksums.py`](scripts/update_release_checksums.py) records reference checksums after an intentional, reviewed output update.

## Reading the analyses

- **ChaCo:** NeMo's change-in-connectivity estimate, summarised for individual brain regions. **ChaCoConn** describes disconnection between pairs of regions.
- **TFNBS:** threshold-free network-based statistics, used here to identify connections associated with subtest performance while correcting across edges.
- **Graph descriptors:** measures of the organisation of NeMo-predicted remaining connectivity. These matrices represent estimates derived from normative tractography, not measured postoperative connectivity.
- **Multivariate analysis:** tests several subtest scores jointly. This addresses the relationship between disconnection patterns and performance profiles. The separate prediction analysis asks whether regional information improves estimates for held-out patients.

## Getting started

1. Read the script overview above.
2. Follow [Running the code](docs/RUNNING.md) to set up Python and run the synthetic tests.
3. With the study inputs available, follow the same guide to reproduce the full workflow or refresh existing exports.

The reference environment uses **Python 3.12.4** and **MRtrix3 `connectomestats 3.0.4-153-g4040c17b`**. Exact versions are recorded in [`metadata/software_versions.json`](metadata/software_versions.json). The code retains study-specific filenames, sample-count checks and verification targets; applying it to a different cohort requires adapting those assumptions.
