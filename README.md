# Structural disconnection analysis in glioma patients

Python analysis scripts for investigating structural disconnection, preoperative language and cognitive performance, and patient-reported quality of life in glioma.

## Study overview

Patients with glioma differ in their language and cognitive performance. Some retain measured function despite extensive tumour-related disconnection. This retrospective study of 163 patients examines how these differences relate to the amount and anatomical pattern of estimated disconnection, and how they are reflected in patient-reported function.

The Network Modification Tool (NeMo) combines tumour masks with tractography from healthy reference brains to estimate affected connections. Regional ChaCo describes disconnection of each of 191 anatomical parcels; ChaCoConn describes disconnection between parcel pairs. Language was assessed with the Aachen Aphasia Test (AAT), cognition with DemTect, and patient-reported function with the EORTC QLQ-C30 and QLQ-BN20.

The scripts accompany the manuscript *Structural disconnection burden and pattern in glioma: preoperative performance and patient-reported function*.

## Analysis approach

We first describe where disconnection occurs and how patients with and without test-defined deficits differ, including those with high disconnection burden. Here, **burden** means mean regional ChaCo, while **pattern** describes how disconnection varies across regions.

The central analysis asks whether burden is associated with performance beyond clinical factors, and whether the pattern adds information beyond burden. It considers the AAT or DemTect subtests together, separately for each battery. Regional maps and connection-level analyses then locate associations with lower performance. Quality-of-life analyses relate both performance and disconnection to the patient's reported experience.

The analyses are exploratory. NeMo estimates lesion effects in reference brains; it does not directly measure individual adaptation or resilience. The [methods overview](docs/METHODS.md) explains adjustment, sensitivity analyses and correction families.

## What this repository contains

- Analysis scripts and shared numerical methods.
- Figure generators and editable table/document exporters.
- Synthetic statistical tests, dependencies and source checksums.

**Patient data, lesion masks, individual NeMo outputs, generated results and manuscript files are not included.** Reproducing the study requires the separately held inputs; the document builder also requires private manuscript templates.

## Getting started

A fresh clone supports code inspection and tests with synthetic data:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See [Running the code](docs/RUNNING.md) for setup, required inputs, the script map and reproduction commands. The reference environment uses Python 3.12.4 and MRtrix3 `connectomestats 3.0.4-153-g4040c17b`; exact [software versions](metadata/software_versions.json) and [source checksums](metadata/code_snapshot.json) are recorded.

## The ICONS-GP project

**ICONS-GP** stands for *Individual-level COnnectomics for Neuro-oncological Stratification in Glioma Patients*. The [official project page](https://www.gesundheitsforschung-bmftr.de/de/icons-gp-stratifizierung-des-neuroonkologischen-und-neurokognitiven-risikos-bei-gliom-19047.php) describes the broader project and its aims.

## Versions

The current code release is [v2.0.0](https://github.com/LucSam/structural-disconnection-glioma/releases/tag/v2.0.0). The earlier, broader workflow—including prediction and graph-topology analyses—is preserved as [v1.0.0](https://github.com/LucSam/structural-disconnection-glioma/tree/v1.0.0). See the [changelog](CHANGELOG.md) for the changes in scope and workflow.

## Licence

The analysis code and accompanying documentation are available under the [MIT License](LICENSE).

Copyright (c) 2026 Lucius S. Fekonja.

This licence does not apply to the manuscript or study data. Third-party dependencies retain their own licences.
