# Structural disconnection analysis in glioma patients

Python analysis scripts examining why preoperative language and cognitive performance vary among patients with glioma, including those without test-defined deficits despite extensive estimated disconnection. The analyses relate structural disconnection to performance and patient-reported function.

**Current version: [v2.0.0](https://github.com/LucSam/structural-disconnection-glioma/releases/tag/v2.0.0).** The earlier workflow is preserved as [v1.0.0](https://github.com/LucSam/structural-disconnection-glioma/tree/v1.0.0). See the [changelog](CHANGELOG.md) for the differences.

## Study and project

The scripts accompany the manuscript *Structural disconnection burden and pattern in glioma: preoperative performance and patient-reported function*.

The retrospective cohort comprised 163 patients. The Network Modification Tool (NeMo) used tumour masks and healthy reference tractography to estimate affected connections. Regional ChaCo describes disconnection of each of 191 anatomical parcels; ChaCoConn describes disconnection between parcel pairs. These estimates describe potential lesion effects in reference brains, rather than measuring each patient's own connectivity or adaptation.

Language was assessed with the Aachen Aphasia Test (AAT), cognition with DemTect, and patient-reported function with the EORTC QLQ-C30 and QLQ-BN20. Anatomical descriptions use the structural fs191 parcellation, grouped by anatomy and hemisphere.

**ICONS-GP** stands for *Individual-level COnnectomics for Neuro-oncological Stratification in Glioma Patients*. The [official project page](https://www.gesundheitsforschung-bmftr.de/de/icons-gp-stratifizierung-des-neuroonkologischen-und-neurokognitiven-risikos-bei-gliom-19047.php) describes the broader project.

## Analysis in brief

Two terms organise the analysis:

- **Disconnection burden:** mean regional ChaCo, describing how much connectivity is affected. AAT uses 82 left cerebral and ten left cerebellar parcels; DemTect uses all 191 parcels.
- **Disconnection pattern:** how ChaCo varies across regions, describing where disconnection occurs. Both batteries use all 191 regional values, summarised by eight principal components in the central model.

| Step | Question | Approach |
| --- | --- | --- |
| Cohort and coverage | Where are lesions and estimated disconnections, and how do patients perform? | Lesion overlap, regional and connection means, score distributions. |
| With and without deficits | How do patients differ at high disconnection burden? | Descriptive regional and connection comparisons within the highest burden third; groups remain unmatched and unadjusted. |
| Burden and pattern | Is burden associated with performance beyond clinical factors, and does the pattern add information beyond burden? | All four AAT or five DemTect subtests considered together, separately for each battery; anatomical-group sensitivity analyses. |
| Anatomical localisation | Which regions and connections are associated with lower performance? | Regional partial correlations and threshold-free network-based statistics (TFNBS), adjusted for clinical factors and hemisphere; additional lobe adjustment as sensitivity. |
| Patient-reported function | How do performance and disconnection relate to everyday functioning? | Neuropsychological–QoL correlations and descriptive structural–QoL summaries. |

Clinical factors are age, WHO grade and tumour volume. The central models also include hemisphere. Profile checks assess whether pattern associations differ between subtests. The analyses were developed iteratively and remain exploratory; preserved test performance does not establish network resilience. [Methods overview](docs/METHODS.md) summarises adjustment and correction families.

## Code and reproduction

This is a **code release**. It includes analysis and figure scripts, table/document exporters, synthetic statistical tests, dependencies and a source checksum manifest. The scripts and tests match the reviewed working package byte for byte. Patient data, lesion masks, NeMo outputs, generated results, private provenance and manuscript text/documents are not distributed.

A fresh clone supports code inspection and the synthetic tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Reproducing the study requires the separately held inputs. The full runner also builds the manuscript and therefore requires the private manuscript templates. It cannot reproduce study results from this public repository alone. See [Running the code](docs/RUNNING.md) for inputs, commands and the script map.

The reference environment is Python 3.12.4 and MRtrix3 `connectomestats 3.0.4-153-g4040c17b`. [Software versions](metadata/software_versions.json) and [source checksums](metadata/code_snapshot.json) identify this release. Figures are exported as PNG.

## Versions and licence

Version 2 replaces the earlier command sequence and input/output layout. Prediction and graph-topology screens belong to the archived version 1 workflow; they are not stages of the focused version 2 analysis. Version numbers identify code snapshots, not independent validation of a manuscript.

The analysis code and accompanying documentation are available under the [MIT License](LICENSE).

Copyright (c) 2026 Lucius S. Fekonja.

This licence does not apply to the manuscript or study data. Third-party dependencies retain their own licences.
