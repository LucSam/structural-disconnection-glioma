# Changelog

## 2.0.0 — 2026-09-25

Focused analysis workflow accompanying *Structural disconnection burden and pattern in glioma: preoperative performance and patient-reported function*.

### Analysis and reporting

- Organise the study around disconnection coverage, patients with and without deficits at high burden, the central burden/pattern comparison, anatomy and patient-reported function.
- Compare burden and pattern within multivariate AAT and DemTect models; include anatomical-group sensitivity analyses and explicit profile contrasts.
- Add clinical-plus-hemisphere TFNBS as the main connection model, retaining the earlier lobe-adjusted results as sensitivity and preserving their correction families.
- Provide regional and connection group descriptions, lesion-side diagnostics, battery-overlap summaries and the structural–QoL link.
- Update figure and table generators, including uncapped connection displays, PNG-only exports and the technical NeMo schematic.
- Use consistent disconnection-burden/pattern terminology and the current figure labels and table captions.
- Replace the previous broad workflow on the default branch. Prediction and graph-topology analyses remain available in v1.0.0; historical correction families used by the focused analysis are retained.

### Reproduction

- Replace the script sequence and input/output layout with the focused pipeline. This is a breaking workflow change, so it receives a major version rather than v1.1.0.
- Include the reviewed scripts and synthetic tests without publication-specific code changes.
- Update the README, input/run guide, methods overview, dependency list and source checksums.
- Keep private inputs, generated results, manuscripts and private metadata out of the repository. The MIT licence remains in the name of Lucius S. Fekonja and covers repository code and documentation.

## 1.0.0 — 2026-09-25

Retrospective version tag for the existing public repository at commit `905cd7e93823db18a4d358c29af52daa8bd66780`. The date above is the date of versioning, not the date the analyses were performed.

- Preserve the original public scripts, documentation and MIT licence without modifying that snapshot.
- Include cohort, regional, connection, graph-topology, multivariate, prediction and QoL analyses, with their original export and verification workflow.

Versions follow the major/minor/patch convention of [Semantic Versioning](https://semver.org/), treating documented script commands and input/output contracts as the workflow interface. A new major version marks incompatible changes to that interface.
