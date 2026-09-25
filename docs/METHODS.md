# Methods overview

The clinical question is why some patients with glioma have preoperative deficits while others retain measured function despite extensive estimated disconnection, and how these differences relate to patient-reported function. NeMo characterises the estimated structural lesion effects; it does not measure patient-specific reserve.

## Central comparison

AAT and DemTect are modelled separately. Each model considers all principal subtests together: four AAT T-scores or five DemTect raw subtest scores. The model sequence is:

1. Age, WHO grade, log-transformed tumour volume and hemisphere.
2. Add disconnection burden (mean regional ChaCo).
3. Add the disconnection pattern, represented by eight principal components of the 191 regional values.

The same patients enter each step within a battery. Scores are inverse-normal rank transformed. PCA is fitted to the standardised regional values without using test scores. The first eight components explain 81.8% of regional ChaCo variation, not performance variation. Anatomical-group means provide an alternative representation of the pattern. Profile contrasts separately examine whether associations differ between subtests.

## Roles and correction families

| Question | Adjustment or role | Multiple-testing control |
| --- | --- | --- |
| Do burden and pattern relate to the subtests together? | Central model sequence; anatomical-group sensitivity. | 19,999 Freedman–Lane permutations; FDR across six model comparisons. |
| Do associations differ between subtests? | Same clinical factors, burden and pattern; interpretive profile checks. | Holm correction across five profile tests; a separate 16-contrast Holm family, interpreted only if the parent profile test passes. All results are retained. |
| Where are regional associations? | Partial rank correlations adjusted for age, grade, tumour volume and hemisphere; descriptive maps of all parcels. | No new regional significance family or threshold-based parcel selection. |
| Which connections relate to lower scores? | Ranked clinical-plus-hemisphere model; additional lobe adjustment as sensitivity. Localising models do not include burden. | TFNBS edge-FWE with 5,000 shuffles, then Bonferroni across four AAT or five DemTect subtests, separately per model. |
| Do test scores relate to reported function? | Unadjusted Spearman correlations. | FDR across all 222 estimable neuropsychological–QoL pairs. |
| Does structural disconnection relate to reported function? | Descriptive burden and selected-connection summaries. | Historical whole-brain burden/top-decile tests retain both the six-test FDR family and broader 51-test correction. No additional significance family is introduced. |

These analyses answer complementary questions. Their p values are not combined into one overall test. The hierarchy describes an exploratory revision, not a prospectively registered plan. Original graph results remain in the historical 51-test correction even though graph analyses are absent from the focused narrative.

## Descriptive comparisons

No AAT deficit requires all four principal subtests to meet their boundaries: Token Test and Repetition ≥63, Naming and Comprehension ≥64. No DemTect deficit means global score ≥13. These classifications concern the measured functions and are not standalone diagnoses.

High burden is the highest third within each battery's complete-case sample. Belonging to the same third does not make groups burden-matched. Regional and anatomical connection differences are unadjusted descriptions. Both regional and pairwise summaries retain all atlas entries, including zeros. Anatomical groups, hemisphere and cerebral/cerebellar compartments describe the structural findings.

Structural–QoL summaries based on TFNBS edges use sets selected through performance in the same cohort. They are descriptive associations, not independent validation or evidence of mediation. The public source retains the numerical checks and full correction families used by the working analysis.
