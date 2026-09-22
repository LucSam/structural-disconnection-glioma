#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from _shared import RESULTS, build_cohort, build_outcomes

OUT = RESULTS / "cohort"
OUT.mkdir(parents=True, exist_ok=True)


def pct(n: int, d: int) -> str:
    return f"{n} ({n / d * 100:.1f}%)"


def counts(series: pd.Series, order: list[object]) -> dict[object, int]:
    vc = series.value_counts(dropna=False)
    return {key: int(vc.get(key, 0)) for key in order}


cohort = build_cohort()
outcomes = build_outcomes()
cohort.to_csv(OUT / "merged_cohort.csv", index=False)
outcomes.to_csv(OUT / "outcomes_tidy.csv", index=False)

n = len(cohort)
age = pd.to_numeric(cohort["age_at_inclusion"], errors="coerce")
vol = pd.to_numeric(cohort["tumor_volume_ml"], errors="coerce")
sex = counts(cohort["sex"], ["F", "M"])
grade = counts(pd.to_numeric(cohort["grade"], errors="coerce"), [1.0, 2.0, 3.0, 4.0])
hemisphere = counts(cohort["hemisphere"], ["L", "R"])
handedness = counts(cohort["handedness"].astype(str).str.strip().str.lower(), ["r", "l", "a", "nan"])
entity = cohort["entity"].astype(str)
entity_counts = {
    "GBM": int((entity == "GBM").sum()),
    "astrocytoma": int((entity == "astrocytoma").sum()),
    "oligodendroglioma": int((entity == "oligodendroglioma").sum()),
}
entity_counts["other/rare glioma entities"] = n - sum(entity_counts.values())
lobe_raw = cohort["dominant_lobe"].fillna("other").replace(
    {"Putamen": "other", "Caudate": "other", "Thalamus": "other", "Occipital": "other", "Cerebellum": "other", "Other": "other"}
)
lobe = counts(lobe_raw, ["Temporal", "Frontal", "Parietal", "Insula", "other"])

aat_total_n = int(outcomes["AAT_total"].notna().sum())
aat_mean_n = int(outcomes["AAT_mean_T_score"].notna().sum())
aat_mean_below_n = int(pd.to_numeric(outcomes["AAT_mean_T_below_threshold"], errors="coerce").eq(1).sum())
aat_subtest_counts = pd.to_numeric(outcomes["AAT_mean_T_components_available"], errors="coerce")
aat_subtests = {
    "Token Test": ("Token Test_impaired", int(pd.to_numeric(outcomes["Token Test_impaired"], errors="coerce").notna().sum())),
    "Repetition": ("Repetition_impaired", int(pd.to_numeric(outcomes["Repetition_impaired"], errors="coerce").notna().sum())),
    "Naming": ("Naming_impaired", int(pd.to_numeric(outcomes["Naming_impaired"], errors="coerce").notna().sum())),
    "Comprehension": ("Comprehension_impaired", int(pd.to_numeric(outcomes["Comprehension_impaired"], errors="coerce").notna().sum())),
}
aat_parts = []
for label, (col, denom) in aat_subtests.items():
    events = int(pd.to_numeric(outcomes[col], errors="coerce").eq(1).sum())
    aat_parts.append(f"{label} {events}/{denom}")

dem_n = int(outcomes["DemTect_global"].notna().sum())
dem_imp = int(pd.to_numeric(outcomes["DemTect_impaired_binary"], errors="coerce").eq(1).sum())
qol = {
    "EORTC QLQ-C30 global health": int(outcomes["EORTC_QLQ_C30_global_health"].notna().sum()),
    "C30 cognitive functioning": int(outcomes["EORTC_QLQ_C30_cognitive_functioning"].notna().sum()),
    "BN20 communication": int(outcomes["EORTC_QLQ_BN20_communication_deficit"].notna().sum()),
}

table = pd.DataFrame(
    [
        ("N", str(n)),
        ("Age at inclusion", f"{age.mean():.1f} +/- {age.std(ddof=1):.1f} years"),
        ("Sex", f"female: {pct(sex['F'], n)}; male: {pct(sex['M'], n)}"),
        ("WHO grade", "; ".join(f"{roman}: {grade[key]}" for key, roman in [(1.0, "I"), (2.0, "II"), (3.0, "III"), (4.0, "IV")])),
        ("Histological entity", "; ".join(f"{key}: {value}" for key, value in entity_counts.items())),
        ("Hemisphere", f"left: {pct(hemisphere['L'], n)}; right: {pct(hemisphere['R'], n)}"),
        (
            "Handedness",
            f"right: {pct(handedness['r'], n)}; left: {pct(handedness['l'], n)}; "
            f"ambidextrous: {pct(handedness['a'], n)}; missing: {pct(handedness['nan'], n)}",
        ),
        ("Dominant lobe", f"Temporal: {lobe['Temporal']}; frontal: {lobe['Frontal']}; parietal: {lobe['Parietal']}; insula: {lobe['Insula']}; other: {lobe['other']}"),
        ("Tumour volume", f"median {vol.median():.1f} ml; IQR {vol.quantile(0.25):.1f}-{vol.quantile(0.75):.1f}; range {vol.min():.1f}-{vol.max():.1f}"),
        (
            "AAT raw total / mean T-score availability",
            f"AAT raw total n={aat_total_n}; recorded mean AAT T-score n={aat_mean_n}; "
            f"four subtests n={int(aat_subtest_counts.eq(4).sum())}; "
            f"three subtests n={int(aat_subtest_counts.eq(3).sum())}",
        ),
        (
            "Mean AAT T-score operational group",
            f"below 63.5: {aat_mean_below_n}/{aat_mean_n} ({aat_mean_below_n / aat_mean_n * 100:.1f}%); "
            f"at or above 63.5: {aat_mean_n - aat_mean_below_n}/{aat_mean_n}",
        ),
        ("AAT subtest T-scores below boundary", "; ".join(aat_parts)),
        ("DemTect availability/impairment", f"DemTect global n={dem_n}; impaired <13: {dem_imp}/{dem_n} ({dem_imp / dem_n * 100:.1f}%)"),
        ("Exploratory QoL endpoints", "; ".join(f"{key} n={value}" for key, value in qol.items())),
    ],
    columns=["Variable", "Value"],
)
table.to_csv(OUT / "cohort_characteristics.csv", index=False)

availability = outcomes.notna().sum().rename("n_nonmissing").reset_index()
availability.columns = ["measure", "n_nonmissing"]
availability.to_csv(OUT / "outcome_availability.csv", index=False)
print("OK: computed cohort and outcome summaries from raw source files")
