#!/usr/bin/env python3
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.io import loadmat  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RESULTS = ROOT / "results/analysis"

AAT_MEAN_T_THRESHOLD = 63.5
DEMTECT_THRESHOLD = 13.0
SUBTEST_THRESHOLDS = {
    "aat_token_t_max73": 63.0,
    "aat_rep_overall_t_max71": 63.0,
    "aat_nam_t_max80": 64.0,
    "aat_comp_t_max78": 64.0,
}
SUBTEST_LABELS = {
    "aat_token_t_max73": "Token Test",
    "aat_rep_overall_t_max71": "Repetition",
    "aat_nam_t_max80": "Naming",
    "aat_comp_t_max78": "Comprehension",
}
# Manuscript-facing labels; source column names remain stable for data joins.
# AAT task identities follow the hand manual (Nachsprechen, p. 21;
# Sprachverstaendnis, p. 23) and the OUTCOME_MAP source-variable mapping.
REGIONAL_OUTCOME_LABELS = {
    "AAT_total": "AAT total",
    "DemTect_global": "DemTect global",
    "Token Test | T Score (p. 133)": "Token Test",
    "repetition | average T Score across both tests": "Repetition",
    "repetition | T Score test 4 (p. 139)": "Repetition: compound words",
    "repetition | T Score test 5 (p. 139)": "Repetition: sentences",
    "naming | T Score (p. 135)": "Naming",
    "comprehension | T Score (p. 136)": "Comprehension",
    "auditory comprehension (general) | T Score aud. comp. (p. 140)": "Auditory Comprehension",
    "auditory comprehension (detailed) | T Score test 2 (p. 139)": "Auditory Comprehension: sentences",
    "reading comprehension (general) | T Score read. comp,. (p. 140)": "Reading Comprehension",
    "reading comprehension (detailed) | T Score test 4 (139)": "Reading Comprehension: sentences",
    "word list | first pass - out of 20": "Word List",
    "word list | delayed recall - out of 10": "Delayed Recall",
    "number conversion | out of 4": "Number Conversion",
    "verbal fluency | out of 20": "Verbal Fluency",
    "digit span backwards | out of 6": "Digit Span Backwards",
}
OUTCOME_MAP = {
    "DemTect_global": "demtect_overall_max18",
    "word list | first pass - out of 20": "demtect_imm_recall_max20",
    "word list | delayed recall - out of 10": "demtect_del_recall_max10",
    "number conversion | out of 4": "demtect_num_conv_max4",
    "verbal fluency | out of 20": "demtect_verb_flu_max20",
    "digit span backwards | out of 6": "demtect_digit_span_backw_max6",
    "Token Test | T Score (p. 133)": "aat_token_t_max73",
    "naming | T Score (p. 135)": "aat_nam_t_max80",
    "comprehension | T Score (p. 136)": "aat_comp_t_max78",
    "auditory comprehension (general) | T Score aud. comp. (p. 140)": "aat_audcomp_t_max80",
    "auditory comprehension (detailed) | T Score test 1 (p. 139)": "aat_audcomp_words_t_max72",
    "auditory comprehension (detailed) | T Score test 2 (p. 139)": "aat_audcomp_sent_t_max76",
    "reading comprehension (general) | T Score read. comp,. (p. 140)": "aat_readcomp_t_max78",
    "reading comprehension (detailed) | T Score test 3 (p. 139)": "aat_readcomp_words_t_max72",
    "reading comprehension (detailed) | T Score test 4 (139)": "aat_readcomp_sent_t_max75",
    "repetition | T Score test 4 (p. 139)": "aat_rep_comp_t_max71",
    "repetition | T Score test 5 (p. 139)": "aat_rep_sent_t_max71",
    "repetition | average T Score across both tests": "aat_rep_overall_t_max71",
    "written language | T score test 1 (p. 139)": "aat_writ_read_t_max70",
    "written language | T score test 3 (p. 139)": "aat_writ_writ_t_max74",
    "written language | average T Score": "aat_writ_overall_t_max72",
    "EORTC_QLQ_C30_global_health": "c30_funct_global_score_best100",
    "EORTC_QLQ_C30_physical_functioning": "c30_funct_phys_score_best100",
    "EORTC_QLQ_C30_role_functioning": "c30_funct_role_score_best100",
    "EORTC_QLQ_C30_emotional_functioning": "c30_funct_emo_score_best100",
    "EORTC_QLQ_C30_cognitive_functioning": "c30_funct_cogn_score_best100",
    "EORTC_QLQ_C30_social_functioning": "c30_funct_soc_score_best100",
    "EORTC_QLQ_C30_fatigue": "c30_sympt_fatigue_score_best0",
    "EORTC_QLQ_C30_nausea_vomiting": "c30_sympt_naus_score_best0",
    "EORTC_QLQ_C30_pain": "c30_sympt_pain_score_best0",
    "EORTC_QLQ_BN20_future_uncertainty": "bn20_funct_futunc_score_best100",
    "EORTC_QLQ_BN20_visual_disorder": "bn20_funct_vis_score_best100",
    "EORTC_QLQ_BN20_motor_dysfunction": "bn20_funct_motor_score_best100",
    "EORTC_QLQ_BN20_communication_deficit": "bn20_funct_comm_score_best100",
    "EORTC_QLQ_BN20_headaches": "bn20_sympt_headach_score_best0",
    "EORTC_QLQ_BN20_seizures": "bn20_sympt_seiz_score_best0",
    "EORTC_QLQ_BN20_drowsiness": "bn20_sympt_drows_score_best0",
    "EORTC_QLQ_BN20_hair_loss": "bn20_sympt_hairl_score_best0",
    "EORTC_QLQ_BN20_itchy_skin": "bn20_sympt_itch_score_best0",
    "EORTC_QLQ_BN20_bladder_control": "bn20_sympt_bladcon_score_best0",
}
ANATOMICAL_ORDER = [
    "Subcortical/limbic",
    "Cerebellum",
    "Frontal/insula",
    "Perisylvian/central sulci",
    "Temporal",
    "Parietal",
    "Occipital/visual",
    "Cingulate/medial",
]
ANATOMICAL_GROUP_ORDER = {name: idx for idx, name in enumerate(ANATOMICAL_ORDER)}


def clean_subject_id(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("P"):
        try:
            return f"P{int(text[1:]):03d}"
        except ValueError:
            return text
    return text


def clean_subject_series(series: pd.Series) -> pd.Series:
    return series.map(clean_subject_id)


def read_raw_outcomes() -> pd.DataFrame:
    path = RAW / "clinical_outcome_metadata/data_mr.csv"
    data = pd.read_csv(path, sep=";", decimal=",", na_values=["N/A", "NA", ""], encoding="utf-8-sig", engine="python")
    data = data[[col for col in data.columns if not str(col).startswith("Unnamed:")]].copy()
    data["subject_id"] = clean_subject_series(data["icons_id"])
    data = data[data["subject_id"].ne("NAN")].drop_duplicates("subject_id", keep="first")
    return data.set_index("subject_id", drop=False)


def read_raw_patients() -> pd.DataFrame:
    patients = pd.read_csv(RAW / "clinical_outcome_metadata/patients_mr.csv")
    patients["subject_id"] = clean_subject_series(patients["icons_id"])
    patients = patients[patients["subject_id"].ne("NAN")].drop_duplicates("subject_id", keep="first")
    return patients


def read_tumor_volumes() -> pd.DataFrame:
    volumes = pd.read_csv(RAW / "clinical_outcome_metadata/tumor_volumes_ml.csv")
    volumes["subject_id"] = clean_subject_series(volumes["subject_id"])
    volumes["tumor_volume_ml"] = pd.to_numeric(volumes["tumor_volume_ml"], errors="coerce")
    return volumes


def dominant_lobe_from_atlas() -> pd.DataFrame:
    coverage = pd.read_csv(RAW / "lesion_metadata/atlas_mask_coverage.csv")
    coverage["subject_id"] = clean_subject_series(coverage["subject_id"])
    coverage = coverage[coverage["atlas"].eq("MNI Structural Atlas")].copy()
    # FSL atlasq reports the 0-100-scaled mean atlas probability across mask voxels.
    # Keep the legacy output name for release-table compatibility.
    idx = coverage.groupby("subject_id")["pct"].idxmax()
    dominant = coverage.loc[idx, ["subject_id", "area", "pct"]].rename(columns={"area": "dominant_region", "pct": "dominant_region_pct"})
    lobe_map = {
        "Frontal Lobe": "Frontal",
        "Temporal Lobe": "Temporal",
        "Parietal Lobe": "Parietal",
        "Occipital Lobe": "Occipital",
        "Insula": "Insula",
        "Caudate": "Caudate",
        "Putamen": "Putamen",
        "Thalamus": "Thalamus",
        "Cerebellum": "Cerebellum",
    }
    dominant["dominant_lobe"] = dominant["dominant_region"].map(lobe_map).fillna("Other")
    return dominant


def build_outcomes() -> pd.DataFrame:
    raw = read_raw_outcomes()
    subjects = raw["subject_id"].tolist()
    outcomes = pd.DataFrame({"subject_id": subjects})
    outcomes["AAT_total"] = pd.to_numeric(raw["aat_overall_raw_max350"], errors="coerce").reindex(subjects).to_numpy()
    for clean_name, raw_col in OUTCOME_MAP.items():
        outcomes[clean_name] = pd.to_numeric(raw[raw_col], errors="coerce").reindex(subjects).to_numpy()
    aat_component_columns = [
        "Token Test | T Score (p. 133)",
        "repetition | average T Score across both tests",
        "naming | T Score (p. 135)",
        "comprehension | T Score (p. 136)",
    ]
    component_count = outcomes[aat_component_columns].notna().sum(axis=1)
    component_mean = outcomes[aat_component_columns].mean(axis=1, skipna=True).where(component_count.ge(3))
    recorded_mean = pd.to_numeric(raw["aat_overall_t_max755"], errors="coerce").reindex(subjects).reset_index(drop=True)
    comparable = component_mean.notna() & recorded_mean.notna()
    if not np.allclose(component_mean.loc[comparable], recorded_mean.loc[comparable]):
        raise ValueError("Recorded AAT mean T-scores do not match the available subtest-score means")
    aat_mean_t_score = recorded_mean.where(component_count.ge(3))
    outcomes["AAT_mean_T_score"] = aat_mean_t_score
    outcomes["AAT_mean_T_components_available"] = component_count.where(aat_mean_t_score.notna())
    outcomes["AAT_mean_T_below_threshold"] = np.where(
        aat_mean_t_score.notna(),
        aat_mean_t_score.lt(AAT_MEAN_T_THRESHOLD).astype(float),
        np.nan,
    )
    outcomes["DemTect_impaired_binary"] = np.where(
        outcomes["DemTect_global"].notna(),
        (outcomes["DemTect_global"] < DEMTECT_THRESHOLD).astype(float),
        np.nan,
    )
    for raw_col, label in SUBTEST_LABELS.items():
        values = pd.to_numeric(raw[raw_col], errors="coerce").reindex(subjects)
        outcomes[f"{label}_impaired"] = np.where(values.notna(), (values < SUBTEST_THRESHOLDS[raw_col]).astype(bool), np.nan)
    return outcomes


def build_cohort() -> pd.DataFrame:
    raw = read_raw_outcomes().reset_index(drop=True)
    patients = read_raw_patients()
    volumes = read_tumor_volumes()
    dominant = dominant_lobe_from_atlas()
    cohort = raw.merge(patients, on="subject_id", how="left", suffixes=("", "_patient"))
    cohort = cohort.merge(volumes, on="subject_id", how="left")
    cohort = cohort.merge(dominant, on="subject_id", how="left")
    cohort["age_at_inclusion"] = pd.to_numeric(cohort["age_at_incl"], errors="coerce")
    cohort["sex"] = cohort["sex"].astype(str).str.strip().str.upper().map({"F": "F", "M": "M"})
    cohort["hemisphere"] = cohort["tumour_hemisphere"].astype(str).str.strip().str.upper()
    cohort["grade"] = np.floor(pd.to_numeric(cohort["tumour_grade"], errors="coerce"))
    cohort["entity"] = cohort["tumour_entity"].astype(str).str.strip()
    return cohort


def anatomical_group(roi_name: object) -> str:
    name = str(roi_name).lower()
    first_token = name.replace("(", " ").replace(")", " ").split()[0] if name.strip() else ""
    if any(term in name for term in ["subcallosal", "pericallosal", "cingul", "precuneus"]):
        return "Cingulate/medial"
    if any(term in name for term in ["thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens", "ventraldc"]):
        return "Subcortical/limbic"
    if first_token in {"i_iv", "v", "vi", "viib", "viiia", "viiib", "ix", "x"} or any(term in name for term in ["crus", "lobule"]):
        return "Cerebellum"
    if any(term in name for term in ["lat_fis", "s_central", "g_ins_lg", "cent_ins"]):
        return "Perisylvian/central sulci"
    if any(term in name for term in ["occip", "cuneus", "calcarine", "lingual", "lunatus", "s_oc_"]):
        return "Occipital/visual"
    if any(term in name for term in ["temp", "heschl", "fusiform", "sts", "transverse", "collat"]):
        return "Temporal"
    if any(term in name for term in ["pariet", "postcentral", "angular", "supramarg", "intrapariet", "jensen"]):
        return "Parietal"
    if any(term in name for term in ["front", "precentral", "paracentral", "subcentral", "insula", "orbital", "opercular", "rectus"]):
        return "Frontal/insula"
    return "Perisylvian/central sulci"


def load_fs191_labels() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "data/resources/network_definitions.csv")
    labels["roi_index"] = labels["roi_index"].astype(int)
    labels["roi_col"] = "chaco_roi_" + labels["roi_index"].astype(str)
    labels["anatomical_group"] = labels["roi_name"].map(anatomical_group)
    labels["anatomical_code"] = labels["anatomical_group"].map(ANATOMICAL_GROUP_ORDER)
    labels["yeo7_name"] = labels["primary_yeo_network"].astype(str)
    return labels.sort_values("roi_index").reset_index(drop=True)


def subject_id_from_nemo_dir(path: Path) -> str:
    return clean_subject_id(path.name.replace("sub-", ""))


def load_nemo_regional_chaco() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sub_dir in sorted((ROOT / "data/nemo/ifod2act_fs191").glob("sub-P*")):
        files = sorted(sub_dir.glob("*ifod2act_chacovol_fs191subj_mean.pkl"))
        if not files:
            continue
        with files[0].open("rb") as handle:
            values = np.asarray(pickle.load(handle), dtype=float).reshape(-1)
        row: dict[str, float | str] = {"subject_id": subject_id_from_nemo_dir(sub_dir)}
        for idx, value in enumerate(values, start=1):
            row[f"chaco_roi_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def mirror_nemo_upper_triangle(matrix: np.ndarray, *, matrix_name: str) -> np.ndarray:
    """Mirror a NeMo upper-triangular undirected matrix without rescaling it."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_name} must be a square matrix; observed {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{matrix_name} contains non-finite values")
    lower = np.tril(matrix, k=-1)
    if np.any(np.abs(lower) > 1e-12):
        raise ValueError(f"{matrix_name} is not encoded as an upper-triangular NeMo matrix")
    upper = np.triu(matrix, k=1)
    mirrored = upper + upper.T
    np.fill_diagonal(mirrored, 0.0)
    return mirrored


def load_nemo_edge_matrices() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load NeMo predicted remaining SIFT2 SC and pairwise ChaCo matrices.

    NeMo's ``nemoSC`` output is connectivity after lesion-intersecting
    streamlines have been removed. It is therefore not an intact normative
    matrix and must not be attenuated by ``(1 - ChaCoConn)`` a second time.
    Both NeMo outputs are stored as upper triangles and are mirrored here by
    addition, not averaging.
    """
    remaining_sc: dict[str, np.ndarray] = {}
    chaco_edge: dict[str, np.ndarray] = {}
    for sub_dir in sorted((ROOT / "data/nemo/ifod2act_fs191").glob("sub-P*")):
        sid = subject_id_from_nemo_dir(sub_dir)
        sc_files = sorted(sub_dir.glob("*ifod2act*fs191subj*nemoSC*mean.mat"))
        cc_files = sorted(sub_dir.glob("*ifod2act_chacoconn_fs191subj_mean.pkl"))
        if not sc_files or not cc_files:
            continue
        sc_raw = loadmat(sc_files[0])["SC"].astype(float)
        sc_sym = mirror_nemo_upper_triangle(sc_raw, matrix_name=f"{sid} nemoSC")
        with cc_files[0].open("rb") as handle, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Please import `csr_matrix`", category=DeprecationWarning)
            cc_sparse = pickle.load(handle)
        cc_dense = cc_sparse.toarray().astype(float)
        if np.nanmin(cc_dense) < -1e-8 or np.nanmax(cc_dense) > 1.0 + 1e-8:
            raise ValueError(f"{sid} ChaCoConn values fall outside the expected [0, 1] range")
        cc_sym = np.clip(
            mirror_nemo_upper_triangle(cc_dense, matrix_name=f"{sid} ChaCoConn"),
            0.0,
            1.0,
        )
        if sc_sym.shape != cc_sym.shape:
            raise ValueError(f"{sid} nemoSC and ChaCoConn shapes differ")
        remaining_sc[sid] = sc_sym
        chaco_edge[sid] = cc_sym
    return remaining_sc, chaco_edge


def generated_outcomes() -> pd.DataFrame:
    path = RESULTS / "cohort/outcomes_tidy.csv"
    if not path.exists():
        return build_outcomes()
    return pd.read_csv(path)


def generated_cohort() -> pd.DataFrame:
    path = RESULTS / "cohort/merged_cohort.csv"
    if not path.exists():
        return build_cohort()
    return pd.read_csv(path)


def generated_regional_chaco() -> pd.DataFrame:
    path = RESULTS / "regional_chaco/fs191_regional_chaco_wide.csv"
    if not path.exists():
        return load_nemo_regional_chaco()
    return pd.read_csv(path)
