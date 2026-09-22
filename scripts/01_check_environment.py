#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "bct", "matplotlib", "seaborn", "nibabel", "nilearn", "PIL", "docx"]
PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "bctpy": "bctpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "nibabel": "nibabel",
    "nilearn": "nilearn",
    "Pillow": "Pillow",
    "python-docx": "python-docx",
}

missing = []
for package in PACKAGES:
    if find_spec(package) is None:
        missing.append(package)

checks = {
    "lesion masks": len(list((ROOT / "data/raw/lesion_masks_mni").glob("*.nii.gz"))),
    "NeMo upload zips": len(list((ROOT / "data/nemo/upload_batches").glob("*.zip"))),
    "NeMo download zips": len(list((ROOT / "data/nemo/downloaded_results").glob("*.zip"))),
    "NeMo iFOD2+ACT/fs191 files": len(list((ROOT / "data/nemo/ifod2act_fs191").glob("sub-P*/*"))),
}
expected = {"lesion masks": 163, "NeMo upload zips": 18, "NeMo download zips": 18}
for label, value in checks.items():
    if label in expected and value != expected[label]:
        missing.append(f"{label}: expected {expected[label]}, observed {value}")
    if label == "NeMo iFOD2+ACT/fs191 files" and value < 480:
        missing.append(f"{label}: expected at least 480, observed {value}")
required_files = [
    "data/raw/clinical_outcome_metadata/data_mr.csv",
    "data/raw/clinical_outcome_metadata/patients_mr.csv",
    "data/raw/clinical_outcome_metadata/tumor_volumes_ml.csv",
    "data/raw/lesion_metadata/atlas_mask_coverage.csv",
    "data/resources/network_definitions.csv",
]
for rel in required_files:
    if not (ROOT / rel).exists():
        missing.append(f"missing required file: {rel}")

connectomestats = os.environ.get("CONNECTOMESTATS") or shutil.which("connectomestats")
if not connectomestats:
    missing.append("missing executable: connectomestats; set CONNECTOMESTATS to the MRtrix3 connectomestats path")

expected_versions = json.loads((ROOT / "metadata/software_versions.json").read_text(encoding="utf-8"))
version_mismatches = []
if sys.version.split()[0] != expected_versions["python"]:
    version_mismatches.append(f"python: expected {expected_versions['python']}, observed {sys.version.split()[0]}")
for key, distribution in PACKAGE_DISTRIBUTIONS.items():
    try:
        observed = version(distribution)
    except PackageNotFoundError:
        continue
    if observed != expected_versions[key]:
        version_mismatches.append(f"{distribution}: expected {expected_versions[key]}, observed {observed}")
if connectomestats:
    completed = subprocess.run([connectomestats, "-version"], capture_output=True, text=True, check=True)
    version_line = next((line for line in completed.stdout.splitlines() if "connectomestats" in line), "")
    observed = version_line.replace("==", "").replace("connectomestats", "").strip()
    if observed != expected_versions["mrtrix_connectomestats"]:
        version_mismatches.append(
            f"connectomestats: expected {expected_versions['mrtrix_connectomestats']}, observed {observed or 'unknown'}"
        )

if version_mismatches and os.environ.get("ICONS_ALLOW_VERSION_MISMATCH") != "1":
    missing.extend([f"version mismatch: {item}" for item in version_mismatches])

if missing:
    raise SystemExit("\n".join(missing))
print("OK: pinned environment and raw/NeMo source data counts")
