#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "01_check_environment.py",
    "02_compute_cohort_and_outcomes.py",
    "03_compute_regional_chaco_statistics.py",
    "04_compute_tfnbs_statistics.py",
    "05_compute_graph_topology_statistics.py",
    "06_compute_multivariate_statistics.py",
    "07_compute_prediction_performance.py",
    "08_compute_qol_associations.py",
    "09_generate_manuscript_outputs.py",
    "11_export_tables_to_docx.py",
    "10_verify_outputs.py",
]

for script in SCRIPTS:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
