#!/usr/bin/env python3
"""Reproduce the focused analyses, figures, tables and manuscript from derived inputs."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-inputs', action='store_true',
                        help='Copy minimal inputs again from the original manuscript-data package')
    parser.add_argument('--report-only', action='store_true',
                        help='Rebuild figures and documents from existing v2 result tables')
    args = parser.parse_args()
    if args.refresh_inputs and args.report_only:
        parser.error('--refresh-inputs and --report-only cannot be combined')
    for folder in ['results', 'outputs/figures', 'outputs/tables', 'metadata']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    steps = []
    if args.refresh_inputs:
        steps += ['01_prepare_inputs.py']
    if not args.report_only:
        steps += ['02_compute_staged_models.py', '03_compute_clinical_qol.py',
                  '04_compute_profile_checks.py', '05d_compute_hemisphere_tfnbs.py']
    steps += ['05_prepare_anatomy.py', '05b_prepare_displays.py', '05c_audit_hemisphere_support.py',
              '05e_prepare_group_anatomy.py', '05f_prepare_regional_displays.py', '05g_prepare_network_context.py',
              '05h_describe_battery_overlap.py', '06_create_figures.py', '09_create_circle_alternative.py',
              '07_build_manuscript.py', '08_validate_package.py']
    for step in steps:
        print(f'Running {step}', flush=True)
        command = [sys.executable, str(ROOT / 'scripts' / step)]
        if step in ['05c_audit_hemisphere_support.py', '05d_compute_hemisphere_tfnbs.py'] and args.refresh_inputs:
            command.append('--refresh-inputs')
        subprocess.run(command, check=True, cwd=ROOT)
    print('Finished. Manuscript: manuscript/icons_gp_manuscript_v2.docx', flush=True)


if __name__ == '__main__':
    main()
