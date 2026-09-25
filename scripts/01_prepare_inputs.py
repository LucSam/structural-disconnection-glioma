#!/usr/bin/env python3
"""Copy the necessary derived inputs and validated reference outputs into v2."""
import argparse
import json
import shutil
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from _common import ROOT, sha256


def prepare_figure_inputs(source):
    masks = sorted((source / 'data/raw/lesion_masks_mni').glob('*.nii.gz'))
    if len(masks) != 163:
        raise ValueError('The cohort figure requires all 163 original lesion masks')
    reference = nib.load(masks[0])
    overlap = np.zeros(reference.shape, dtype=np.uint16)
    content_hash = hashlib.sha256()
    for path in masks:
        image = nib.load(path)
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError('Lesion mask grids differ')
        overlap += np.asarray(image.dataobj) > 0
        content_hash.update(bytes.fromhex(sha256(path)))
    destination = ROOT / 'data/resources/cohort_lesion_overlap.nii.gz'
    nib.save(nib.Nifti1Image(overlap, reference.affine), destination)
    (ROOT / 'metadata/lesion_overlap_provenance.json').write_text(json.dumps({
        'source': 'Original cohort of 163 MNI lesion masks', 'n': len(masks),
        'source_content_digest': content_hash.hexdigest(),
        'destination': str(destination.relative_to(ROOT)), 'destination_sha256': sha256(destination),
        'maximum_overlap': int(overlap.max()),
        'individual_masks_copied': False,
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'manuscript-data')
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = []
    for folder in ['data/private', 'data/resources', 'data/reference', 'metadata']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    def record(src, dest, action):
        manifest.append({'source': str(src.relative_to(source)), 'source_sha256': sha256(src),
                         'destination': str(dest.relative_to(ROOT)), 'destination_sha256': sha256(dest), 'action': action})

    for srcname, destname, columns in [
        ('cohort/merged_cohort.csv', 'cohort.csv', ['subject_id','age_at_inclusion','grade','tumor_volume_ml','hemisphere','sex']),
        ('cohort/outcomes_tidy.csv', 'outcomes.csv', None),
        ('regional_chaco/fs191_regional_chaco_wide.csv', 'regional_chaco.csv', None),
    ]:
        src = source / 'results/analysis' / srcname
        dest = ROOT / 'data/private' / destname
        if columns:
            available = pd.read_csv(src, nrows=0).columns
            columns = [c for c in columns if c in available]
        pd.read_csv(src, usecols=columns).to_csv(dest, index=False)
        record(src, dest, 'selected columns' if columns else 'derived table')

    resources = {'data/resources/network_definitions.csv': 'data/resources/network_definitions.csv',
                 'results/analysis/regional_chaco/fs191_centroids.csv': 'data/resources/fs191_centroids.csv'}
    references = {
        'multivariate/conditional_component_tests.csv': 'conditional_component_tests.csv',
        'multivariate/regional_chaco_pca_variance.csv': 'regional_chaco_pca_variance.csv',
        'cohort/cohort_characteristics.csv': 'cohort_characteristics.csv',
        'qol/qol_neuropsych_correlations.csv': 'qol_neuropsych_correlations.csv',
        'tfnbs/tfnbs_significant_and_top_edges.csv': 'tfnbs_edges.csv',
        'tfnbs/table_02_tfnbs_subtest_summary.csv': 'tfnbs_summary.csv',
        'tfnbs/tfnbs_edge_counts_from_mrtrix.csv': 'tfnbs_counts.csv',
        'tfnbs/tfnbs_parameters.json': 'tfnbs_parameters.json',
        'regional_chaco/regional_chaco_global_score_partial_sensitivity_summary.csv': 'regional_sensitivity_summary.csv',
    }
    resources.update({f'results/analysis/{k}':f'data/reference/{v}' for k,v in references.items()})
    for srcname, destname in resources.items():
        src, dest = source / srcname, ROOT / destname
        shutil.copy2(src, dest)
        record(src, dest, 'unchanged reference')

    # Preserve the original TFNBS generator and its helpers as provenance.
    # They are not called by this focused pipeline; TFNBS is unchanged.
    original = ROOT / 'metadata/original_tfnbs_code'
    original.mkdir(exist_ok=True)
    for name in ['04_compute_tfnbs_statistics.py','_shared.py','_table_exports.py']:
        src, dest = source / 'scripts' / name, original / name
        shutil.copy2(src, dest)
        record(src, dest, 'TFNBS source provenance; original package layout required to rerun')
    (ROOT/'metadata/input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    prepare_figure_inputs(source)
    print('Prepared minimal derived inputs; direct clinical identifier columns excluded.')


if __name__ == '__main__':
    main()
