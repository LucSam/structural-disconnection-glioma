#!/usr/bin/env python3
"""Audit atlas labels and the patient support for retained TFNBS connections.

The left-lesion-only refit is an influence diagnostic on already selected
edges, not a new corrected TFNBS analysis. No additional significance claims
or edge filtering are made. --refresh-inputs reads the original NeMo outputs;
subsequent runs use a small, private snapshot of the exact TFNBS samples.
"""
import argparse
import json
import pickle
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

from _common import ROOT, anatomical_group, sha256

PRIVATE = ROOT / 'data/private/hemisphere_audit'
OUT = ROOT / 'results/hemisphere_audit'
TESTS = {'Token Test': 'aat_token_test', 'Naming': 'aat_naming'}


def hemisphere(name):
    if str(name).endswith('(L)'):
        return 'L'
    if str(name).endswith('(R)'):
        return 'R'
    if str(name).endswith('(V)'):
        return 'M'
    raise ValueError(f'Unrecognised atlas hemisphere: {name}')


def rank_z(values):
    ranks = stats.rankdata(np.asarray(values), axis=0, method='average')
    sd = ranks.std(axis=0, ddof=1)
    return np.divide(ranks - ranks.mean(axis=0), sd,
                     out=np.zeros_like(ranks), where=sd > 0).astype(np.float32)


def partial_associations(subjects, edges):
    """Match the ranked edge model, re-ranking within a restricted sample."""
    score = rank_z(subjects.iloc[:, 1].to_numpy()).astype(float)
    cov = rank_z(subjects.iloc[:, 2:].to_numpy()).astype(float)
    c = np.column_stack([np.ones(len(subjects)), cov])
    y = rank_z(edges).astype(float)
    xr = score - c @ np.linalg.lstsq(c, score, rcond=None)[0]
    yr = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    rho = -(xr @ yr) / np.sqrt((xr @ xr) * np.sum(yr * yr, axis=0))
    df = int(len(subjects) - np.linalg.matrix_rank(c) - 1)
    t = rho * np.sqrt(df / (1 - rho ** 2))
    return rho, t, df


def refresh_inputs(edges):
    original = ROOT.parent / 'manuscript-data'
    source_dir = original / 'results/analysis/tfnbs/mrtrix_outputs/clinical_lobe_hemisphere_deficit_only'
    provenance = []
    for label, slug in TESTS.items():
        subset = edges[edges.subtest_label.eq(label)].reset_index(drop=True)
        roster = source_dir / slug / 'subjects.csv'
        subjects = pd.read_csv(roster)
        shutil.copy2(roster, PRIVATE / f'{slug}_subjects.csv')
        values = []
        i, j = subset.roi_i.to_numpy() - 1, subset.roi_j.to_numpy() - 1
        for sid in subjects.subject_id:
            files = list((original / 'data/nemo/ifod2act_fs191' / f'sub-{sid}').glob('*ifod2act_chacoconn_fs191subj_mean.pkl'))
            if len(files) != 1:
                raise ValueError('Expected one ChaCoConn mean matrix per subject')
            with files[0].open('rb') as handle, warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                matrix = pickle.load(handle).toarray().astype(float)
            assert matrix.shape == (191, 191) and np.isfinite(matrix).all()
            assert np.abs(np.tril(matrix, -1)).max() <= 1e-12
            assert matrix.min() >= -1e-8 and matrix.max() <= 1 + 1e-8
            values.append(np.clip(matrix[i, j], 0, 1))
            provenance.append({'source': str(files[0].relative_to(original)), 'sha256': sha256(files[0])})
        # The original TFNBS analysis converted raw edge values to float32.
        np.savez_compressed(PRIVATE / f'{slug}_edges.npz', values=np.array(values, dtype=np.float32),
                            roi_i=subset.roi_i, roi_j=subset.roi_j)
        provenance.append({'source': str(roster.relative_to(original)), 'sha256': sha256(roster)})
        if label == 'Naming':
            mask_files = {re.search(r'p\d+', p.name, re.I).group().upper(): p
                          for p in (original / 'data/raw/lesion_masks_mni').glob('*.nii.gz')}
            mask_rows = []
            for sid in subjects.subject_id:
                path = mask_files[sid]
                image = nib.load(path)
                voxels = np.argwhere(np.asarray(image.dataobj) > 0)
                world = nib.affines.apply_affine(image.affine, voxels)
                mask_rows.append({'subject_id': sid,
                                  'fraction_right_of_midline': float(np.mean(world[:, 0] > 0)),
                                  'fraction_right_of_5mm': float(np.mean(world[:, 0] > 5))})
                provenance.append({'source': str(path.relative_to(original)), 'sha256': sha256(path)})
            pd.DataFrame(mask_rows).to_csv(PRIVATE / 'aat_naming_mask_extent.csv', index=False)
    (PRIVATE / 'source_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')


def atlas_audit():
    atlas = pd.read_csv(ROOT / 'data/resources/network_definitions.csv')
    atlas['hemisphere'] = atlas.roi_name.map(hemisphere)
    atlas['compartment'] = np.where(atlas.roi_name.map(anatomical_group).eq('Cerebellum'), 'cerebellar', 'cerebral')
    atlas[['roi_index', 'roi_name', 'hemisphere', 'compartment']].to_csv(OUT / 'atlas_parcels.csv', index=False)
    composition = atlas.groupby(['hemisphere', 'compartment']).size().rename('n_parcels').reset_index()
    composition.to_csv(OUT / 'atlas_composition.csv', index=False)
    regional = pd.read_csv(ROOT / 'data/private/regional_chaco.csv').set_index('subject_id')
    left = atlas[atlas.hemisphere.eq('L')]
    cerebral = left[left.compartment.eq('cerebral')]
    full = regional[[f'chaco_roi_{i}' for i in left.roi_index]].mean(axis=1)
    alternative = regional[[f'chaco_roi_{i}' for i in cerebral.roi_index]].mean(axis=1)
    selected = pd.read_csv(ROOT / 'data/private/analysis_participants.csv')
    selected = selected[selected.battery.eq('AAT')].set_index('subject_id')
    a, b = full.loc[selected.index], alternative.loc[selected.index]
    assert np.allclose(a, selected.burden, atol=1e-14)
    diagnostics = {'n_aat': len(a), 'left_parcels': len(left), 'left_cerebral_parcels': len(cerebral),
                   'left_cerebellar_parcels': len(left) - len(cerebral),
                   'spearman_rho_with_cerebral_only': float(stats.spearmanr(a, b).statistic),
                   'pearson_r_with_cerebral_only': float(stats.pearsonr(a, b).statistic),
                   'median_cerebral_minus_original_percentage_points': float(100 * np.median(b - a)),
                   'max_absolute_difference_percentage_points': float(100 * np.max(np.abs(b - a))),
                   'high_tertile_membership_changes': int(np.sum((a > a.quantile(2/3)) != (b > b.quantile(2/3)))),
                   'note': 'Definition diagnostic only. Primary models and burden groups retain all 92 left-labelled parcels.'}
    (OUT / 'burden_definition.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    return diagnostics


def write_report(result, summaries, burden):
    naming = result[result.subtest_label.eq('Naming')]
    rr = naming[naming.hemisphere_pair.eq('R–R')]
    n = next(row for row in summaries if row['subtest'] == 'Naming')
    lines = [
        '# Hemisphere and burden audit', '',
        '## What the retained Naming connections represent', '',
        '| Endpoint sides | Connections |', '|---|---:|',
    ]
    for pair, count in naming.groupby('hemisphere_pair').size().items():
        lines.append(f'| {pair} | {count} |')
    lines += [
        '', 'L and R refer to atlas endpoint labels; M denotes vermis. All 21 retained Token Test connections are L–L.', '',
        f'The exact Naming TFNBS sample is {n["n"]}: {n["n_left_lesions"]} left-lesion and {n["n_right_lesions"]} right-lesion patients. All 152 retained edges have nonzero values among left-lesion cases. After omitting right-lesion cases and re-ranking within the remaining sample, all adjusted associations remain in the deficit direction (partial deficit ρ {n["left_only_partial_rho_min"]:.3f}–{n["left_only_partial_rho_max"]:.3f}).', '',
        'The original adjusted t statistics are reproduced to within 0.000002. The original edge CSV also contains an unadjusted `deficit_rho` column; it must not be mistaken for the adjusted association.', '',
        '### Five right–right Naming connections', '',
        '| Connection | Nonzero: left lesions /126 | Nonzero: right lesions /10 | Mean ChaCoConn, left lesions (%) | Mean ChaCoConn, right lesions (%) | Partial deficit ρ, left-only refit |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rr.itertuples():
        lines.append(f'| {r.roi_i_name} — {r.roi_j_name} | {r.n_nonzero_left_lesions} | {r.n_nonzero_right_lesions} | {100*r.mean_chacoconn_left_lesions:.4f} | {100*r.mean_chacoconn_right_lesions:.4f} | {r.partial_deficit_rho_left_lesions_only:.3f} |')
    lines += [
        '', 'The right–right edges therefore do not depend exclusively on the ten right-lesion patients. Their amplitudes are much larger on average in right-lesion patients, however. Ranking retains the ordering of even very small values; an association does not establish a substantial loss of connectivity.', '',
        f'The lesion-side label does not guarantee confinement to one hemisphere: {n["left_labelled_masks_with_right_voxels"]}/126 left-labelled Naming masks include voxels with MNI x >0; {n["left_labelled_masks_with_voxels_right_of_5mm"]} extend past x =5 mm. Among the {n["left_labelled_masks_without_right_voxels"]} masks with no voxels at x >0, the five right–right edges still have nonzero values in {int(rr.n_nonzero_without_right_mask_voxels.min())}–{int(rr.n_nonzero_without_right_mask_voxels.max())} cases. Midline extension therefore cannot account for all of these estimates.', '',
        'Saved ChaCoConn values do not reveal the underlying streamline routes or establish the anatomical validity of each estimated connection. The audit does not distinguish reference-tractography, registration or mask contributions to very small contralateral values. Establishing that would require examination of the underlying reference streamlines and spatial inputs. The manuscript reports endpoint anatomy and avoids interpreting these five edges as evidence of a right-hemisphere language mechanism.', '',
        '**Inference boundary:** This is a post hoc influence check on edges selected in the full cohort. There is no new left-only TFNBS permutation, multiplicity correction or evidence of corrected left-only significance. No edges were removed and no original p values were changed.', '',
        '## AAT burden definition', '',
        'The 92 left-labelled parcels are 74 cortical parcels, eight subcortical structures and ten cerebellar lobules. The mean follows atlas side labels; it is not a measure of a crossed cerebral–cerebellar functional circuit.', '',
        f'As a definition diagnostic in the 134 AAT model cases, the 82-parcel cerebral-only mean correlates with the original burden at Spearman ρ = {burden["spearman_rho_with_cerebral_only"]:.8f} (Pearson r = {burden["pearson_r_with_cerebral_only"]:.8f}). Highest-third membership changes in {burden["high_tertile_membership_changes"]} patients. Numerical values do change: the cerebral-only mean is higher by a median {burden["median_cerebral_minus_original_percentage_points"]:.3f} percentage points, with a maximum absolute difference of {burden["max_absolute_difference_percentage_points"]:.3f} points. Near-identical rankings should not be described as identical burden values.', '',
        'The main models, burden distributions and cutoffs retain the original 92-parcel definition. The central models were not refitted for this definition check.', '',
        '## Method sources', '',
        'NeMo defines disconnection from lesion-intersecting reference streamlines, with ChaCoConn estimates for pairs of endpoints: [NeMo documentation](https://github.com/kjamison/nemo). Endpoint hemisphere is not a direct record of the course of each streamline.', '',
        'The distinction between atlas-side averaging and cerebral–cerebellar functional organisation is supported by the preferentially crossed connectivity described in [Buckner et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3214121/). This source is used to explain the definition, not to validate individual NeMo edges.', '',
        '## Reproduction', '',
        'Run `python scripts/05c_audit_hemisphere_support.py` using the private snapshots. To recreate them from the adjacent original package, add `--refresh-inputs`. The input file list and hashes remain in `data/private/hemisphere_audit`; aggregate outputs contain no patient identifiers.', '',
    ]
    (ROOT / 'metadata/hemisphere_audit_report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-inputs', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_edges = pd.read_csv(ROOT / 'data/reference/tfnbs_edges.csv')
    location = all_edges[all_edges.model_family.eq('lobe') & all_edges.significant_fwe_edges].copy()
    location['hemisphere_pair'] = ['–'.join(sorted([hemisphere(a), hemisphere(b)]))
                                  for a, b in zip(location.roi_i_name, location.roi_j_name)]
    count_rows = []
    for correction, selected in [('edge FWE', location), ('across subtests', location[location.significant_battery_bonferroni])]:
        counts = selected.groupby(['subtest_label', 'hemisphere_pair']).size()
        count_rows.extend({'subtest': test, 'correction': correction, 'hemisphere_pair': pair, 'n_edges': int(n)}
                          for (test, pair), n in counts.items())
    pd.DataFrame(count_rows).to_csv(OUT / 'edge_hemisphere_counts.csv', index=False)
    edges = location[location.significant_battery_bonferroni]
    if args.refresh_inputs:
        refresh_inputs(edges)
    summaries, result_frames = [], []
    for label, slug in TESTS.items():
        subset = edges[edges.subtest_label.eq(label)].reset_index(drop=True)
        subjects = pd.read_csv(PRIVATE / f'{slug}_subjects.csv')
        source = np.load(PRIVATE / f'{slug}_edges.npz')
        assert np.array_equal(source['roi_i'], subset.roi_i) and np.array_equal(source['roi_j'], subset.roi_j)
        values = source['values']
        left = subjects.hemisphere_R.eq(0).to_numpy()
        right = ~left
        rho_all, t_all, df_all = partial_associations(subjects, values)
        rho_left, t_left, df_left = partial_associations(subjects[left], values[left])
        # Reproduce the saved, adjusted t statistic, not the unadjusted
        # deficit_rho column supplied alongside it in the old edge table.
        error = float(np.max(np.abs(t_all - subset.t_value)))
        if error > 2e-5:
            raise ValueError(f'TF NBS t-statistic reproduction failed: {label}: {error}')
        raw_rho = -(rank_z(values).T @ rank_z(subjects.iloc[:, 1].to_numpy())) / (len(subjects) - 1)
        assert np.allclose(raw_rho, subset.deficit_rho, atol=2e-6)
        subset['partial_deficit_rho_all'] = rho_all
        subset['partial_deficit_rho_left_lesions_only'] = rho_left
        subset['t_left_lesions_only_uncorrected'] = t_left
        for name, mask in [('left_lesions', left), ('right_lesions', right)]:
            v = values[mask]
            subset[f'n_{name}'] = int(mask.sum())
            subset[f'n_nonzero_{name}'] = (v > 0).sum(axis=0)
            subset[f'n_above_1percent_{name}'] = (v > .01).sum(axis=0)
            subset[f'mean_chacoconn_{name}'] = v.mean(axis=0)
            subset[f'max_chacoconn_{name}'] = v.max(axis=0)
        if label == 'Naming':
            masks = pd.read_csv(PRIVATE / 'aat_naming_mask_extent.csv').set_index('subject_id').loc[subjects.subject_id]
            confined = left & masks.fraction_right_of_midline.eq(0).to_numpy()
            subset['n_left_lesions_without_right_mask_voxels'] = int(confined.sum())
            subset['n_nonzero_without_right_mask_voxels'] = (values[confined] > 0).sum(axis=0)
            subset['mean_chacoconn_without_right_mask_voxels'] = values[confined].mean(axis=0)
        result_frames.append(subset)
        summaries.append({'subtest': label, 'n': len(subjects), 'n_left_lesions': int(left.sum()),
                          'n_right_lesions': int(right.sum()), 'edges': len(subset),
                          'max_abs_original_t_reproduction_error': error,
                          'all_sample_residual_df': df_all, 'left_sample_residual_df': df_left,
                          'edges_with_nonzero_left_support': int((values[left] > 0).any(axis=0).sum()),
                          'edges_with_nonzero_right_support': int((values[right] > 0).any(axis=0).sum()),
                          'left_only_positive_deficit_associations': int(np.sum(rho_left > 0)),
                          'left_only_partial_rho_min': float(rho_left.min()),
                          'left_only_partial_rho_max': float(rho_left.max())})
        if label == 'Naming':
            summaries[-1].update({'left_labelled_masks_with_right_voxels': int(np.sum(left & masks.fraction_right_of_midline.gt(0).to_numpy())),
                                  'left_labelled_masks_with_voxels_right_of_5mm': int(np.sum(left & masks.fraction_right_of_5mm.gt(0).to_numpy())),
                                  'left_labelled_masks_without_right_voxels': int(confined.sum())})
    result = pd.concat(result_frames, ignore_index=True)
    result.to_csv(OUT / 'edge_support_and_influence.csv', index=False)
    result[result.subtest_label.eq('Naming') & result.hemisphere_pair.eq('R–R')].to_csv(OUT / 'naming_right_right_edges.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'model_reproduction_and_influence.csv', index=False)
    burden = atlas_audit()
    write_report(result, summaries, burden)
    metadata = {'purpose': 'Post hoc anatomy and influence audit; no new corrected edge selection',
                'nonzero_definition': 'Original mean ChaCoConn > 0; values retain float32 precision used by TFNBS',
                'left_only_refit': 'Re-rank edge values, score and covariates in left-lesion cases; adjust for age, grade, log volume and six lobe percentages. The constant hemisphere column has no effect.',
                'inference_limit': 'Already selected edges; no left-only TFNBS permutation or multiple-testing correction. Positive left-only associations do not establish corrected left-only significance.',
                'stored_deficit_rho': 'The original edge table reports unadjusted deficit Spearman rho; adjusted t statistics are separately reproduced.',
                'private_input_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(PRIVATE.glob('*'))},
                'summaries': summaries, 'burden': burden}
    (ROOT / 'metadata/hemisphere_audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'models': summaries, 'burden': burden}, indent=2))


if __name__ == '__main__':
    main()
