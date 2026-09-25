#!/usr/bin/env python3
"""TFNBS with clinical covariates and hemisphere for all nine subtests.

Uses the original per-subtest samples, ranked inputs, one-sided deficit
contrast and correction families. Lobe-adjusted results remain a separate
sensitivity analysis. Private input snapshots permit reruns without the
original project tree. Existing runs are reused only if their inputs,
parameters, executable and output hashes agree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from _common import ROOT, OUTCOMES, anatomical_group, sha256

PRIVATE = ROOT / 'data/private/tfnbs_hemisphere'
RUNS = PRIVATE / 'runs'
OUT = ROOT / 'results/tfnbs_hemisphere'
COVARIATES = ['age_years', 'grade_int', 'log_tumor_volume_ml', 'hemisphere_R']
SLUGS = ['aat_token_test', 'aat_repetition', 'aat_naming', 'aat_comprehension',
         'demtect_word_list', 'demtect_delayed_recall', 'demtect_number_conversion',
         'demtect_verbal_fluency', 'demtect_digit_span_backwards']
SPECS = [(battery, name, col, slug, len(tests))
         for (battery, name, col), slug in zip(
             [(b, name, col) for b, tests in OUTCOMES.items() for name, col in tests.items()], SLUGS, strict=True)
         for tests in [OUTCOMES[battery]]]
PARAMETERS = {'nshuffles': 5000, 'tfce_dh': .1, 'tfce_e': .4, 'tfce_h': 3,
              'base_seed': 20260923, 'nthreads': 2, 'contrast': 'greater disconnection with lower score'}


def rank_z(values):
    ranks = stats.rankdata(values, axis=0, method='average')
    sd = ranks.std(axis=0, ddof=1)
    return np.divide(ranks - ranks.mean(axis=0), sd,
                     out=np.zeros_like(ranks), where=sd > 0).astype(np.float32)


def prepare_inputs():
    original = ROOT.parent / 'manuscript-data'
    files = sorted((original / 'data/nemo/ifod2act_fs191').glob('sub-*/*ifod2act_chacoconn_fs191subj_mean.pkl'))
    i, j = np.triu_indices(191, 1)
    values, ids, manifest = [], [], []
    for path in files:
        with path.open('rb') as handle, warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            matrix = pickle.load(handle).toarray().astype(float)
        assert matrix.shape == (191, 191) and np.isfinite(matrix).all()
        assert np.abs(np.tril(matrix, -1)).max() <= 1e-12
        assert matrix.min() >= -1e-8 and matrix.max() <= 1 + 1e-8
        ids.append(path.parent.name.removeprefix('sub-'))
        values.append(np.clip(matrix[i, j], 0, 1))
        manifest.append({'source': str(path.relative_to(original)), 'sha256': sha256(path)})
    assert len(ids) == len(set(ids)) == 163
    np.savez_compressed(PRIVATE / 'chacoconn_edges.npz', values=np.array(values, dtype=np.float32),
                        subject_ids=np.array(ids), roi_i=i + 1, roi_j=j + 1)
    for _, _, col, slug, _ in SPECS:
        source = original / 'results/analysis/tfnbs/mrtrix_outputs/clinical_lobe_hemisphere_deficit_only' / slug / 'subjects.csv'
        frame = pd.read_csv(source)[['subject_id', col, *COVARIATES]]
        assert frame.drop(columns='subject_id').notna().all().all()
        frame.to_csv(PRIVATE / f'{slug}_subjects.csv', index=False)
        manifest.append({'source': str(source.relative_to(original)), 'sha256': sha256(source)})
    (PRIVATE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run_one(spec, source, executable, version):
    battery, name, column, slug, n_tests = spec
    start = time.monotonic()
    subjects = pd.read_csv(PRIVATE / f'{slug}_subjects.csv')
    ids = subjects.subject_id.to_numpy()
    seed_text = f'{PARAMETERS["base_seed"]}|clinical_hemisphere/{slug}|' + '|'.join(ids)
    digest = hashlib.sha256(seed_text.encode()).digest()
    seed = int.from_bytes(digest[:8], 'little') % (2**32)
    signature = {'parameters': PARAMETERS, 'seed': seed, 'version': version,
                 'executable_sha256': sha256(executable), 'edge_input_sha256': sha256(PRIVATE / 'chacoconn_edges.npz'),
                 'subjects_sha256': sha256(PRIVATE / f'{slug}_subjects.csv')}
    # MRtrix records its full command, including local paths, in CSV comments.
    # Keep these unmodified raw files private; public-facing exports are below OUT.
    run = RUNS / slug
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['signature'] == signature and all((run / f).exists() and sha256(run / f) == h
                                                       for f, h in previous['output_hashes'].items()):
            print(f'Reused verified TFNBS: {name}', flush=True)
            return previous
        raise RuntimeError(f'Inputs or outputs changed for {slug}; archive the previous run before replacing it')
    index = {sid: k for k, sid in enumerate(source['subject_ids'])}
    raw = source['values'][[index[sid] for sid in ids]]
    ranked = rank_z(raw)
    design = np.column_stack([np.ones(len(ids)), rank_z(subjects[column]), rank_z(subjects[COVARIATES])]).astype(np.float32)
    assert np.linalg.matrix_rank(design) == design.shape[1] == 6
    contrast = np.array([[0, -1, 0, 0, 0, 0]], dtype=float)
    private_run = PRIVATE / slug
    private_run.mkdir(parents=True, exist_ok=True)
    np.savetxt(private_run / 'design.txt', design, fmt='%.7g')
    np.savetxt(private_run / 'contrast.txt', contrast, fmt='%.7g')
    i, j = source['roi_i'] - 1, source['roi_j'] - 1
    # Generated matrix files are temporary; raw private snapshots, design,
    # parameters, seeds and the original subject order are retained.
    with tempfile.TemporaryDirectory(prefix=slug + '_', dir=PRIVATE) as temporary:
        from pathlib import Path
        temporary = Path(temporary)
        paths = []
        for k, row in enumerate(ranked):
            matrix = np.zeros((191, 191), dtype=np.float32)
            matrix[i, j] = matrix[j, i] = row
            path = temporary / f'row_{k:03d}.csv'
            np.savetxt(path, matrix, delimiter=',', fmt='%.7g')
            paths.append(path.name)
        listing = temporary / 'input_connectomes.txt'
        listing.write_text('\n'.join(paths) + '\n')
        command = [str(executable), '-force', '-nshuffles', str(PARAMETERS['nshuffles']),
                   '-tfce_dh', str(PARAMETERS['tfce_dh']), '-tfce_e', str(PARAMETERS['tfce_e']),
                   '-tfce_h', str(PARAMETERS['tfce_h']), '-nthreads', str(PARAMETERS['nthreads']),
                   str(listing), 'tfnbs', str(private_run / 'design.txt'), str(private_run / 'contrast.txt'), str(run / 'tfnbs_')]
        environment = os.environ.copy()
        environment['MRTRIX_RNG_SEED'] = str(seed)
        print(f'Running TFNBS: {name}, n = {len(ids)}, 5,000 shuffles', flush=True)
        with (private_run / 'connectomestats.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=log, env=environment, cwd=temporary)
    result = {'battery': battery, 'subtest': name, 'n': len(ids), 'n_left': int(subjects.hemisphere_R.eq(0).sum()),
              'n_right': int(subjects.hemisphere_R.eq(1).sum()), 'n_battery_tests': n_tests,
              'signature': signature, 'elapsed_seconds': time.monotonic() - start,
              'output_hashes': {p.name: sha256(p) for p in sorted(run.glob('tfnbs_*')) if p.is_file()}}
    manifest_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Completed TFNBS: {name} ({result["elapsed_seconds"]:.1f} s)', flush=True)
    return result


def collect(source, runs):
    labels = pd.read_csv(ROOT / 'data/resources/network_definitions.csv').set_index('roi_index')
    frames, counts = [], []
    i, j = source['roi_i'] - 1, source['roi_j'] - 1
    for battery, name, column, slug, n_tests in SPECS:
        run = RUNS / slug
        load = lambda suffix: np.loadtxt(run / ('tfnbs_' + suffix + '.csv'), delimiter=',')[i, j]
        p = np.maximum(np.clip(1 - load('fwe_1mpvalue'), 0, 1), 1 / PARAMETERS['nshuffles'])
        t = load('tvalue')
        subjects = pd.read_csv(PRIVATE / f'{slug}_subjects.csv')
        index = {sid: k for k, sid in enumerate(source['subject_ids'])}
        raw = source['values'][[index[sid] for sid in subjects.subject_id]]
        y = rank_z(raw).astype(float)
        x = rank_z(subjects[column]).astype(float)
        c = np.column_stack([np.ones(len(x)), rank_z(subjects[COVARIATES])]).astype(float)
        xr = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
        yr = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
        denominator = np.sqrt(np.sum(xr*xr) * np.sum(yr*yr, axis=0))
        rho = np.divide(-(xr @ yr), denominator, out=np.zeros_like(denominator), where=denominator > 1e-10)
        df = len(x) - 6
        expected_t = rho * np.sqrt(df / np.maximum(1 - rho*rho, 1e-15))
        valid = denominator > 1e-10
        error = float(np.max(np.abs(t[valid] - expected_t[valid])))
        if error > 3e-5:
            raise ValueError(f'Adjusted t-statistic reproduction failed for {name}: {error}')
        names_i, names_j = labels.loc[i+1, 'roi_name'].to_numpy(), labels.loc[j+1, 'roi_name'].to_numpy()
        frame = pd.DataFrame({'model_family': 'hemisphere', 'battery': battery, 'subtest_label': name,
                              'roi_i': i+1, 'roi_j': j+1, 'roi_i_name': names_i, 'roi_j_name': names_j,
                              'roi_i_anatomical_group': [anatomical_group(v) for v in names_i],
                              'roi_j_anatomical_group': [anatomical_group(v) for v in names_j],
                              'partial_deficit_rho': rho, 't_value': t, 'tfnbs_enhanced': load('enhanced'),
                              'p_fwe_edges': p, 'p_battery_bonferroni': np.minimum(p*n_tests, 1),
                              'significant_fwe_edges': p < .05, 'significant_battery_bonferroni': p*n_tests < .05})
        frames.append(frame)
        counts.append({'battery': battery, 'subtest': name, 'n': len(x), 'edge_FWE': int(np.sum(p < .05)),
                       'across_subtests': int(np.sum(p*n_tests < .05)), 'max_abs_t_reproduction_error': error})
    pd.concat(frames, ignore_index=True).to_csv(OUT / 'edge_statistics.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(counts).to_csv(OUT / 'counts.csv', index=False)
    (ROOT / 'metadata/hemisphere_tfnbs.json').write_text(json.dumps({'models': runs, 'counts': counts,
        'parameters': PARAMETERS, 'new_model': True, 'interpretation': 'Exploratory clinical-plus-hemisphere model; original lobe model retained as sensitivity'}, indent=2) + '\n')
    print(pd.DataFrame(counts).to_string(index=False), flush=True)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-inputs', action='store_true')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_inputs:
        prepare_inputs()
    source = dict(np.load(PRIVATE / 'chacoconn_edges.npz'))
    executable = shutil.which(os.environ.get('CONNECTOMESTATS', 'connectomestats'))
    if not executable:
        raise RuntimeError('MRtrix3 connectomestats is required')
    executable = Path(executable)
    version_result = subprocess.run([str(executable), '-version'], capture_output=True, text=True, check=True)
    version = (version_result.stdout + version_result.stderr).splitlines()[0]
    if args.collect_only:
        runs = [json.loads((RUNS / slug / 'run_manifest.json').read_text()) for slug in SLUGS]
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, spec, source, executable, version) for spec in SPECS]
            runs = [future.result() for future in as_completed(futures)]
    collect(source, sorted(runs, key=lambda r: r['subtest']))


if __name__ == '__main__':
    main()
