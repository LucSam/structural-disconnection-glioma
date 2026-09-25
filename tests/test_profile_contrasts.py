from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import linalg
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import (
    holm, orthogonal_design, permutation_test, pillai_from_residuals,
    profile_contrasts, rank_normalise,
)


class ProfileContrastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(808)
        self.c = rng.normal(size=(80, 3))
        self.x = rng.normal(size=(80, 4)) + self.c[:, [0]]
        self.y = self.c @ rng.normal(size=(3, 4)) + self.x @ rng.normal(size=(4, 4)) + rng.normal(size=(80, 4))

    def test_helmert_removes_common_level_and_has_full_contrast_rank(self):
        z, h = profile_contrasts(self.y)
        np.testing.assert_allclose(h @ h.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(h @ np.ones(4), 0, atol=1e-14)
        shared = self.x[:, [0]] * 13 + self.c[:, [1]] * 7
        np.testing.assert_allclose(profile_contrasts(self.y + shared)[0], z, atol=1e-12)

    def test_change_of_contrast_basis_leaves_test_unchanged(self):
        z, _ = profile_contrasts(self.y)
        # Three reference differences span the same space as Helmert contrasts.
        reference = self.y[:, [0]] - self.y[:, 1:]
        a = permutation_test(z, self.x, self.c, 199, np.random.default_rng(11))
        b = permutation_test(reference, self.x, self.c, 199, np.random.default_rng(11))
        self.assertAlmostEqual(a["pillai"], b["pillai"], places=12)
        self.assertEqual(a["p_permutation"], b["p_permutation"])

    def test_statistic_matches_statsmodels_general_linear_hypothesis(self):
        z, h = profile_contrasts(self.y)
        design = np.column_stack([np.ones(len(self.y)), self.c, self.x])
        left_contrast = np.column_stack([np.zeros((4, 4)), np.eye(4)])
        model = MANOVA(self.y, design)
        expected = model.mv_test([("profile", left_contrast, h.T)]).results["profile"]["stat"].loc["Pillai's trace", "Value"]
        qc, qx = orthogonal_design(self.x, self.c)
        observed = pillai_from_residuals(z - qc @ (qc.T @ z), qx)
        self.assertAlmostEqual(observed, expected, places=12)

    def test_permutations_match_explicit_reduced_fit_reference(self):
        z, _ = profile_contrasts(self.y)
        cov = np.column_stack([np.ones(len(z)), self.c])
        xr = self.x - cov @ np.linalg.lstsq(cov, self.x, rcond=None)[0]
        fitted = cov @ np.linalg.lstsq(cov, z, rcond=None)[0]
        residual = z - fitted

        def statistic(y):
            y = y - cov @ np.linalg.lstsq(cov, y, rcond=None)[0]
            yh = xr @ np.linalg.lstsq(xr, y, rcond=None)[0]
            e = y - yh
            h = yh.T @ yh
            return float(np.trace(h @ np.linalg.pinv(h + e.T @ e)))

        observed = statistic(z)
        rng = np.random.default_rng(334)
        exceed = sum(statistic(fitted + residual[rng.permutation(len(z))]) >= observed for _ in range(299))
        actual = permutation_test(z, self.x, self.c, 299, np.random.default_rng(334))
        self.assertAlmostEqual(observed, actual["pillai"], places=12)
        self.assertEqual(exceed, actual["n_exceedances"])
        self.assertEqual((exceed + 1) / 300, actual["p_permutation"])

    def test_known_differential_signal_is_detected(self):
        rng = np.random.default_rng(24)
        y = np.outer(self.x[:, 0], [4, -4, 0, 0]) + rng.normal(scale=0.3, size=(80, 4))
        z, _ = profile_contrasts(y)
        actual = permutation_test(z, self.x, self.c, 199, np.random.default_rng(445))
        self.assertLessEqual(actual["p_permutation"], 0.01)

    def test_contrasts_preserve_selected_scale_without_second_transform(self):
        ranked = rank_normalise(np.round(self.y))
        contrasted, h = profile_contrasts(ranked)
        np.testing.assert_allclose(contrasted, ranked @ h.T)
        self.assertFalse(np.allclose(contrasted, rank_normalise(contrasted)))

    def test_rank_deficient_structural_predictors_are_rejected(self):
        duplicated = np.column_stack([self.x, self.x[:, 0]])
        with self.assertRaisesRegex(ValueError, "structural block is rank deficient"):
            permutation_test(self.y, duplicated, self.c, 99, np.random.default_rng(0))

    def test_holm_matches_independent_implementation(self):
        p = np.array([0.25, 0.001, 0.05, 0.12, 0.01])
        np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])


if __name__ == "__main__":
    unittest.main()
