from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _shared import load_nemo_edge_matrices, mirror_nemo_upper_triangle  # noqa: E402


class NemoGraphLoadingTests(unittest.TestCase):
    def test_upper_triangle_is_mirrored_without_halving(self) -> None:
        raw = np.array([[0.0, 4.0, 2.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]])
        observed = mirror_nemo_upper_triangle(raw, matrix_name="synthetic")
        expected = np.array([[0.0, 4.0, 2.0], [4.0, 0.0, 6.0], [2.0, 6.0, 0.0]])
        np.testing.assert_array_equal(observed, expected)

    def test_nontriangular_input_is_rejected(self) -> None:
        raw = np.array([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "upper-triangular"):
            mirror_nemo_upper_triangle(raw, matrix_name="synthetic")

    def test_loader_returns_unattenuated_remaining_sc(self) -> None:
        remaining_sc, chaco = load_nemo_edge_matrices()
        self.assertEqual(len(remaining_sc), 163)
        self.assertEqual(set(remaining_sc), set(chaco))
        sid = sorted(remaining_sc)[0]
        sub_dir = ROOT / "data/nemo/ifod2act_fs191" / f"sub-{sid}"
        sc_file = sorted(sub_dir.glob("*ifod2act*fs191subj*nemoSC*mean.mat"))[0]
        raw = loadmat(sc_file)["SC"].astype(float)
        upper = np.triu_indices_from(raw, k=1)
        np.testing.assert_allclose(remaining_sc[sid][upper], raw[upper], rtol=0, atol=0)
        self.assertTrue(np.allclose(remaining_sc[sid], remaining_sc[sid].T))
        self.assertTrue(np.allclose(chaco[sid], chaco[sid].T))


if __name__ == "__main__":
    unittest.main()
