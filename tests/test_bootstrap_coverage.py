import unittest
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cafaeval.evaluation import (
    bootstrap,
    compute_metrics,
    compute_confusion_matrix_exclude,
    get_bootstrap_test_indices,
    get_metrics_B,
    make_paired_bootstrap_indices as evaluation_make_paired_bootstrap_indices,
    normalize,
    use_paired_bootstrap,
)
from cafaeval.evaluation_bootstrap import make_paired_bootstrap_indices
from cafaeval.tests import test_bootstrap_metrics as _test_bootstrap_metrics


class BootstrapCoverageTests(unittest.TestCase):

    def test_paired_bootstrap_env_gate_defaults_on(self):
        old_paired = os.environ.get("PAIRED_BOOTSTRAP")
        try:
            os.environ.pop("PAIRED_BOOTSTRAP", None)
            self.assertTrue(use_paired_bootstrap())
            os.environ["PAIRED_BOOTSTRAP"] = "0"
            self.assertFalse(use_paired_bootstrap())
            os.environ["PAIRED_BOOTSTRAP"] = "false"
            self.assertFalse(use_paired_bootstrap())
            os.environ["PAIRED_BOOTSTRAP"] = "1"
            self.assertTrue(use_paired_bootstrap())
        finally:
            if old_paired is None:
                os.environ.pop("PAIRED_BOOTSTRAP", None)
            else:
                os.environ["PAIRED_BOOTSTRAP"] = old_paired

    def test_evaluation_bootstrap_reexports_optimized_paired_helper(self):
        self.assertIs(make_paired_bootstrap_indices, evaluation_make_paired_bootstrap_indices)

    def test_bootstrap_runtime_checks_use_ten_percent_of_indices(self):
        B_ind = [[i] for i in range(20)]

        self.assertEqual(get_bootstrap_test_indices([]), set())
        self.assertEqual(get_bootstrap_test_indices(B_ind), {0, 19})
        self.assertEqual(len(get_bootstrap_test_indices([[i] for i in range(101)])), 11)

    def test_paired_bootstrap_indices_are_created_once_per_namespace_mode(self):
        gt = {
            "biological_process": SimpleNamespace(
                matrix=np.array([
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ], dtype=bool)
            )
        }
        ontologies = {
            "biological_process": SimpleNamespace(
                toi=np.array([0, 1]),
                toi_ia=np.array([1, 2]),
                ia=np.array([0.0, 1.0, 1.0]),
            )
        }

        paired = make_paired_bootstrap_indices(gt, ontologies, B=5, B_pct=100)

        self.assertEqual(set(paired["biological_process"].keys()), {"unweighted", "weighted"})
        self.assertEqual(len(paired["biological_process"]["unweighted"]), 5)
        self.assertEqual(len(paired["biological_process"]["weighted"]), 5)
        # The weighted and unweighted row spaces both have two rows here, but
        # they refer to different proteins. The paired evaluator must not reuse
        # integer row positions unless the protein row identities match.
        self.assertIsNot(
            paired["biological_process"]["unweighted"],
            paired["biological_process"]["weighted"],
        )

    def test_bootstrap_without_eligibility_keeps_existing_columns_and_n(self):
        metrics_per_protein = pd.DataFrame({
            "n_pred": [1.0, 1.0, 2.0],
            "TP": [1.0, 0.0, 1.0],
            "FP": [0.0, 1.0, 1.0],
            "FN": [0.0, 0.0, 1.0],
            "n_gt": [1.0, 0.0, 2.0],
        })

        metrics_b = bootstrap(metrics_per_protein, [[0, 1, 2]])

        self.assertEqual(metrics_b.shape, (1, 7))
        np.testing.assert_allclose(
            metrics_b[0],
            np.array([3.0, 2.0, 2.0, 1.0, 1.5, 1.5, 3.0]),
        )

    def test_bootstrap_with_eligibility_changes_only_coverage_fields(self):
        metrics_per_protein = pd.DataFrame({
            "n_pred": [2.0, 1.0, 0.0],
            "TP": [1.0, 0.0, 0.0],
            "FP": [1.0, 1.0, 0.0],
            "FN": [0.0, 0.0, 1.0],
            "n_gt": [1.0, 0.0, 1.0],
            "eligible": [True, False, True],
        })

        metrics_b = bootstrap(metrics_per_protein, [[0, 1, 2], [0, 0]])

        self.assertEqual(metrics_b.shape, (2, 7))
        # Row 1 samples one predicted eligible row, one predicted ineligible
        # row, and one eligible row with no prediction. Only n and n_eval differ
        # from the old all-sampled-row coverage count; TP/FP/FN/pr/rc are the
        # same sums over the sampled proteins.
        np.testing.assert_allclose(
            metrics_b[0],
            np.array([1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0]),
        )
        # Duplicated eligible rows increase both n and the coverage denominator,
        # keeping bootstrap coverage bounded by one after normalization.
        np.testing.assert_allclose(
            metrics_b[1],
            np.array([2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0]),
        )
        _test_bootstrap_metrics(metrics_b[0], metrics_per_protein.iloc[[0, 1, 2]], True)
        _test_bootstrap_metrics(metrics_b[1], metrics_per_protein.iloc[[0, 0]], True)

    def test_normalize_uses_bootstrap_denominator_for_gt_normalized_metrics(self):
        metrics = pd.DataFrame(
            [[2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0]],
            columns=["n", "tp", "fp", "fn", "pr", "rc", "n_eval"],
        )

        normalized = normalize(metrics, "biological_process", np.array([0.5]), np.array([1.0]), "cafa")

        self.assertNotIn("n_eval", normalized.columns)
        self.assertEqual(normalized.loc[0, "cov"], 1.0)
        self.assertEqual(normalized.loc[0, "tp"], 1.0)
        self.assertEqual(normalized.loc[0, "fp"], 1.0)
        self.assertEqual(normalized.loc[0, "fn"], 0.0)
        self.assertEqual(normalized.loc[0, "pr"], 0.5)
        self.assertEqual(normalized.loc[0, "rc"], 1.0)

    def test_partial_knowledge_bootstrap_carries_eligibility_mask(self):
        tau_arr = np.array([0.5])
        g_perprotein = [np.array([True]), np.array([False])]
        pred_matrix = np.array([[0.9], [0.9]])
        toi_perprotein = [np.array([0]), np.array([0])]
        n_gt = np.array([6.13, 0.0])
        eligible_rows = np.array([True, False])
        ic_arr = np.array([6.13])

        metrics, metrics_b_tau = compute_confusion_matrix_exclude(
            tau_arr,
            g_perprotein,
            pred_matrix,
            toi_perprotein,
            n_gt,
            eligible_rows,
            ic_arr,
            B_ind=[[0, 1]],
        )

        np.testing.assert_allclose(metrics[0], np.array([1.0, 6.13, 6.13, 0.0, 1.0, 1.0]))
        np.testing.assert_allclose(
            metrics_b_tau[0.5][0],
            np.array([1.0, 6.13, 6.13, 0.0, 1.0, 1.0, 1.0]),
        )

        metrics_b = get_metrics_B(metrics_b_tau)[0]
        normalized_b = normalize(metrics_b, "biological_process", tau_arr, np.array([1.0]), "cafa")
        self.assertEqual(normalized_b.loc[0, "cov"], 1.0)

    def test_sparse_and_dense_partial_knowledge_bootstrap_match(self):
        old_sparse = os.environ.get("CAFAEVAL_SPARSE")
        gt_matrix = np.array([
            [True, False, True, False],
            [True, True, False, False],
            [False, False, False, True],
        ])
        exclude_matrix = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
        ])
        pred = np.array([
            [0.9, 0.0, 0.8, 0.4],
            [0.2, 0.7, 0.9, 0.3],
            [0.0, 0.0, 0.5, 0.95],
        ])
        exclude = SimpleNamespace(matrix=exclude_matrix)
        tau_arr = np.array([0.25, 0.75])
        toi = np.arange(gt_matrix.shape[1])
        B_ind = [[0, 1, 2], [0, 0, 1]]

        try:
            os.environ["CAFAEVAL_SPARSE"] = "1"
            metrics_sparse, boot_sparse = compute_metrics(
                csr_matrix(pred), gt_matrix, tau_arr, toi, exclude, None, n_cpu=1, B_ind=B_ind
            )
            os.environ["CAFAEVAL_SPARSE"] = "0"
            metrics_dense, boot_dense = compute_metrics(
                csr_matrix(pred), gt_matrix, tau_arr, toi, exclude, None, n_cpu=1, B_ind=B_ind
            )
        finally:
            if old_sparse is None:
                os.environ.pop("CAFAEVAL_SPARSE", None)
            else:
                os.environ["CAFAEVAL_SPARSE"] = old_sparse

        pd.testing.assert_frame_equal(metrics_sparse, metrics_dense)
        self.assertEqual(set(boot_sparse), set(boot_dense))
        for b in boot_sparse:
            pd.testing.assert_frame_equal(boot_sparse[b], boot_dense[b])


if __name__ == "__main__":
    unittest.main()
