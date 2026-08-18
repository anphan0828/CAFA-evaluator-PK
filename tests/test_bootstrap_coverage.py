import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cafaeval.evaluation import (
    bootstrap,
    compute_metrics,
    compute_confusion_matrix_exclude,
    evaluate_prediction,
    get_bootstrap_test_indices,
    get_metrics_B,
    make_paired_bootstrap_indices as evaluation_make_paired_bootstrap_indices,
    normalize,
    use_paired_bootstrap,
)
from cafaeval.evaluation_bootstrap import make_paired_bootstrap_indices
from cafaeval.sparse import _bootstrap_count_matrix, _bootstrap_from_arrays
from cafaeval.tests import test_bootstrap_metrics as _test_bootstrap_metrics


class BootstrapCoverageTests(unittest.TestCase):

    def _loop_bootstrap_from_arrays(self, tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind, eligible_rows=None):
        metrics_B_tau = {}
        has_eligibility = eligible_rows is not None
        if has_eligibility:
            eligible_rows = np.asarray(eligible_rows, dtype=bool)
        n_gt = np.asarray(n_gt, dtype="float")

        for t_idx, tau in enumerate(tau_arr):
            n_pred = n_pred_at_tau[:, t_idx]
            tp = tp_at_tau[:, t_idx]
            fp = n_pred - tp
            fn = n_gt - tp
            metrics_B = np.zeros((len(B_ind), 7), dtype="float")
            for b, ind in enumerate(B_ind):
                ind = np.asarray(ind, dtype=np.int64)
                n_pred_b = n_pred[ind]
                tp_b = tp[ind]
                n_gt_b = n_gt[ind]
                if has_eligibility:
                    eligible_b = eligible_rows[ind]
                    metrics_B[b, 0] = ((n_pred_b > 0) & eligible_b).sum()
                    metrics_B[b, 6] = eligible_b.sum()
                else:
                    metrics_B[b, 0] = (n_pred_b > 0).sum()
                    metrics_B[b, 6] = len(n_pred_b)
                metrics_B[b, 1] = tp_b.sum()
                metrics_B[b, 2] = fp[ind].sum()
                metrics_B[b, 3] = fn[ind].sum()
                metrics_B[b, 4] = np.divide(
                    tp_b, n_pred_b, out=np.zeros_like(tp_b, dtype="float"), where=n_pred_b > 0
                ).sum()
                metrics_B[b, 5] = np.divide(
                    tp_b, n_gt_b, out=np.zeros_like(n_gt_b, dtype="float"), where=n_gt_b > 0
                ).sum()
            metrics_B_tau[tau] = metrics_B
        return metrics_B_tau

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

    def test_bootstrap_count_matrix_counts_duplicates(self):
        count_matrix = _bootstrap_count_matrix([[0, 1, 1, 3], [2, 2, 2]], 4)

        self.assertEqual(count_matrix.shape, (2, 4))
        np.testing.assert_allclose(
            count_matrix.toarray(),
            np.array([
                [1.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 3.0, 0.0],
            ]),
        )
        np.testing.assert_allclose(np.asarray(count_matrix.sum(axis=1)).ravel(), np.array([4.0, 3.0]))

    def test_vectorized_bootstrap_matches_loop_without_eligibility(self):
        tau_arr = np.array([0.25, 0.5, 0.75])
        n_pred_at_tau = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        tp_at_tau = np.array([
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ])
        n_gt = np.array([2.0, 1.0, 0.0])
        B_ind = [[0, 1, 2], [0, 0, 1]]

        expected = self._loop_bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind)
        actual = _bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind)

        self.assertEqual(set(actual), set(expected))
        for tau in tau_arr:
            np.testing.assert_allclose(actual[tau], expected[tau])

    def test_vectorized_bootstrap_matches_loop_with_pk_eligibility(self):
        tau_arr = np.array([0.1, 0.4])
        n_pred_at_tau = np.array([
            [2.0, 1.0],
            [1.0, 1.0],
            [3.0, 0.0],
        ])
        tp_at_tau = np.array([
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ])
        n_gt = np.array([1.0, 0.0, 2.0])
        eligible_rows = np.array([True, False, True])
        B_ind = [[0, 1, 2], [1, 1, 2]]

        expected = self._loop_bootstrap_from_arrays(
            tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind, eligible_rows
        )
        actual = _bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind, eligible_rows)

        for tau in tau_arr:
            np.testing.assert_allclose(actual[tau], expected[tau])

    def test_vectorized_bootstrap_matches_loop_with_ia_weights(self):
        tau_arr = np.array([0.2, 0.6])
        n_pred_at_tau = np.array([
            [4.5, 2.5],
            [1.25, 0.0],
            [0.0, 0.0],
        ])
        tp_at_tau = np.array([
            [2.5, 2.5],
            [0.25, 0.0],
            [0.0, 0.0],
        ])
        n_gt = np.array([3.5, 0.25, 0.0])
        B_ind = [[0, 1, 2], [0, 0]]

        expected = self._loop_bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind)
        actual = _bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind)

        for tau in tau_arr:
            np.testing.assert_allclose(actual[tau], expected[tau])

    def test_sparse_bootstrap_large_threshold_grid_smoke(self):
        old_chunk_size = os.environ.get("CAFAEVAL_BOOTSTRAP_CHUNK_SIZE")
        try:
            os.environ["CAFAEVAL_BOOTSTRAP_CHUNK_SIZE"] = "7"
            tau_arr = np.linspace(0.01, 0.99, 99)
            n_pred_at_tau = np.vstack([
                np.linspace(5.0, 0.0, 99),
                np.linspace(2.0, 0.0, 99),
                np.zeros(99),
            ])
            tp_at_tau = np.minimum(n_pred_at_tau, np.array([[3.0], [1.0], [0.0]]))
            n_gt = np.array([3.0, 1.0, 0.0])
            B_ind = [[0, 1, 2], [0, 0, 1]]

            actual = _bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind)
        finally:
            if old_chunk_size is None:
                os.environ.pop("CAFAEVAL_BOOTSTRAP_CHUNK_SIZE", None)
            else:
                os.environ["CAFAEVAL_BOOTSTRAP_CHUNK_SIZE"] = old_chunk_size

        self.assertEqual(len(actual), len(tau_arr))
        for tau in tau_arr:
            self.assertEqual(actual[tau].shape, (2, 7))

    def test_bootstrap_runtime_validation_env_gate_defaults_off(self):
        old_checks = os.environ.get("CAFAEVAL_BOOTSTRAP_CHECKS")
        try:
            os.environ.pop("CAFAEVAL_BOOTSTRAP_CHECKS", None)
            with patch("cafaeval.sparse.test_bootstrap_metrics") as test_func:
                _bootstrap_from_arrays(
                    np.array([0.5]),
                    np.array([[1.0], [0.0]]),
                    np.array([[1.0], [0.0]]),
                    np.array([1.0, 0.0]),
                    [[0, 1]],
                )
        finally:
            if old_checks is None:
                os.environ.pop("CAFAEVAL_BOOTSTRAP_CHECKS", None)
            else:
                os.environ["CAFAEVAL_BOOTSTRAP_CHECKS"] = old_checks

        test_func.assert_not_called()

    def test_bootstrap_runtime_validation_env_gate_runs_checks_when_enabled(self):
        old_checks = os.environ.get("CAFAEVAL_BOOTSTRAP_CHECKS")
        try:
            os.environ["CAFAEVAL_BOOTSTRAP_CHECKS"] = "1"
            with patch("cafaeval.sparse.test_bootstrap_metrics") as test_func:
                _bootstrap_from_arrays(
                    np.array([0.5]),
                    np.array([[1.0], [0.0]]),
                    np.array([[1.0], [0.0]]),
                    np.array([1.0, 0.0]),
                    [[0, 1]],
                )
        finally:
            if old_checks is None:
                os.environ.pop("CAFAEVAL_BOOTSTRAP_CHECKS", None)
            else:
                os.environ["CAFAEVAL_BOOTSTRAP_CHECKS"] = old_checks

        test_func.assert_called()

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

    def test_paired_bootstrap_rebuilds_count_matrix_from_shared_indices(self):
        gt = {
            "biological_process": SimpleNamespace(
                matrix=np.array([
                    [1, 0],
                    [0, 1],
                ], dtype=bool)
            )
        }
        ontologies = {
            "biological_process": SimpleNamespace(
                toi=np.array([0, 1]),
                toi_ia=np.array([], dtype=int),
                ia=None,
            )
        }
        B_ind = [[0, 1], [1, 1]]
        bootstrap_indices = {"biological_process": {"unweighted": B_ind, "weighted": []}}
        prediction_a = {"biological_process": SimpleNamespace(matrix=np.array([[0.8, 0.0], [0.0, 0.7]]))}
        prediction_b = {"biological_process": SimpleNamespace(matrix=np.array([[0.6, 0.0], [0.0, 0.9]]))}
        metric_df = pd.DataFrame([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]], columns=["n", "tp", "fp", "fn", "pr", "rc"])

        with patch(
            "cafaeval.evaluation.compute_metrics",
            side_effect=[(metric_df.copy(), {}), (metric_df.copy(), {})],
        ) as compute_func:
            evaluate_prediction(
                prediction_a,
                gt,
                ontologies,
                np.array([0.5]),
                B=2,
                B_pct=100,
                bootstrap_indices=bootstrap_indices,
            )
            evaluate_prediction(
                prediction_b,
                gt,
                ontologies,
                np.array([0.5]),
                B=2,
                B_pct=100,
                bootstrap_indices=bootstrap_indices,
            )

        self.assertIs(compute_func.call_args_list[0].kwargs["B_ind"], B_ind)
        self.assertIs(compute_func.call_args_list[1].kwargs["B_ind"], B_ind)
        np.testing.assert_allclose(
            _bootstrap_count_matrix(B_ind, 2).toarray(),
            np.array([[1.0, 1.0], [0.0, 2.0]]),
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
