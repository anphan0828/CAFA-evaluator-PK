import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from cafaeval.tests import test_bootstrap_metrics


def use_sparse():
    return os.environ.get("CAFAEVAL_SPARSE", "1") not in ("0", "false", "False")


def use_fast_parser():
    return os.environ.get("CAFAEVAL_FAST_PARSER", "1") not in ("0", "false", "False")


def csr_nonzeros(mat):
    """Return stored non-zero coordinates for a CSR-like sparse matrix."""
    mat = mat.tocsr()
    data = mat.data
    cols = mat.indices.astype(np.int64, copy=False)
    rows = np.repeat(np.arange(mat.shape[0], dtype=np.int64), np.diff(mat.indptr))
    if data.size and not data.all():
        keep = data != 0
        return rows[keep], cols[keep], data[keep]
    return rows, cols, data


def toi_is_full(toi, n_terms):
    return (
        isinstance(toi, np.ndarray)
        and toi.ndim == 1
        and toi.size == n_terms
        and bool(np.array_equal(toi, np.arange(n_terms)))
    )


def count_proteins_in_toi(gt_matrix, toi, exclude_matrix=None):
    """Count proteins with at least one evaluable GT annotation in TOI."""
    n_terms = gt_matrix.shape[1]
    full_toi = toi_is_full(toi, n_terms)

    if exclude_matrix is None:
        if full_toi:
            return int((gt_matrix != 0).any(axis=1).sum())
        return int((gt_matrix[:, toi] != 0).any(axis=1).sum())

    if full_toi:
        toi_mask = np.ones(n_terms, dtype=bool)
    else:
        toi_mask = np.zeros(n_terms, dtype=bool)
        toi_mask[toi] = True
    valid = (gt_matrix != 0) & toi_mask[None, :] & (exclude_matrix == 0)
    return int(valid.any(axis=1).sum())


def _bootstrap_from_arrays(tau_arr, n_pred_at_tau, tp_at_tau, n_gt, B_ind, eligible_rows=None):
    metrics_B_tau = {}
    if B_ind is None:
        return metrics_B_tau

    has_eligibility = eligible_rows is not None
    if has_eligibility:
        eligible_rows = np.asarray(eligible_rows, dtype=bool)

    test_indices = _bootstrap_test_indices(B_ind)
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
            fp_b = fp[ind]
            fn_b = fn[ind]
            n_gt_b = n_gt[ind]

            if has_eligibility:
                eligible_b = eligible_rows[ind]
                metrics_B[b, 0] = ((n_pred_b > 0) & eligible_b).sum()
                metrics_B[b, 6] = eligible_b.sum()
            else:
                metrics_B[b, 0] = (n_pred_b > 0).sum()
                metrics_B[b, 6] = len(n_pred_b)

            metrics_B[b, 1] = tp_b.sum()
            metrics_B[b, 2] = fp_b.sum()
            metrics_B[b, 3] = fn_b.sum()
            metrics_B[b, 4] = np.divide(
                tp_b, n_pred_b, out=np.zeros_like(tp_b, dtype="float"), where=n_pred_b > 0
            ).sum()
            metrics_B[b, 5] = np.divide(
                tp_b, n_gt_b, out=np.zeros_like(n_gt_b, dtype="float"), where=n_gt_b > 0
            ).sum()

            if b in test_indices:
                metrics_per_protein_b = pd.DataFrame({
                    "n_pred": n_pred_b,
                    "TP": tp_b,
                    "FP": fp_b,
                    "FN": fn_b,
                    "n_gt": n_gt_b,
                })
                if has_eligibility:
                    metrics_per_protein_b["eligible"] = eligible_b
                test_bootstrap_metrics(metrics_B[b], metrics_per_protein_b, has_eligibility)

        metrics_B_tau[tau] = metrics_B

    return metrics_B_tau


def _bootstrap_test_indices(B_ind):
    if len(B_ind) == 0:
        return set()
    n_tests = max(1, int(np.ceil(len(B_ind) * 0.10)))
    return set(np.linspace(0, len(B_ind) - 1, n_tests, dtype="int"))


def compute_confusion_matrix_sparse(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None, B_ind=None):
    """Sparse alternative to the dense NK/LK confusion-matrix sweep."""
    n_prot, _n_toi = pred_matrix.shape
    n_tau = len(tau_arr)
    metrics = np.zeros((n_tau, 6), dtype="float")

    w_toi = None if ic_arr is None else ic_arr[toi].astype(np.float64, copy=False)
    total_gt = float(n_gt.sum())

    if issparse(pred_matrix):
        nz_rows, nz_cols, nz_scores = csr_nonzeros(pred_matrix)
    else:
        nz_rows, nz_cols = np.nonzero(pred_matrix)
        nz_scores = pred_matrix[nz_rows, nz_cols]
    if nz_rows.size == 0:
        metrics[:, 3] = total_gt
        metrics_B_tau = _bootstrap_from_arrays(
            tau_arr, np.zeros((n_prot, n_tau)), np.zeros((n_prot, n_tau)), n_gt, B_ind
        )
        return metrics, metrics_B_tau

    nz_is_tp = np.asarray(g[nz_rows, nz_cols]).ravel().astype(np.float64, copy=False)
    last_idx = np.searchsorted(tau_arr, nz_scores, side="right") - 1
    if last_idx.min() < 0:
        keep = last_idx >= 0
        last_idx = last_idx[keep]
        nz_rows = nz_rows[keep]
        nz_cols = nz_cols[keep]
        nz_is_tp = nz_is_tp[keep]
        if last_idx.size == 0:
            metrics[:, 3] = total_gt
            metrics_B_tau = _bootstrap_from_arrays(
                tau_arr, np.zeros((n_prot, n_tau)), np.zeros((n_prot, n_tau)), n_gt, B_ind
            )
            return metrics, metrics_B_tau

    flat = nz_rows.astype(np.int64, copy=False) * n_tau + last_idx.astype(np.int64, copy=False)
    length = n_prot * n_tau

    w_pred = np.ones_like(nz_is_tp) if w_toi is None else w_toi[nz_cols]
    w_tp = w_pred * nz_is_tp
    delta_pred_w = np.bincount(flat, weights=w_pred, minlength=length).reshape(n_prot, n_tau)
    delta_tp_w = np.bincount(flat, weights=w_tp, minlength=length).reshape(n_prot, n_tau)

    pred_at_tau = np.cumsum(delta_pred_w[:, ::-1], axis=1)[:, ::-1]
    tp_at_tau = np.cumsum(delta_tp_w[:, ::-1], axis=1)[:, ::-1]

    _fill_metrics_from_threshold_arrays(metrics, pred_at_tau, tp_at_tau, n_gt, total_gt)
    metrics_B_tau = _bootstrap_from_arrays(tau_arr, pred_at_tau, tp_at_tau, n_gt, B_ind)
    return metrics, metrics_B_tau


def compute_confusion_matrix_exclude_sparse(
    tau_arr, pred_sub, gt_sub, toi_mask, excluded_mask, n_gt, ic_arr=None, B_ind=None
):
    """Sparse alternative to the dense PK confusion-matrix sweep."""
    n_prot, n_terms = pred_sub.shape
    n_tau = len(tau_arr)
    metrics = np.zeros((n_tau, 6), dtype="float")
    total_gt = float(n_gt.sum())

    if issparse(pred_sub):
        p_rows, p_cols, p_scores = csr_nonzeros(pred_sub)
    else:
        p_rows, p_cols = np.nonzero(pred_sub)
        p_scores = pred_sub[p_rows, p_cols]

    keep = toi_mask[p_cols] & (~excluded_mask[p_rows, p_cols])
    nz_rows = p_rows[keep]
    nz_cols = p_cols[keep]
    nz_scores = p_scores[keep]

    eligible_rows = ((gt_sub != 0) & toi_mask[None, :] & (~excluded_mask)).any(axis=1)

    if nz_rows.size == 0:
        metrics[:, 3] = total_gt
        zero = np.zeros((n_prot, n_tau))
        metrics_B_tau = _bootstrap_from_arrays(tau_arr, zero, zero, n_gt, B_ind, eligible_rows)
        return metrics, metrics_B_tau, eligible_rows

    nz_is_tp = np.asarray(gt_sub[nz_rows, nz_cols]).ravel().astype(np.float64, copy=False)
    last_idx = np.searchsorted(tau_arr, nz_scores, side="right") - 1
    if last_idx.min() < 0:
        keep = last_idx >= 0
        last_idx = last_idx[keep]
        nz_rows = nz_rows[keep]
        nz_cols = nz_cols[keep]
        nz_is_tp = nz_is_tp[keep]
        if last_idx.size == 0:
            metrics[:, 3] = total_gt
            zero = np.zeros((n_prot, n_tau))
            metrics_B_tau = _bootstrap_from_arrays(tau_arr, zero, zero, n_gt, B_ind, eligible_rows)
            return metrics, metrics_B_tau, eligible_rows

    flat = nz_rows.astype(np.int64, copy=False) * n_tau + last_idx.astype(np.int64, copy=False)
    length = n_prot * n_tau

    w_pred = np.ones_like(nz_is_tp) if ic_arr is None else ic_arr[nz_cols].astype(np.float64, copy=False)
    w_tp = w_pred * nz_is_tp

    delta_pred_w = np.bincount(flat, weights=w_pred, minlength=length).reshape(n_prot, n_tau)
    delta_tp_w = np.bincount(flat, weights=w_tp, minlength=length).reshape(n_prot, n_tau)
    pred_at_tau = np.cumsum(delta_pred_w[:, ::-1], axis=1)[:, ::-1]
    tp_at_tau = np.cumsum(delta_tp_w[:, ::-1], axis=1)[:, ::-1]

    _fill_metrics_from_threshold_arrays(metrics, pred_at_tau, tp_at_tau, n_gt, total_gt, eligible_rows)
    metrics_B_tau = _bootstrap_from_arrays(tau_arr, pred_at_tau, tp_at_tau, n_gt, B_ind, eligible_rows)
    return metrics, metrics_B_tau, eligible_rows


def _fill_metrics_from_threshold_arrays(metrics, pred_at_tau, tp_at_tau, n_gt, total_gt, eligible_rows=None):
    tp_totals = tp_at_tau.sum(axis=0)
    pred_totals = pred_at_tau.sum(axis=0)

    if eligible_rows is None:
        metrics[:, 0] = (pred_at_tau > 0).sum(axis=0)
    else:
        metrics[:, 0] = ((pred_at_tau > 0) & eligible_rows[:, None]).sum(axis=0)
    metrics[:, 1] = tp_totals
    metrics[:, 2] = pred_totals - tp_totals
    metrics[:, 3] = total_gt - tp_totals

    safe_pred = np.where(pred_at_tau > 0, pred_at_tau, 1.0)
    precision = np.where(pred_at_tau > 0, tp_at_tau / safe_pred, 0.0)
    metrics[:, 4] = precision.sum(axis=0)

    if np.any(n_gt > 0):
        n_gt_col = n_gt.astype(np.float64, copy=False)[:, None]
        safe_gt = np.where(n_gt_col > 0, n_gt_col, 1.0)
        recall = np.where(n_gt_col > 0, tp_at_tau / safe_gt, 0.0)
        metrics[:, 5] = recall.sum(axis=0)


def children_cache(ont):
    children_by_term = getattr(ont, "_children_by_term", None)
    if children_by_term is None:
        indptr, idx = ont._chi_indptr, ont._chi_idx
        children_by_term = [
            idx[indptr[term_id]:indptr[term_id + 1]]
            for term_id in range(len(indptr) - 1)
        ]
        ont._children_by_term = children_by_term
    return children_by_term


def propagate_serial(matrix, order_, children_by_term, mode):
    for term_id in order_:
        children = children_by_term[term_id]
        if children.size == 0:
            continue
        if mode == "max":
            child_max = matrix[:, children].max(axis=1)
            matrix[:, term_id] = np.maximum(matrix[:, term_id], child_max)
        elif mode == "fill":
            rows = np.flatnonzero(matrix[:, term_id] == 0)
            if rows.size:
                idx = np.ix_(rows, children)
                matrix[rows, term_id] = matrix[idx].max(axis=1)


def ancestors_csr(ont):
    cached = getattr(ont, "_ancestors_csr", None)
    if cached is not None:
        return cached

    n = int(ont.idxs)
    p_indptr = ont._par_indptr
    sorted_cols = ont._par_idx
    top_down = np.asarray(ont.order)[::-1]

    ancestors = [None] * n
    for t in top_down:
        t_int = int(t)
        s = {t_int}
        parents_slice = sorted_cols[p_indptr[t_int]:p_indptr[t_int + 1]]
        for p in parents_slice:
            s.update(ancestors[int(p)])
        ancestors[t_int] = s

    lens = np.fromiter((len(ancestors[t]) for t in range(n)), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for t in range(n):
        a = ancestors[t]
        if a:
            indices[indptr[t]:indptr[t + 1]] = np.fromiter(a, dtype=np.int64, count=len(a))

    ont._ancestors_csr = (indptr, indices)
    return indptr, indices


def propagate_sparse_fill(matrix, ont, triples=None):
    n_prot, n_terms = matrix.shape
    if n_prot == 0 or n_terms == 0:
        return

    if triples is not None:
        nz_rows, nz_cols, nz_scores = triples
        nz_rows = np.asarray(nz_rows, dtype=np.int64)
        nz_cols = np.asarray(nz_cols, dtype=np.int64)
        nz_scores = np.asarray(nz_scores)
    else:
        nz_rows, nz_cols = np.nonzero(matrix)
        if nz_rows.size == 0:
            return
        nz_scores = matrix[nz_rows, nz_cols]

    if nz_rows.size == 0:
        return

    chi_indptr = ont._chi_indptr
    chi_idx = ont._chi_idx
    current = defaultdict(dict)
    orig_rows = defaultdict(set)
    for r, c, s in zip(nz_rows.tolist(), nz_cols.tolist(), nz_scores.tolist()):
        current[c][r] = s
        orig_rows[c].add(r)

    for t in ont.order:
        c0 = chi_indptr[t]
        c1 = chi_indptr[t + 1]
        if c1 == c0:
            continue
        row_max = {}
        for child in chi_idx[c0:c1]:
            child_vals = current.get(int(child))
            if not child_vals:
                continue
            for r, v in child_vals.items():
                if v > row_max.get(r, -np.inf):
                    row_max[r] = v
        if not row_max:
            continue
        blocked = orig_rows.get(t, ())
        cur_t = current[t]
        for r, v in row_max.items():
            if r in blocked:
                continue
            cur_t[r] = v
            matrix[r, t] = v


def propagate_sparse_pushup(matrix, ont, mode, triples=None):
    if mode == "fill":
        propagate_sparse_fill(matrix, ont, triples=triples)
        return

    n_prot, n_terms = matrix.shape
    if n_prot == 0 or n_terms == 0:
        return

    if triples is not None:
        nz_rows, nz_cols, nz_scores = triples
    else:
        nz_rows, nz_cols = np.nonzero(matrix)
        if nz_rows.size == 0:
            return
        nz_scores = matrix[nz_rows, nz_cols]
    if nz_rows.size == 0:
        return

    out_rows, out_cols, out_scores = propagate_to_coo((nz_rows, nz_cols, nz_scores), ont, mode="max")
    current = matrix[out_rows, out_cols]
    np.maximum(current, out_scores, out=current)
    matrix[out_rows, out_cols] = current


def propagate_to_coo(triples, ont, mode="max"):
    if mode != "max":
        raise ValueError("propagate_to_coo only supports mode='max'")
    rows, cols, scores = triples
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    scores = np.asarray(scores)
    if rows.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=scores.dtype)

    indptr, anc_indices = ancestors_csr(ont)
    n_terms = int(ont.idxs)
    n_anc = (indptr[cols + 1] - indptr[cols]).astype(np.int64)
    total = int(n_anc.sum())
    if total == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=scores.dtype)

    block_starts = indptr[cols]
    global_offsets = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(n_anc, out=global_offsets[1:])
    base_per_j = np.repeat(block_starts, n_anc)
    local_offset = np.arange(total, dtype=np.int64) - np.repeat(global_offsets[:-1], n_anc)
    expanded_cols = anc_indices[base_per_j + local_offset]
    expanded_rows = np.repeat(rows, n_anc)
    expanded_scores = np.repeat(scores, n_anc)

    flat = expanded_rows.astype(np.int64) * n_terms + expanded_cols
    order_idx = np.argsort(flat, kind="stable")
    flat_s = flat[order_idx]
    scores_s = expanded_scores[order_idx]
    group_starts = np.empty(flat_s.size, dtype=bool)
    group_starts[0] = True
    np.not_equal(flat_s[1:], flat_s[:-1], out=group_starts[1:])
    start_idx = np.flatnonzero(group_starts)
    group_max = np.maximum.reduceat(scores_s, start_idx)
    unique_flat = flat_s[start_idx]
    out_rows = unique_flat // n_terms
    out_cols = unique_flat % n_terms
    return out_rows, out_cols, group_max
