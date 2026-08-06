import os
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.sparse import issparse
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.sparse import (
    compute_confusion_matrix_exclude_sparse,
    compute_confusion_matrix_sparse,
    count_proteins_in_toi,
    toi_is_full,
    use_sparse,
)
from cafaeval.tests import test_norm_metric, test_intersection, test_bootstrap_metrics
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def warn_empty_post_exclusion_gt(n_gt, ic_arr=None):
    n_empty = int(np.count_nonzero(np.asarray(n_gt) == 0))
    if n_empty == 0:
        return
    metric_mode = "IA-weighted" if ic_arr is not None else "unweighted"
    positive_ia_note = " positive-IA" if ic_arr is not None else ""
    logging.warning(
        "%s partial-knowledge evaluation: %d proteins had ground-truth "
        "annotations in the terms of interest before known-annotation "
        "exclusion, but no evaluable%s ground-truth annotations remained.",
        metric_mode,
        n_empty,
        positive_ia_note,
    )


def use_paired_bootstrap():
    return os.environ.get("PAIRED_BOOTSTRAP", "1") not in ("0", "false", "False")


def _proteins_with_gt(gt_matrix, toi):
    return np.where((gt_matrix[:, toi] != 0).any(axis=1))[0]


def compute_confusion_matrix(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None, B_ind = None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')
    metrics_B_tau = {}
    metrics_B = []

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p = solidify_prediction(pred_matrix, tau)

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP
        mis = np.logical_and(p, np.logical_not(g))  # FP, predicted but not in the ground truth
        remaining = np.logical_and(np.logical_not(p), g)  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p = p * ic_arr[toi]
            intersection = intersection * ic_arr[toi]  # TP
            mis = mis * ic_arr[toi]  # FP, predicted but not in the ground truth
            remaining = remaining * ic_arr[toi]  # FN, not predicted but in the ground truth

        n_pred = p.sum(axis=1)  # TP + FP (number of terms predicted in each protein)
        n_intersection = intersection.sum(axis=1)  # TP (number of TP terms per protein)
        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP (total terms)
        metrics[i, 2] = mis.sum(axis=1).sum()  # FP
        metrics[i, 3] = remaining.sum(axis=1).sum()  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0).sum()  # Precision
        metrics[i, 5] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum()  # Recall

        metrics_per_protein = pd.DataFrame({'n_pred': n_pred, 'TP': n_intersection, 'FP': mis.sum(axis=1), 'FN': remaining.sum(axis=1), 'n_gt': n_gt})
        if B_ind is not None:
            metrics_B_tau[tau] = bootstrap(metrics_per_protein, B_ind)
    #if B_ind is not None:
    #    metrics_B = get_metrics_B(metrics_B_tau)
    return metrics, metrics_B_tau


def compute_confusion_matrix_exclude(tau_arr, g_perprotein, pred_matrix, toi_perprotein, n_gt, eligible_rows, ic_arr=None, B_ind = None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is

    Here, g is the full ground truth matrix without filtering terms of interest (toi).
    Instead,
    eligible_rows marks proteins that still have at least one evaluable GT
    annotation after the per-protein exclude set is removed. Coverage must
    count predictions only for this same post-exclusion population.
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')
    metrics_B_tau = {}
    metrics_B = []

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p_perprotein = [solidify_prediction(pred_matrix[p_idx, tois], tau) for p_idx, tois in enumerate(toi_perprotein)]

        # Terms subsets
        intersection = [np.logical_and(p_i, g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # TP
        mis = [np.logical_and(p_i, np.logical_not(g_i)) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FP, predicted but not in the ground truth
        remaining = [np.logical_and(np.logical_not(p_i), g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p_perprotein = [p_i * ic_arr[tois] for p_i, tois in zip(p_perprotein, toi_perprotein)]
            intersection = [inter * ic_arr[tois] for inter, tois in zip(intersection, toi_perprotein)]  # TP
            mis = [misinf * ic_arr[tois] for misinf, tois in zip(mis, toi_perprotein)]  # FP, predicted but not in the ground truth
            remaining = [rem * ic_arr[tois] for rem, tois in zip(remaining, toi_perprotein)]  # FN, not predicted but in the ground truth

        n_pred = np.array([p_i.sum() for p_i in p_perprotein])  # TP + FP
        n_intersection = np.array([inter.sum() for inter in intersection])  # TP
        precision = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0)
        recall = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0)

        # metrics tests
        test_norm_metric(precision, name='precision')
        test_norm_metric(recall, name='recall')
        test_intersection(n_intersection, n_pred, n_gt)


        # Coverage numerator: count only proteins that remain eligible after
        # exclusion. Keep TP/FP/FN over the original PK prediction cells so
        # predictions on ineligible proteins can still contribute FP/MI.
        metrics[i, 0] = ((n_pred > 0) & eligible_rows).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP
        metrics[i, 2] = np.sum([m.sum() for m in mis])  # FP
        metrics[i, 3] = np.sum([r.sum() for r in remaining])  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0).sum()  # Precision
        metrics[i, 5] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum()  # Recall

        metrics_per_protein = pd.DataFrame({
            'n_pred': n_pred,
            'TP': n_intersection,
            'FP': [m.sum() for m in mis],
            'FN': [r.sum() for r in remaining],
            'n_gt': n_gt,
            # Bootstrap coverage resamples proteins, so it needs the same
            # post-exclusion eligibility predicate used by the main coverage
            # numerator. Without carrying this mask into bootstrap, sampled
            # rows whose GT was fully removed can inflate cov/cov_w.
            'eligible': eligible_rows,
        })
        if B_ind is not None:
            metrics_B_tau[tau] = bootstrap(metrics_per_protein, B_ind)
    #if B_ind is not None:
    #    metrics_B = get_metrics_B(metrics_B_tau)

    return metrics, metrics_B_tau

# Input-> metrics_B_tau : a dict where thresholds are the keys, and a metrics array per threshold containing B rows, is in the values
# output-> metrics_B: a dict where a b index corresponding to each bootstrap round is the key and a metrics array containing one row per tau (threshold) is the output
def get_metrics_B(metrics_B_tau):
    taus = sorted(list(metrics_B_tau.keys()))
    B = len(metrics_B_tau[taus[0]]) #B = number of rows in the dict at the first key (threshold)
    metrics_B = {}
    columns = ["n", "tp", "fp", "fn", "pr", "rc"]
    if metrics_B_tau[taus[0]].shape[1] == 7:
        columns.append("n_eval")
    for b in range(B):
        rows = []
        metrics_b = np.zeros((len(metrics_B_tau.keys()), len(columns)), dtype='float')
        for i, tau in enumerate(taus):
            metrics_b[i] = metrics_B_tau[tau][b]
        metrics_B[b] = pd.DataFrame(metrics_b, columns=columns)
    return metrics_B


def get_bootstrap_test_indices(B_ind):
    """
    Select a deterministic 10% subset of bootstrap replicates for runtime checks.

    The invariant checks recompute per-replicate sums from the sampled rows.
    Running them for every bootstrap replicate is redundant and can be costly
    when B is large, so validate a small subset derived from B_ind.
    """
    if len(B_ind) == 0:
        return set()
    n_tests = max(1, int(np.ceil(len(B_ind) * 0.10)))
    return set(np.linspace(0, len(B_ind) - 1, n_tests, dtype='int'))


def bootstrap(metrics_per_protein, B_ind):
    has_eligibility = "eligible" in metrics_per_protein.columns
    # test_indices = get_bootstrap_test_indices(B_ind)
    metrics_B_tau = np.zeros((len(B_ind), 7), dtype='float')
    for b, ind in enumerate(B_ind):
        metrics_per_protein_b = metrics_per_protein.iloc[ind]
        # n_gt_b = n_gt[ind]
        # p_b = p[ind]
        # intersection_b = intersection[ind]
        # mis_b = mis[ind]
        # remaining_b = remaining[ind]

        #n_pred_b = p_b.sum(axis=1)  # TP + FP
        #n_intersection_b = intersection_b.sum(axis=1)  # TP

        # Number of proteins with at least one term predicted with score >= tau.
        # In partial-knowledge evaluation, coverage counts only sampled rows
        # that still have evaluable GT after known annotations are excluded.
        if has_eligibility:
            eligible_b = metrics_per_protein_b["eligible"].astype(bool)
            metrics_B_tau[b, 0] = ((metrics_per_protein_b["n_pred"] > 0) & eligible_b).sum()
            metrics_B_tau[b, 6] = eligible_b.sum()
        else:
            metrics_B_tau[b, 0] = (metrics_per_protein_b["n_pred"] > 0).sum()
            metrics_B_tau[b, 6] = len(metrics_per_protein_b)

        # Sum of confusion matrices
        metrics_B_tau[b, 1] = metrics_per_protein_b["TP"].sum()  # TP
        metrics_B_tau[b, 2] = metrics_per_protein_b["FP"].sum()  # FP
        metrics_B_tau[b, 3] = metrics_per_protein_b["FN"].sum()  # FN

        # Macro-averaging
        metrics_B_tau[b, 4] = np.divide(metrics_per_protein_b["TP"], metrics_per_protein_b["n_pred"], out=np.zeros_like(metrics_per_protein_b["TP"], dtype='float'),
                                  where=metrics_per_protein_b["n_pred"] > 0).sum()  # Precision
        metrics_B_tau[b, 5] = np.divide(metrics_per_protein_b["TP"], metrics_per_protein_b["n_gt"], out=np.zeros_like(metrics_per_protein_b["n_gt"], dtype='float'),
                                  where=metrics_per_protein_b["n_gt"] > 0).sum()  # Recall
        # if b in test_indices:
        #     test_bootstrap_metrics(metrics_B_tau[b], metrics_per_protein_b, has_eligibility)

    return metrics_B_tau


def bootstrap_exclude(p_perprotein, intersection, mis, remaining, n_gt, B_ind, eligible_rows=None):
    has_eligibility = eligible_rows is not None
    if has_eligibility:
        eligible_rows = np.asarray(eligible_rows)
    # test_indices = get_bootstrap_test_indices(B_ind)
    metrics_B_tau = np.zeros((len(B_ind), 7), dtype='float')
    for b, ind in enumerate(B_ind):
        n_gt_b = n_gt[ind]

        p_perprotein_b = [p_perprotein[p] for p in ind]
        intersection_b = [intersection[p] for p in ind]
        mis_b = [mis[p] for p in ind]
        remaining_b = [remaining[p] for p in ind]

        n_pred_b = np.array([p_i.sum() for p_i in p_perprotein_b])  # TP + FP
        n_intersection_b = np.array([inter.sum() for inter in intersection_b])  # TP

        # Number of proteins with at least one term predicted with score >= tau
        if has_eligibility:
            eligible_b = eligible_rows[ind].astype(bool)
            metrics_B_tau[b, 0] = ((n_pred_b > 0) & eligible_b).sum()
            metrics_B_tau[b, 6] = eligible_b.sum()
        else:
            metrics_B_tau[b, 0] = (n_pred_b > 0).sum()
            metrics_B_tau[b, 6] = len(n_pred_b)

        # Sum of confusion matrices
        metrics_B_tau[b, 1] = n_intersection_b.sum()  # TP
        metrics_B_tau[b, 2] = np.sum([m.sum() for m in mis_b])  # FP
        metrics_B_tau[b, 3] = np.sum([r.sum() for r in remaining_b])  # FN

        # Macro-averaging
        metrics_B_tau[b, 4] = np.divide(n_intersection_b, n_pred_b, out=np.zeros_like(n_intersection_b, dtype='float'),
                                  where=n_pred_b > 0).sum()  # Precision
        metrics_B_tau[b, 5] = np.divide(n_intersection_b, n_gt_b, out=np.zeros_like(n_gt_b, dtype='float'),
                                  where=n_gt_b > 0).sum()  # Recall
        metrics_per_protein_b = pd.DataFrame({
            'n_pred': n_pred_b,
            'TP': n_intersection_b,
            'FP': [m.sum() for m in mis_b],
            'FN': [r.sum() for r in remaining_b],
            'n_gt': n_gt_b,
        })
        if has_eligibility:
            metrics_per_protein_b['eligible'] = eligible_b
        # if b in test_indices:
        #     test_bootstrap_metrics(metrics_B_tau[b], metrics_per_protein_b, has_eligibility)

    return metrics_B_tau

def compute_metrics(pred, gt_matrix, tau_arr, toi, gt_exclude=None, ic_arr=None, n_cpu=0, B_ind = None):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    if n_cpu == 0:
        n_cpu = mp.cpu_count()
    n_cpu = max(1, min(int(n_cpu), max(1, len(tau_arr))))

    columns = ["n", "tp", "fp", "fn", "pr", "rc"]
    n_terms = gt_matrix.shape[1]
    full_toi = toi_is_full(toi, n_terms)
    sparse_enabled = use_sparse()

    if full_toi:
        proteins_has_gt = (gt_matrix != 0).any(axis=1)
    else:
        proteins_has_gt = (gt_matrix[:, toi] != 0).any(axis=1)
    proteins_with_gt = np.where(proteins_has_gt)[0]

    metrics_B = []
    metrics_B_tau = {}

    if gt_exclude is None:
        if proteins_has_gt.all():
            gt_with_annots = gt_matrix
            pred_filtered = pred
        else:
            gt_with_annots = gt_matrix[proteins_with_gt, :]
            pred_filtered = pred[proteins_has_gt, :]

        if full_toi:
            g = gt_with_annots
            p = pred_filtered
        else:
            g = gt_with_annots[:, toi]
            p = pred_filtered[:, toi]

        if ic_arr is None:
            n_gt = g.sum(axis=1)
        else:
            n_gt = (g * ic_arr[toi]).sum(axis=1)

        if sparse_enabled:
            metrics_arr, metrics_B_tau = compute_confusion_matrix_sparse(
                tau_arr, g, p, toi, n_gt, ic_arr, B_ind=B_ind
            )
            metrics = pd.DataFrame(metrics_arr, columns=columns)
            if metrics_B_tau:
                metrics_B = get_metrics_B(metrics_B_tau)
        else:
            if issparse(p):
                p = p.toarray()
            arg_lists = [
                [tau_chunk, g, p, toi, n_gt, ic_arr, B_ind]
                for tau_chunk in np.array_split(tau_arr, n_cpu)
            ]
            with mp.Pool(processes=n_cpu) as pool:
                results = pool.starmap(compute_confusion_matrix, arg_lists)
            metrics = pd.DataFrame(np.concatenate([r[0] for r in results]), columns=columns)
            for _thread, result in enumerate(results):
                for tau, metrics_b in result[1].items():
                    metrics_B_tau[tau] = metrics_b
            if metrics_B_tau:
                metrics_B = get_metrics_B(metrics_B_tau)
    else:
        if proteins_has_gt.all():
            gt_with_annots = gt_matrix
            pred_sub = pred
        else:
            gt_with_annots = gt_matrix[proteins_with_gt, :]
            pred_sub = pred[proteins_has_gt, :]

        if sparse_enabled:
            if full_toi:
                toi_mask = np.ones(n_terms, dtype=bool)
            else:
                toi_mask = np.zeros(n_terms, dtype=bool)
                toi_mask[toi] = True
            excluded_mask = gt_exclude.matrix[proteins_with_gt, :]

            gt_nz_rows, gt_nz_cols = np.nonzero(gt_with_annots)
            if gt_nz_rows.size:
                if not full_toi:
                    keep = toi_mask[gt_nz_cols]
                    gt_nz_rows = gt_nz_rows[keep]
                    gt_nz_cols = gt_nz_cols[keep]
                keep = ~excluded_mask[gt_nz_rows, gt_nz_cols]
                gt_nz_rows = gt_nz_rows[keep]
                gt_nz_cols = gt_nz_cols[keep]

            n_prot_with_gt = gt_with_annots.shape[0]
            if ic_arr is None:
                n_gt = np.bincount(gt_nz_rows, minlength=n_prot_with_gt).astype(np.float64)
            else:
                n_gt = np.bincount(
                    gt_nz_rows, weights=ic_arr[gt_nz_cols], minlength=n_prot_with_gt
                ).astype(np.float64)
            warn_empty_post_exclusion_gt(n_gt, ic_arr)

            metrics_arr, metrics_B_tau, _eligible_rows = compute_confusion_matrix_exclude_sparse(
                tau_arr, pred_sub, gt_with_annots, toi_mask, excluded_mask, n_gt, ic_arr, B_ind=B_ind
            )
            metrics = pd.DataFrame(metrics_arr, columns=columns)
            if metrics_B_tau:
                metrics_B = get_metrics_B(metrics_B_tau)
        else:
            toi_perprotein = [
                np.setdiff1d(toi, gt_exclude.matrix[p, :].nonzero()[0], assume_unique=True)
                for p in proteins_with_gt
            ]
            gt_perprotein = [
                gt_with_annots[p_idx, tois]
                for p_idx, tois in enumerate(toi_perprotein)
            ]
            n_gt = np.array([gpp.sum().item() for gpp in gt_perprotein])
            if ic_arr is not None:
                n_gt = np.array([
                    (gpp * ic_arr[tois]).sum().item()
                    for gpp, tois in zip(gt_perprotein, toi_perprotein)
                ])
            warn_empty_post_exclusion_gt(n_gt, ic_arr)
            eligible_rows = np.array([gpp.sum().item() > 0 for gpp in gt_perprotein])
            if issparse(pred_sub):
                pred_sub = pred_sub.toarray()
            arg_lists = [
                [tau_chunk, gt_perprotein, pred_sub, toi_perprotein, n_gt, eligible_rows, ic_arr, B_ind]
                for tau_chunk in np.array_split(tau_arr, n_cpu)
            ]
            with mp.Pool(processes=n_cpu) as pool:
                results = pool.starmap(compute_confusion_matrix_exclude, arg_lists)
            metrics = pd.DataFrame(np.concatenate([r[0] for r in results]), columns=columns)
            for _thread, result in enumerate(results):
                for tau, metrics_b in result[1].items():
                    metrics_B_tau[tau] = metrics_b
            if metrics_B_tau:
                metrics_B = get_metrics_B(metrics_B_tau)

    return metrics, metrics_B


def make_bootstrap_indices(n_rows, B=0, B_pct=0, rng=None):
    """
    Generate bootstrap indices in the row space used by compute_metrics().

    The evaluator filters GT proteins to the term set before building
    per-protein metric rows, and weighted IA evaluation can use a smaller row
    set than unweighted evaluation. Bootstrap indices therefore must be based
    on the filtered metric-row count, not on the original GT matrix length.
    """
    B_ind = []
    if B and B_pct > 0:
        rng = random if rng is None else rng
        nB = round((B_pct / 100) * n_rows)
        for b in range(B):
            B_ind.append(rng.choices(range(0, n_rows), k=nB))
    return B_ind


def make_paired_bootstrap_indices(gt, ontologies, gt_exclude=None, B=0, B_pct=0, rng=None):
    """
    Generate bootstrap replicate indices once per namespace and metric mode.

    Replicate b then refers to the same resampled target rows for every
    prediction file. Weighted metrics get their own row space when IA filtering
    changes the eligible target set.
    """
    paired_indices = {}
    for ns in gt:
        unweighted_rows = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi)
        unweighted_B_ind = make_bootstrap_indices(len(unweighted_rows), B, B_pct, rng=rng)
        paired_indices[ns] = {
            'unweighted': unweighted_B_ind,
            'weighted': [],
        }

        if ontologies[ns].ia is None:
            continue

        weighted_rows = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi_ia)
        # Matching lengths is not enough: integer row positions must refer to
        # the same proteins before unweighted draws can be reused.
        if np.array_equal(weighted_rows, unweighted_rows):
            paired_indices[ns]['weighted'] = unweighted_B_ind
        else:
            paired_indices[ns]['weighted'] = make_bootstrap_indices(len(weighted_rows), B, B_pct, rng=rng)

    return paired_indices


def normalize(metrics, ns, tau_arr, ne, normalization):

    # Bootstrap metrics are computed on resampled proteins, so every metric
    # normalized by the evaluable GT population must use the resampled
    # denominator (`n_eval`). Otherwise cov uses the bootstrap denominator
    # while recall/MI/RU still use the original full-run denominator.
    gt_denominator = metrics["n_eval"].to_numpy(dtype='float') if "n_eval" in metrics.columns else ne

    # Normalize columns
    for column in metrics.columns:
        if column not in ["n", "n_eval"]:
            # By default normalize by gt
            denominator = gt_denominator
            # Otherwise normalize by pred
            if normalization == 'pred' or (normalization == 'cafa' and column == "pr"):
                denominator = metrics["n"]
            metrics[column] = np.divide(metrics[column], denominator,
                                        out=np.zeros_like(metrics[column], dtype='float'),
                                        where=denominator > 0)

    metrics['ns'] = [ns] * len(tau_arr)
    metrics['tau'] = tau_arr
    metrics['cov'] = np.divide(metrics['n'], gt_denominator,
                               out=np.zeros_like(metrics['n'], dtype='float'),
                               where=gt_denominator > 0)
    test_norm_metric(metrics['cov'], name='coverage')
    metrics['mi'] = metrics['fp']
    metrics['ru'] = metrics['fn']

    metrics['f'] = compute_f(metrics['pr'], metrics['rc'])
    metrics['s'] = compute_s(metrics['ru'], metrics['mi'])

    # Micro-average, calculation is based on the average of the confusion matrices
    metrics['pr_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fp'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fp']) > 0)
    metrics['rc_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fn'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fn']) > 0)
    metrics['f_micro'] = compute_f(metrics['pr_micro'], metrics['rc_micro'])

    if "n_eval" in metrics.columns:
        metrics.drop(columns=["n_eval"], inplace=True)

    return metrics


def evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude=None,
                        normalization='cafa', n_cpu=0, B=0, B_pct=0,
                        bootstrap_indices=None):

    dfs = []
    dfs_w = []
    metrics_B_dfs = []
    metrics_B_w_dfs = []
    paired_bootstrap = bootstrap_indices is not None

    # Unweighted metrics
    for ns in prediction:
        # number of proteins with positive annotations
        proteins_with_gt = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi)
        if gt_exclude is None:
            exclude = None
        else:
            exclude = gt_exclude[ns]
        num_annot_prots = count_proteins_in_toi(
            gt[ns].matrix, ontologies[ns].toi, exclude.matrix if exclude is not None else None
        )

        ne = np.full(len(tau_arr), num_annot_prots)

        # Generate B sets of indices in the filtered metric-row space.
        unweighted_bootstrap_n_rows = len(proteins_with_gt)
        if paired_bootstrap:
            B_ind = bootstrap_indices.get(ns, {}).get('unweighted', [])
        else:
            B_ind = make_bootstrap_indices(unweighted_bootstrap_n_rows, B, B_pct)

        metrics, metrics_B = compute_metrics(prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi, exclude, None, n_cpu, B_ind = B_ind)
        dfs.append(normalize(metrics, ns, tau_arr, ne, normalization))
        metrics_B_df = []
        if metrics_B:
            for b in metrics_B.keys():
                metrics_b = normalize(metrics_B[b], ns, tau_arr, ne, normalization)
                metrics_b["b"] = b
                metrics_B_df.append(metrics_b)
            metrics_B_dfs.append(pd.concat(metrics_B_df)) # Concats dfs for each b iteration together

        # Weighted metrics
        if ontologies[ns].ia is not None:
            # number of proteins with positive annotations
            proteins_with_gt = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi_ia)

            if gt_exclude is None:
                exclude = None
            else:
                exclude = gt_exclude[ns]
            num_annot_prots = count_proteins_in_toi(
                gt[ns].matrix, ontologies[ns].toi_ia, exclude.matrix if exclude is not None else None
            )

            ne = np.full(len(tau_arr), num_annot_prots)
            if paired_bootstrap:
                B_ind_w = bootstrap_indices.get(ns, {}).get('weighted', [])
            elif len(proteins_with_gt) == unweighted_bootstrap_n_rows:
                B_ind_w = B_ind
            else:
                B_ind_w = make_bootstrap_indices(len(proteins_with_gt), B, B_pct)
            metrics_w, metrics_B_w = compute_metrics(prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi_ia, exclude, ontologies[ns].ia, n_cpu, B_ind = B_ind_w)
            dfs_w.append(normalize(metrics_w, ns, tau_arr, ne, normalization))
            metrics_B_w_df = []
            if metrics_B_w:
                for b in metrics_B_w.keys():
                    metrics_b_w = normalize(metrics_B_w[b], ns, tau_arr, ne, normalization)
                    metrics_b_w["b"] = b
                    metrics_B_w_df.append(metrics_b_w)
                metrics_B_w_dfs.append(pd.concat(metrics_B_w_df))  # Concats dfs for each b iteration together

    dfs = pd.concat(dfs) # Concats df from each ns

    # Merge weighted and unweighted dataframes
    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        dfs = pd.merge(dfs, dfs_w, on=['ns', 'tau'], suffixes=('', '_w'))

    if metrics_B_dfs:
        metrics_B_dfs = pd.concat(metrics_B_dfs)
        if metrics_B_w_dfs:
            metrics_B_w_dfs = pd.concat(metrics_B_w_dfs)
            metrics_B_dfs = pd.merge(metrics_B_dfs, metrics_B_w_dfs, on=['ns', 'tau', 'b'], suffixes=('', '_w'))

    return dfs, metrics_B_dfs


def cafa_eval(obo_file, pred_dir, gt_file, ia=None, no_orphans=False, norm='cafa', prop='max',
              exclude=None, toi_file=None, max_terms=None, th_step=0.01, n_cpu=1, B = 0, B_pct = 50):

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    if toi_file is not None:
        ontologies = update_toi(ontologies, toi_file)

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)
    if exclude is not None:
        gt_exclude = gt_exclude_parser(exclude, gt, ontologies)
    else:
        gt_exclude = None
    bootstrap_indices = None
    if use_paired_bootstrap():
        bootstrap_indices = make_paired_bootstrap_indices(gt, ontologies, gt_exclude, B, B_pct)

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    # Parse prediction files and perform evaluation
    dfs = []
    metrics_B_dfs = []
    for file_name in pred_files:
        print(file_name)
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        if not prediction:
            logging.warning("Prediction: {}, not evaluated".format(file_name))
        else:
            df_pred, metrics_B_df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude,
                                          normalization=norm, n_cpu=n_cpu, B = B, B_pct= B_pct,
                                          bootstrap_indices=bootstrap_indices)
            df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
            dfs.append(df_pred)
            if isinstance(metrics_B_df_pred, pd.DataFrame):
                metrics_B_df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
                metrics_B_dfs.append(metrics_B_df_pred)
            logging.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    df = None
    dfs_best = {}
    metrics_B_df = None
    if metrics_B_dfs:
        metrics_B_df = pd.concat(metrics_B_dfs)


    if dfs:
        df = pd.concat(dfs)

        # Remove rows with no coverage
        df = df[df['cov'] > 0].reset_index(drop=True)
        df.set_index(['filename', 'ns', 'tau'], inplace=True)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [('f', ['rc', 'pr']), ('f_w', ['rc_w', 'pr_w']), ('s', ['ru', 'mi']), ('f_micro', ['rc_micro', 'pr_micro']), ('f_micro_w', ['rc_micro_w', 'pr_micro_w'])]:
            if metric in df.columns:
                index_best = df.groupby(level=['filename', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['filename', 'ns'])[metric].idxmin()
                df_best = df.loc[index_best]
                if metric[-2:] != '_w':
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov'].max()
                else:
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov_w'].max()
                dfs_best[metric] = df_best
    else:
        logging.info("No predictions evaluated")

    return df, dfs_best, metrics_B_df


def write_results(df, dfs_best, metrics_B_df, out_dir='results', th_step=0.01):

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.to_csv('{}/evaluation_all.tsv'.format(out_folder), float_format="%.{}f".format(decimals), sep="\t")

    for metric in dfs_best:
        dfs_best[metric].to_csv('{}/evaluation_best_{}.tsv'.format(out_folder, metric), float_format="%.{}f".format(decimals), sep="\t")

    if isinstance(metrics_B_df, pd.DataFrame):
        metrics_B_df.to_csv('{}/Bootstrap_all.tsv'.format(out_folder), float_format="%.{}f".format(decimals), sep="\t")
