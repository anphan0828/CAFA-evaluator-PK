import numpy as np


def test_norm_metric(metric, name=None):
    """
    Test if metric is between 0 and 1
    :param metric: array of metric to test
    :param name: string of metric name for error message if necessary
    """
    name = 'Metric' if name is None else name

    if not (metric.max() <= 1.0 and metric.min() >= 0):
        raise Exception(f'{name} error: max: {metric.max():.4f} min: {metric.min():.4f}')


def test_intersection(intersection, pred_counts, true_counts):
    """
    Test if intersection of counts is always smaller than counts from one method
    :param intersection: array of positive counts in intersection
    :param pred_counts: array of positive counts from prediction
    :param true_counts: array of positive counts from ground truth
    """
    if not np.all(pred_counts >= intersection):
        raise Exception(f'Count of positive prediction smaller than intersection')

    if not np.all(true_counts >= intersection):
        raise Exception(f'Count of positive annotations smaller than intersection')


def test_bootstrap_metrics(metrics_b, metrics_per_protein_b, has_eligibility=False):
    """
    Test that one bootstrap metric row matches the sampled per-protein rows.
    :param metrics_b: array of bootstrap metrics [n, TP, FP, FN, precision, recall, n_eval]
    :param metrics_per_protein_b: sampled per-protein metric dataframe
    :param has_eligibility: whether bootstrap coverage uses a PK eligibility mask
    """
    if len(metrics_b) < 7:
        raise Exception('Bootstrap metrics missing eligible-sample denominator')

    n_pred = np.asarray(metrics_per_protein_b["n_pred"], dtype='float')
    tp = np.asarray(metrics_per_protein_b["TP"], dtype='float')
    fp = np.asarray(metrics_per_protein_b["FP"], dtype='float')
    fn = np.asarray(metrics_per_protein_b["FN"], dtype='float')
    n_gt = np.asarray(metrics_per_protein_b["n_gt"], dtype='float')

    if has_eligibility:
        eligible = np.asarray(metrics_per_protein_b["eligible"], dtype='bool')
        n = ((n_pred > 0) & eligible).sum()
        n_eval = eligible.sum()
    else:
        n = (n_pred > 0).sum()
        n_eval = len(n_pred)

    if metrics_b[6] != n_eval:
        raise Exception(f'Bootstrap eligible count mismatch: {metrics_b[6]} != {n_eval}')
    if metrics_b[0] > metrics_b[6]:
        raise Exception(f'Bootstrap coverage numerator exceeds denominator: {metrics_b[0]} > {metrics_b[6]}')

    expected = np.array([
        n,
        tp.sum(),
        fp.sum(),
        fn.sum(),
        np.divide(tp, n_pred, out=np.zeros_like(tp, dtype='float'), where=n_pred > 0).sum(),
        np.divide(tp, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum(),
    ])

    if not np.allclose(metrics_b[:6], expected):
        raise Exception(f'Bootstrap metrics changed unexpectedly: {metrics_b[:6]} != {expected}')
