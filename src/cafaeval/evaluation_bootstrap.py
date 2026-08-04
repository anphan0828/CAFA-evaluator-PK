import argparse
import logging
import os

import numpy as np
import pandas as pd

from cafaeval.evaluation import compute_metrics, make_bootstrap_indices, normalize, write_results
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi


logging.getLogger(__name__).addHandler(logging.NullHandler())


def _proteins_with_gt(gt_matrix, toi):
    return np.where(gt_matrix[:, toi].sum(1) > 0)[0]


def _count_evaluable_proteins(gt_matrix, toi, proteins_with_gt, gt_exclude=None):
    if gt_exclude is None:
        return len(proteins_with_gt)

    toi_perprotein = [
        np.setdiff1d(toi, gt_exclude.matrix[p, :].nonzero()[0], assume_unique=True)
        for p in proteins_with_gt
    ]
    return sum(
        gt_matrix[p, toi_perprotein[p_idx]].sum() > 0
        for p_idx, p in enumerate(proteins_with_gt)
    )


def make_paired_bootstrap_indices(gt, ontologies, gt_exclude=None, B=0, B_pct=0):
    """
    Generate bootstrap replicate indices once per namespace and metric mode.

    The default evaluator creates bootstrap samples inside each prediction-file
    evaluation. This paired-bootstrap variant moves sample generation to the
    benchmark level so replicate b refers to the same resampled target rows for
    every prediction file. Weighted metrics get their own row space when IA
    filtering changes the eligible target set.
    """
    paired_indices = {}
    for ns in gt:
        unweighted_rows = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi)
        unweighted_B_ind = make_bootstrap_indices(len(unweighted_rows), B, B_pct)
        paired_indices[ns] = {
            'unweighted': unweighted_B_ind,
            'weighted': [],
        }

        if ontologies[ns].ia is None:
            continue

        weighted_rows = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi_ia)
        # Reuse the exact same bootstrap draws only when integer row positions
        # refer to the same proteins. Matching lengths alone is not enough.
        if np.array_equal(weighted_rows, unweighted_rows):
            paired_indices[ns]['weighted'] = unweighted_B_ind
        else:
            paired_indices[ns]['weighted'] = make_bootstrap_indices(len(weighted_rows), B, B_pct)

    return paired_indices


def evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude=None,
                        normalization='cafa', n_cpu=0, bootstrap_indices=None):
    dfs = []
    dfs_w = []
    metrics_B_dfs = []
    metrics_B_w_dfs = []

    for ns in prediction:
        proteins_with_gt = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi)
        exclude = None if gt_exclude is None else gt_exclude[ns]
        num_annot_prots = _count_evaluable_proteins(
            gt[ns].matrix, ontologies[ns].toi, proteins_with_gt, exclude
        )
        ne = np.full(len(tau_arr), num_annot_prots)
        B_ind = bootstrap_indices.get(ns, {}).get('unweighted', []) if bootstrap_indices else []

        metrics, metrics_B = compute_metrics(
            prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi,
            exclude, None, n_cpu, B_ind=B_ind
        )
        dfs.append(normalize(metrics, ns, tau_arr, ne, normalization))

        metrics_B_df = []
        if metrics_B:
            for b in metrics_B.keys():
                metrics_b = normalize(metrics_B[b], ns, tau_arr, ne, normalization)
                metrics_b["b"] = b
                metrics_B_df.append(metrics_b)
            metrics_B_dfs.append(pd.concat(metrics_B_df))

        if ontologies[ns].ia is not None:
            proteins_with_gt = _proteins_with_gt(gt[ns].matrix, ontologies[ns].toi_ia)
            num_annot_prots = _count_evaluable_proteins(
                gt[ns].matrix, ontologies[ns].toi_ia, proteins_with_gt, exclude
            )
            ne = np.full(len(tau_arr), num_annot_prots)
            B_ind_w = bootstrap_indices.get(ns, {}).get('weighted', []) if bootstrap_indices else []

            metrics_w, metrics_B_w = compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi_ia,
                exclude, ontologies[ns].ia, n_cpu, B_ind=B_ind_w
            )
            dfs_w.append(normalize(metrics_w, ns, tau_arr, ne, normalization))

            metrics_B_w_df = []
            if metrics_B_w:
                for b in metrics_B_w.keys():
                    metrics_b_w = normalize(metrics_B_w[b], ns, tau_arr, ne, normalization)
                    metrics_b_w["b"] = b
                    metrics_B_w_df.append(metrics_b_w)
                metrics_B_w_dfs.append(pd.concat(metrics_B_w_df))

    dfs = pd.concat(dfs)

    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        dfs = pd.merge(dfs, dfs_w, on=['ns', 'tau'], suffixes=('', '_w'))

    if metrics_B_dfs:
        metrics_B_dfs = pd.concat(metrics_B_dfs)
        if metrics_B_w_dfs:
            metrics_B_w_dfs = pd.concat(metrics_B_w_dfs)
            metrics_B_dfs = pd.merge(
                metrics_B_dfs, metrics_B_w_dfs, on=['ns', 'tau', 'b'], suffixes=('', '_w')
            )

    return dfs, metrics_B_dfs


def cafa_eval(obo_file, pred_dir, gt_file, ia=None, no_orphans=False, norm='cafa', prop='max',
              exclude=None, toi_file=None, max_terms=None, th_step=0.01, n_cpu=1, B=0, B_pct=50):
    tau_arr = np.arange(th_step, 1, th_step)

    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    if toi_file is not None:
        ontologies = update_toi(ontologies, toi_file)

    gt = gt_parser(gt_file, ontologies)
    if exclude is not None:
        gt_exclude = gt_exclude_parser(exclude, gt, ontologies)
    else:
        gt_exclude = None

    bootstrap_indices = make_paired_bootstrap_indices(gt, ontologies, gt_exclude, B, B_pct)

    pred_folder = os.path.normpath(pred_dir) + "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    dfs = []
    metrics_B_dfs = []
    for file_name in pred_files:
        print(file_name)
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        if not prediction:
            logging.warning("Prediction: {}, not evaluated".format(file_name))
        else:
            df_pred, metrics_B_df_pred = evaluate_prediction(
                prediction, gt, ontologies, tau_arr, gt_exclude,
                normalization=norm, n_cpu=n_cpu, bootstrap_indices=bootstrap_indices
            )
            df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
            dfs.append(df_pred)
            if isinstance(metrics_B_df_pred, pd.DataFrame):
                metrics_B_df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
                metrics_B_dfs.append(metrics_B_df_pred)
            logging.info("Prediction: {}, evaluated".format(file_name))

    df = None
    dfs_best = {}
    metrics_B_df = None
    if metrics_B_dfs:
        metrics_B_df = pd.concat(metrics_B_dfs)

    if dfs:
        df = pd.concat(dfs)
        df = df[df['cov'] > 0].reset_index(drop=True)
        df.set_index(['filename', 'ns', 'tau'], inplace=True)

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


def validate_percentage(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 100:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid percentage value. It must be between 0 and 100."
        )
    return ivalue


def command_line():
    logging.info("CAFA-evaluator paired bootstrap. Calculate precision-recall curves and F-max / S-min")

    parser = argparse.ArgumentParser(
        description='CAFA-evaluator with paired bootstrap samples across prediction files'
    )
    parser.add_argument('obo_file', help='Ontology file, OBO format')
    parser.add_argument('pred_dir', help='Predictions directory. Sub-folders are iterated recursively')
    parser.add_argument('gt_file', help='Ground truth file')
    parser.add_argument('-out_dir', default='results',
                        help='Output directory. By default it creates \"results/\" in the current directory')
    parser.add_argument('-ia', help='Information Accretion file (columns: <term> <information_accretion>)')
    parser.add_argument('-no_orphans', action='store_true', default=False,
                        help='Exclude terms without parents, e.g. the root(s), in the evaluation')
    parser.add_argument('-norm', choices=['cafa', 'pred', 'gt'], default='cafa',
                        help='cafa - implements the CAFA normalization strategy. '
                             'Precision is normalized by the number of predicted targets, '
                             'all other metrics by the number of ground truth targets. '
                             'pred - All metrics are normalized by the number of predicted targets. '
                             'gt - All metrics are normalized by the number of ground truth targets')
    parser.add_argument('-prop', choices=['max', 'fill'], default='max',
                        help='Ancestor propagation strategy. max - Propagate the max score of the traversed subgraph '
                             'iteratively. fill - Propagate with max until a different score is found')
    parser.add_argument('-known', default=None,
                        help='Known annotations to exclude from evaluation. Same format at ground truth file')
    parser.add_argument('-toi', default=None,
                        help='File with GO IDs for terms of interest on which to evaluate. '
                             'Usually, this is to exclude terms that were removed from GO '
                             'since evaluation obo file was created. If None (default), all terms will be'
                             'included in evaluation, possibly excluding roots if -no_orphans is passed')
    parser.add_argument('-th_step', type=float, default=0.01,
                        help='Threshold step size in the range [0, 1). A smaller step, means more calculation')
    parser.add_argument('-max_terms', type=int, default=None,
                        help='Number of terms for protein and namespace to consider in the evaluation')
    parser.add_argument('-threads', type=int, default=4,
                        help='Parallel threads. 0 means use all available CPU threads. '
                             'Do not use multithread if you are short in memory')
    parser.add_argument('-log_level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Log level')
    parser.add_argument('-B', type=int, default=0,
                        help='Number of bootstrap iterations')
    parser.add_argument('-B_pct', type=validate_percentage, default=0,
                        help='Percentage of points to include in bootstrap iterations')

    args = parser.parse_args()

    logging.basicConfig()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(args.log_level.upper()))
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(log_formatter)

    df, dfs_best, metrics_B_df = cafa_eval(
        args.obo_file, args.pred_dir, args.gt_file,
        ia=args.ia, no_orphans=args.no_orphans, norm=args.norm, prop=args.prop,
        exclude=args.known, toi_file=args.toi, max_terms=args.max_terms,
        th_step=args.th_step, n_cpu=args.threads, B=args.B, B_pct=args.B_pct
    )
    write_results(df, dfs_best, metrics_B_df, out_dir=args.out_dir, th_step=args.th_step)


if __name__ == "__main__":
    command_line()
