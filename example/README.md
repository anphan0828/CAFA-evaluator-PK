# Example Evaluation

This directory contains files for an example of partial knowledge evaluation and the use of a terms-of-interest file.

This is a "toy" example using a subgraph of the Biological Process (BP) aspect of the Gene Ontology. It is only for
illustrative purposess.

![Example of partial knowledge evaluation](sample_eval.jpg)



To run the example,
``python3 /path/to/CAFA-evaluator/src/cafaeval/__main__.py go-sample.obo predictions ground_truth_partial.tsv -toi toi.tsv -known known_t0.tsv``

To run the same example with IA-weighted metrics,
``python3 /path/to/CAFA-evaluator/src/cafaeval/__main__.py go-sample.obo predictions ground_truth_partial.tsv -toi toi.tsv -known known_t0.tsv -ia ia_bug.tsv -out_dir results_fixcov``


## Known annotations

Partial knowledge evaluation is used when there are existing known annotations for a target (protein) in some aspect of the 
ontology on which evaluation will be done. These known annotations must be excluded from the evaluation since 
predictions are not needed for known annotations.

To exclude known terms, a list of annotations is passed in with the `-known` input option followed by a path 
to the file containing the list. In this example, the known annotations are known with a yellow background.
The proteins target1 and target2 both have known annotations in this aspect which we will exclude from evaluation.


Known annotations file (`known_t0.tsv`): list all known annotations for all targets  
```
target1 GO:0008150
target1 GO:0040011
target1 GO:0032501
target1 GO:0031987
target2 GO:0008150
target2 GO:0040011
target2 GO:0033058
```

## Ground truth
New annotations for target1 and target2 are considered in partial knowledge evaluation since these targets 
already had knonw annotations before/during the prediction phase.

Ground truth file (`ground_truth_partial.tsv`): list new annotations on which to evaluate
```
target1 GO:0033058
target1 GO:0043056
target1 GO:0050879
target1 GO:0071965
target2 GO:0031987
target2 GO:0032501
```

Note: if any of the known annotations are listed in the ground truth file, they will be ignored during 
evaluation. 

## Terms of Interest
It is also possible to exclude some terms from evaluation for _all_ targets. The terms-of-interest file
is an optional file that is used to list all terms on which to evaluate predictions. This list is 
applied for all targets. The list of terms is passed in with the `-toi` input option. If no file is 
passed in, all terms in the graph will be used for evaluation

In this example, terms of interest are shown in figure with a red, bold outline. Note that two of the 
terms in the ontology are excluded (GO:0008150 and GO:0050879) which will not be included in any computations
for evaluation, even if they appear in the ground truth file.

Terms-of-Interest file (`toi.tsv`):
```
GO:0040011
GO:0032501
GO:0033058
GO:0031987
GO:0071965
GO:0043056
GO:0043057
``` 

## Information Accretion and Weighted Evaluation

Information accretion (IA) values can be supplied with the `-ia` option to compute weighted precision, recall, F-score, misinformation, remaining uncertainty, and S-score. Weighted columns in the output use the `_w` suffix, for example `pr_w`, `rc_w`, `f_w`, `mi_w`, `ru_w`, `s_w`, and `cov_w`.

The IA file is a TSV file with two tab-separated columns (some GO terms IA are unrealistically set to zero as a minimal example to reproduce the bug):
```
GO:0040011	6.13
GO:0032501	0.00
GO:0033058	6.30
GO:0031987	0.00
GO:0071965	0.00
GO:0043056	1.70
GO:0043057	1.70
GO:0008150	0.00
GO:0050879	7.72
```

Only terms with positive IA are included in the weighted term set. In this toy example, `target2` remains eligible for the unweighted partial-knowledge evaluation after known annotations are excluded, but its remaining ground-truth terms (`GO:0031987` and `GO:0032501`) have IA 0. This makes `target2` useful for checking that weighted coverage is counted over the same post-exclusion eligible proteins as the weighted denominator.


## Example Prediction



The example predictions both achieve precision (pr) and recall (rc) of 1.0 (see table below). 
Looking at the precition files, pred_2 predicts eveything perfect, while pred_1 predicts many terms "incorrectly." 
However, all of the "errors" in pred_1 are on terms that are excluded either because they were not in the terms of interest 
or because they are previously known annotations.


![Example of predictions](predictions.png)


|   filename    |   ns                  |   tau    |   n      |   tp     |   fp     |   fn     |   pr     |   rc     |   cov    |   mi     |   ru     |   f      |   s      |   pr_micro  |   rc_micro  |   f_micro  |   cov_max  |
|---------------|-----------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-------------|-------------|------------|------------|
|   pred_1.tsv  |   biological_process  |   0.010  |   2.000  |   2.500  |   0.000  |   0.000  |   1.000  |   1.000  |   1.000  |   0.000  |   0.000  |   1.000  |   0.000  |   1.000     |   1.000     |   1.000    |   1.000    |
|   pred_2.tsv  |   biological_process  |   0.010  |   2.000  |   2.500  |   0.000  |   0.000  |   1.000  |   1.000  |   1.000  |   0.000  |   0.000  |   1.000  |   0.000  |   1.000     |   1.000     |   1.000    |   1.000    |
|               |                       |          |          |          |          |          |          |          |          |          |          |          |          |             |             |            |            |

## Weighted Coverage Regression Example

The `pred_3.tsv` file is a small regression fixture for the partial-knowledge weighted coverage calculation:
```
target1 GO:0043056 0.9999
target2 GO:0043056 0.9999
```

With `ia_bug.tsv`, `target1` has positive-IA ground truth after exclusion and `target2` does not. The old coverage calculation counted both proteins in the weighted coverage numerator if they had a positive-IA prediction, but counted only `target1` in the weighted denominator. That produced `cov_w > 1`.

Before the coverage fix, `pred_3.tsv` produced this invalid weighted coverage at `tau=0.01`:
```
filename    n    cov    n_w    cov_w
pred_3.tsv  2.0  1.0    2.0    2.0
```

The fixed evaluator should keep regular coverage and weighted coverage bounded by 1. At `tau=0.01`, `pred_3.tsv` should produce:
```
filename    n    cov    n_w    cov_w
pred_3.tsv  2.0  1.0    1.0    1.0
```
