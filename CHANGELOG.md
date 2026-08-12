# Change Log
All notable changes to this project will be documented in this file.

## [current] - 
- We are working to include the calculation of bootstrap confidence intervals.

- 2026-08-03: fixed the partial-knowledge coverage bug where `cov` or
  `cov_w` could exceed 1 when a protein had ground-truth annotations before
  known-annotation exclusion, but no evaluable annotations remained afterward.
  The evaluator now uses the same post-exclusion eligible protein population
  for the coverage numerator and denominator, including the IA-weighted
  setting. See `example/README.md` for the toy reproduction and expected
  corrected output. The same fix, with the same post-exclusion eligibility
  predicate, is recorded in `cafaeval-protea` as commit `cec8ccd` dated
  2026-04-23 and documented in `CHANGES.md` there.
- 2026-08-04: integrated the sparse optimization strategy from
  `cafaeval-protea` (Francisco Miguel Pérez Canales,
  https://github.com/frapercan/cafaeval-protea) into CAFA-evaluator-PK while
  preserving the committed bootstrap API and CLI. Source commits include
  protea `52acad7`
  (`perf: bring sparse eval (CSR storage + sparse GO DAG) to main`) and
  `80d705a` (`fix(graph): propagation fill bit-parity to main`).
- The shared-memory parallel DAG propagation that arrived with that port, the
  cached per-term children map and the serial propagation helper in
  `graph.py`, originate in CAFA-evaluator-PK-speedup by Antonina Dolgorukova
  (https://github.com/T0chka/CAFA-evaluator-PK-speedup), shared in the CAFA 6
  Kaggle discussion "Speeding up cafaeval". They reached this repository
  through `cafaeval-protea`, which cherry-picked her commits with git
  authorship preserved.
- Added `NOTICE` recording the full attribution chain for this line, from the
  original CAFA-evaluator through the partial-knowledge fork, the bootstrap
  work and the sparse port.
- Added `src/cafaeval/sparse.py` as the shared sparse helper module for
  environment gates, CSR non-zero extraction, sparse NK/LK and PK metric
  kernels, bootstrap-aware sparse metric aggregation, sparse propagation,
  and sparse-native COO propagation.
- Updated `evaluation.py` so `compute_metrics(..., B_ind=None)` still returns
  `(metrics_df, metrics_B)` while using the sparse kernels by default. Dense
  fallback remains available with `CAFAEVAL_SPARSE=0`.
- Adapted sparse PK evaluation to the existing bootstrap functions. Protea did
  not include bootstrap support, so this port adds sparse bootstrap aggregation
  with the same `n_eval` denominator semantics used by the PK bootstrap code.
- Preserved the 2026-08-03 partial-knowledge coverage fix while porting the
  sparse kernels. The sparse PK coverage numerator is restricted to the same
  post-exclusion eligible protein population as the denominator, including the
  IA-weighted setting.
- Updated `graph.py` to store ontology edges as sparse CSR-style parent and
  child arrays instead of a dense `(n_terms, n_terms)` DAG matrix.
- Updated `parser.py` to store predictions as `scipy.sparse.csr_matrix`,
  propagate default `prop='max'` predictions through sparse COO coordinates,
  and fall back to dense propagation for `prop='fill'`.
- Included protea parser delimiter edits: the legacy parser handles comma,
  tab, and whitespace prediction files; the optional PyArrow fast parser
  detects tab, comma, semicolon, pipe, and single-space delimiters and uses
  `pa.large_string()` for term IDs.
- Added `scipy` as a required dependency and `fast = ['pyarrow>=12']` as an
  optional dependency group.
- Migrated protea's synthetic parity harness and regression tests where
  portable, adapting them to this repository's three-value `cafa_eval()` return
  shape and the intentional PK oracle divergence caused by the coverage fix.
- 2026-08-04: made paired bootstrap the default `evaluation.py` behavior via
  `PAIRED_BOOTSTRAP=1`. Set `PAIRED_BOOTSTRAP=0` to recover the original
  per-prediction-file bootstrap sampling. `evaluation_bootstrap.py` is now a
  compatibility wrapper over the optimized `evaluation.py` implementation, so
  paired and non-paired modes use the same sparse metric and parser paths.

## [1.2.1] - 2024-03-26
- Minor bugfix affecting multi-thread calculation. 

## [1.2.0] - 2024-02-01 
- We included micro-average metrics. Now precision, recall and F-score 
in addition to previously reported metrics are calculated as micro-averages 
by averaging the confusion matrices over targets before calculating aggregated metrics
(precision, recall, ect.).
- We include the calculation of the average precision score (APS) in the plot notebook.

### Changed
- plot.ipynb, added calculation of average precision score (APS) in the plot notebook.
- evaluation.py, micro-average calculation, some refactoring of the core functions.
- parser.py, minor fixes and improvements.

## [1.1.0] - 2024-01-24
  
- We changed the way alternative identifiers in ontology files are considered.
Now alternative identifiers are recognized in both the ground truth and prediction files
and mapped to the "canonical" term.
 
### Added
- CHANGELOG.md, this file!
 
### Changed
- graph.py, changed the Graph class.
- parser.py, changed the ground truth and prediction parsers in order 
to replace alternative identifiers with canonical terms.
- plot.ipynb, cleaned up.

 
## [1.0.0] - 2023-08-04
 
First release after CAFA5 challenge closed. The version used in Kaggle is
provided under the 'Kaggle' branch. The 'main' branch includes instead a
packed version of the code. While usage and performance has been improved 
the calculation is exactly the same.
