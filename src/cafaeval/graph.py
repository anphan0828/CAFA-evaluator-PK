import logging
from collections import deque

import numpy as np

from cafaeval.sparse import (
    children_cache,
    propagate_serial,
    propagate_sparse_pushup,
    propagate_to_coo,
    use_sparse,
)


logging.getLogger(__name__).addHandler(logging.NullHandler())


class Graph:
    """
    Ontology class. One ontology == one namespace.
    Parent and child edges are stored as sparse CSR-style index arrays.
    """
    def __init__(self, namespace, terms_dict, ia_dict=None, orphans=False):
        self.namespace = namespace
        self.terms_dict = {}
        self.terms_dict_alt = {}
        self.terms_list = []
        self.idxs = None
        self.order = None
        self.toi = None
        self.toi_ia = None
        self.ia = None

        rel_list = []
        for self.idxs, (term_id, term) in enumerate(terms_dict.items()):
            rel_list.extend([[term_id, rel, term['namespace']] for rel in term['rel']])
            self.terms_list.append({
                'id': term_id,
                'name': term['name'],
                'namespace': namespace,
                'def': term['def'],
                'adj': set(),
                'children': set(),
            })
            self.terms_dict[term_id] = {
                'index': self.idxs,
                'name': term['name'],
                'namespace': namespace,
                'def': term['def'],
            }
            for a_id in term['alt_id']:
                self.terms_dict_alt.setdefault(a_id, set()).add(term_id)

        self.idxs += 1

        for id1, id2, ns in rel_list:
            if self.terms_dict.get(id2):
                i = self.terms_dict[id1]['index']
                j = self.terms_dict[id2]['index']
                self.terms_list[i]['adj'].add(j)
                self.terms_list[j]['children'].add(i)
                logging.debug("i,j {},{} {},{}".format(i, j, id1, id2))
            else:
                logging.debug('Skipping branch to external namespace: {}'.format(id2))

        self._build_sparse_dag()
        logging.debug("dag edges {}".format(int(self._par_indptr[-1])))

        self.top_sort()
        logging.debug("order sorted {}".format(self.order))

        if orphans:
            self.toi = np.arange(self.idxs)
        else:
            self.toi = np.nonzero(self._parent_count > 0)[0]
        logging.debug("toi {}".format(self.toi))

        if ia_dict is not None:
            self.set_ia(ia_dict)

        logging.info("Ontology: {}, total {}, roots {}, leaves {}, alternative_ids {}".format(
            self.namespace,
            int(np.count_nonzero(self._parent_count)),
            int(np.count_nonzero(self._parent_count == 0)),
            int(np.count_nonzero(self._child_count == 0)),
            len(self.terms_dict_alt),
        ))

    def _build_sparse_dag(self):
        n = self.idxs
        parent_count = np.fromiter((len(t['adj']) for t in self.terms_list), dtype=np.int64, count=n)
        child_count = np.fromiter((len(t['children']) for t in self.terms_list), dtype=np.int64, count=n)

        par_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(parent_count, out=par_indptr[1:])
        chi_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(child_count, out=chi_indptr[1:])

        par_idx = np.empty(int(par_indptr[-1]), dtype=np.int64)
        chi_idx = np.empty(int(chi_indptr[-1]), dtype=np.int64)
        for t in range(n):
            adj = self.terms_list[t]['adj']
            if adj:
                par_idx[par_indptr[t]:par_indptr[t + 1]] = sorted(adj)
            children = self.terms_list[t]['children']
            if children:
                chi_idx[chi_indptr[t]:chi_indptr[t + 1]] = sorted(children)

        self._parent_count = parent_count
        self._child_count = child_count
        self._par_indptr, self._par_idx = par_indptr, par_idx
        self._chi_indptr, self._chi_idx = chi_indptr, chi_idx

    def top_sort(self):
        indexes = []
        visited = 0
        rows = self.idxs
        in_degree = self._child_count.copy()
        queue = deque(np.nonzero(in_degree == 0)[0].tolist())

        while queue:
            visited += 1
            idx = queue.popleft()
            indexes.append(idx)
            in_degree[idx] -= 1
            adj = self.terms_list[idx]['adj']
            if len(adj) > 0:
                for j in adj:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        if visited == rows:
            self.order = indexes
        else:
            raise Exception("The sparse matrix doesn't represent an acyclic graph")

    def set_ia(self, ia_dict):
        self.ia = np.zeros(self.idxs, dtype='float')
        for term_id in self.terms_dict:
            if ia_dict.get(term_id):
                self.ia[self.terms_dict[term_id]['index']] = ia_dict.get(term_id)
            else:
                logging.debug('Missing IA for term: {}'.format(term_id))
        np.nan_to_num(self.ia, copy=False, nan=0, posinf=0, neginf=0)
        self.toi_ia = np.nonzero(self.ia > 0)[0]


class Prediction:
    """Prediction score matrix for one ontology namespace."""
    def __init__(self, ids, matrix, namespace=None):
        self.ids = ids
        self.matrix = matrix
        self.namespace = namespace

    def __str__(self):
        return "\n".join([
            "{}\t{}\t{}".format(index, self.matrix[index], self.namespace)
            for index, _id in enumerate(self.ids)
        ])


class GroundTruth:
    def __init__(self, ids, matrix, namespace=None):
        self.ids = ids
        self.matrix = matrix
        self.namespace = namespace


_PROPAGATE_WORK_THRESHOLD = 800_000_000


def propagate(matrix, ont, order, mode='max', parallel=0, chunk_rows=65536,
              _shm_name=None, _shape=None, _dtype_str=None,
              _row_start=None, _row_end=None, _deepest=None, _triples=None):
    """
    Update inplace the score matrix (proteins x terms) propagating scores up
    to the root. Sparse mode is enabled by default via CAFAEVAL_SPARSE.
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    if _shm_name is None:
        if matrix is None:
            raise TypeError("matrix must not be None")
        if matrix.shape[0] == 0:
            raise Exception("Empty matrix")

        if use_sparse():
            propagate_sparse_pushup(matrix, ont, mode, triples=_triples)
            return

        has_any = np.any(matrix[:, order] != 0, axis=0)
        nonzero_idx = np.flatnonzero(has_any)
        if nonzero_idx.size == 0:
            raise Exception("The matrix is empty")
        deepest = int(nonzero_idx[0])
        order_ = order[deepest:]
        children_by_term = children_cache(ont)

        n_proc = int(parallel) if parallel else 0
        if n_proc > 1:
            sum_children = int(sum(children_by_term[t].size for t in order_))
            work = int(matrix.shape[0]) * sum_children
            if work < _PROPAGATE_WORK_THRESHOLD:
                n_proc = 0

        if n_proc <= 1:
            propagate_serial(matrix, order_, children_by_term, mode)
            return

        shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
        shm_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shm_matrix[:] = matrix
        try:
            n_rows = int(matrix.shape[0])
            chunk_rows = int(np.ceil(n_rows / n_proc))
            ctx = mp.get_context("spawn")
            procs = []
            for row_start in range(0, n_rows, chunk_rows):
                row_end = min(n_rows, row_start + chunk_rows)
                proc = ctx.Process(
                    target=propagate,
                    args=(None, ont, order),
                    kwargs={
                        "mode": mode,
                        "parallel": 0,
                        "chunk_rows": chunk_rows,
                        "_shm_name": shm.name,
                        "_shape": matrix.shape,
                        "_dtype_str": matrix.dtype.str,
                        "_row_start": row_start,
                        "_row_end": row_end,
                        "_deepest": deepest,
                    },
                )
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError("Worker failed")
            matrix[:] = shm_matrix
        finally:
            shm.close()
            shm.unlink()
        return

    existing = shared_memory.SharedMemory(name=_shm_name)
    try:
        shm_matrix = np.ndarray(_shape, dtype=np.dtype(_dtype_str), buffer=existing.buf)
        rows = shm_matrix[_row_start:_row_end, :]
        order_ = order[_deepest:]
        propagate_serial(rows, order_, children_cache(ont), mode)
    finally:
        existing.close()
