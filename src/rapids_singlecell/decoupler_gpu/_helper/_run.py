from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as csps
import numpy as np
import pandas as pd
import scipy.sparse as sps
from anndata import AnnData
from tqdm.auto import tqdm

from rapids_singlecell.decoupler_gpu._helper._data import extract
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._net import adjmat, idxmat, prune
from rapids_singlecell.decoupler_gpu._helper._pv import fdr_bh_axis1
from rapids_singlecell.preprocessing._utils import _sparse_to_dense

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapids_singlecell.decoupler_gpu._helper._data import DataType

# Dask is optional; if missing, we keep current behavior
try:
    import dask.array as da
except Exception:
    da = None

def _to_dask_obsm(df: pd.DataFrame, *, row_chunks: int | None = None):
    """
    Wrap a pandas DataFrame as a Dask array and return (darr, index, columns).
    Chunk rows by `row_chunks` if provided, else one chunk for all rows.
    """
    if da is None:
        return None, None, None
    vals = df.values  # NumPy (CPU) at this point
    n_rows = vals.shape[0]
    chunks = (min(max(1, row_chunks or n_rows), n_rows), -1)
    darr = da.from_array(vals, chunks=chunks, asarray=False)
    return darr, df.index.to_numpy(), df.columns.to_numpy()

def _return(
    name: str,
    data: DataType,
    es: pd.DataFrame,
    pv: pd.DataFrame,
    *,
    verbose: bool = False,
    # INTERNAL: whether the input X was Dask-backed (controls Dask obsm wrap)
    _as_dask: bool = False,
    _row_chunks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    if isinstance(data, AnnData):
        # if Dask requested and available, put Dask arrays into obsm and labels into uns
        if _as_dask and da is not None:
            darr_es, idx_es, cols_es = _to_dask_obsm(es, row_chunks=_row_chunks)
            darr_pv, idx_pv, cols_pv = (None, None, None)
            if pv is not None:
                darr_pv, idx_pv, cols_pv = _to_dask_obsm(pv, row_chunks=_row_chunks)

        if data.obs_names.size != es.index.size:
            _log("Provided AnnData contains empty observations, returning repaired object",
                 level="warn", verbose=verbose)
            data = data[es.index, :].copy()

        if _as_dask and da is not None and darr_es is not None:
            # scores
            data.obsm[f"score_{name}"] = darr_es
            data.uns[f"score_{name}_index"] = idx_es
            data.uns[f"score_{name}_columns"] = cols_es
            # pvals
            if pv is not None and darr_pv is not None:
                data.obsm[f"padj_{name}"] = darr_pv
                data.uns[f"padj_{name}_index"] = idx_pv
                data.uns[f"padj_{name}_columns"] = cols_pv
        else:
            data.obsm[f"score_{name}"] = es
            if pv is not None:
                data.obsm[f"padj_{name}"] = pv

        # if we sliced out empties, return repaired object; else keep None (in-place)
        return data if data.obs_names.size != es.index.size else None
    else:
        return es, pv



def _get_batch(mat, srt, end):
    # NEW: if Dask-backed, compute just this row block to a CPU array/sparse
    if da is not None and isinstance(mat, da.Array):
        mat = mat[srt:end, :].compute()
        srt, end = 0, mat.shape[0]  # we already sliced; normalize indices

    if sps.issparse(mat):
        bmat = csps.csr_matrix(mat[srt:end])
        bmat = _sparse_to_dense(bmat)
    elif csps.issparse(mat):
        bmat = _sparse_to_dense(mat[srt:end])
    elif isinstance(mat, np.ndarray):
        bmat = cp.array(mat[srt:end, :])
    elif isinstance(mat, cp.ndarray):
        bmat = mat[srt:end, :]
    else:
        # backed tuple path (unchanged)
        bmat, msk_col = mat
        bmat = bmat[srt:end, :]
        if sps.issparse(bmat):
            bmat = csps.csr_matrix(bmat)
            bmat = _sparse_to_dense(bmat)
        else:
            bmat = cp.array(bmat)
        bmat = bmat[:, msk_col]
    return bmat.astype(cp.float32)



def _mat_to_array(mat):
    if sps.issparse(mat):
        mat = csps.csr_matrix(mat)
        mat = _sparse_to_dense(mat)
    elif csps.issparse(mat):
        mat = _sparse_to_dense(mat)
    elif isinstance(mat, np.ndarray):
        mat = cp.array(mat)
    elif isinstance(mat, cp.ndarray):
        mat = mat
    else:
        raise ValueError(f"Unsupported matrix type: {type(mat)}")
    return mat.astype(cp.float32)


def _run(
    name: str,
    func: Callable,
    *,
    adj: bool,
    test: bool,
    data: DataType,
    net: pd.DataFrame,
    tmin: int | float = 5,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    bsize: int | float = 5000,
    verbose: bool = False,
    pre_load: bool = False,
    adj_pv_gpu: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    _log(f"{name} - Running {name}", level="info", verbose=verbose)
    # Process data
    mat, obs, var = extract(
        data,
        layer=layer,
        raw=raw,
        empty=empty,
        verbose=verbose,
        bsize=bsize,
        pre_load=pre_load,
    )
    issparse = sps.issparse(mat) or csps.issparse(mat)
    isbacked = isinstance(mat, tuple)
    isdask = (da is not None and isinstance(mat, da.Array))


    # Process net
    net = prune(features=var, net=net, tmin=tmin, verbose=verbose)
    # Handle stat type
    if adj:
        sources, targets, adjm = adjmat(features=var, net=net, verbose=verbose)
        adjm = cp.array(adjm, dtype=cp.float32)
        # Handle batches
        if issparse or isbacked or isdask:
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                if i == 0 and verbose:
                    batch_verbose = True
                else:
                    batch_verbose = False
                srt, end = i * bsize, i * bsize + bsize
                bmat = _get_batch(mat, srt, end)
                bes, bpv = func(bmat, adjm, verbose=batch_verbose, **kwargs)
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
            es = pd.DataFrame(es, index=obs, columns=sources)
        else:
            mat = _mat_to_array(mat)
            es, pv = func(mat, adjm, verbose=verbose, **kwargs)
            es = pd.DataFrame(es, index=obs, columns=sources)
    else:
        sources, cnct, starts, offsets = idxmat(features=var, net=net, verbose=verbose)
        cnct = cp.array(cnct, dtype=cp.int32)
        starts = cp.array(starts, dtype=cp.int32)
        offsets = cp.array(offsets, dtype=cp.int32)
        nbatch = int(np.ceil(obs.size / bsize))
        es, pv = [], []
        for i in tqdm(range(nbatch), disable=not verbose):
            srt, end = i * bsize, i * bsize + bsize
            bmat = _get_batch(mat, srt, end)
            if i == 0 and verbose:
                batch_verbose = True
            else:
                batch_verbose = False
            bes, bpv = func(
                bmat,
                cnct=cnct,
                starts=starts,
                offsets=offsets,
                verbose=batch_verbose,
                **kwargs,
            )
            es.append(bes)
            pv.append(bpv)
        es = np.vstack(es)
        es = pd.DataFrame(es, index=obs, columns=sources)
    # Handle pvals and FDR correction
    if test:
        pv = np.vstack(pv)
        pv = pd.DataFrame(pv, index=obs, columns=sources)
        if name != "mlm":
            _log(f"{name} - adjusting p-values by FDR", level="info", verbose=verbose)
            pv.loc[:, :] = fdr_bh_axis1(pv.values, if_gpu=adj_pv_gpu)
    else:
        pv = None
    _log(f"{name} - done", level="info", verbose=verbose)
    # Only wrap to Dask if input X was Dask-backed
    return _return(name, data, es, pv, verbose=verbose, _as_dask=isdask, _row_chunks=int(bsize) if isinstance(bsize, (int, np.integer, float)) else None,)

