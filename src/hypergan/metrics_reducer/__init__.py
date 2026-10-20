"""Compatibility imports for HyperGAN clients; Foldlite DB owns the reducers."""
from foldlitedb.reducers import (
    MAX_BATCH_VALUES, MAX_REQUEST_BYTES, MAX_SAFE_INTEGER,
    Reducer, ReducerError, append_frame, assets, bootstrap, descriptor,
    merge_bootstraps, validate_bootstrap,
)

__all__ = [
    'MAX_BATCH_VALUES', 'MAX_REQUEST_BYTES', 'MAX_SAFE_INTEGER',
    'Reducer', 'ReducerError', 'append_frame', 'assets', 'bootstrap', 'descriptor',
    'merge_bootstraps', 'validate_bootstrap',
]
