"""Optional shared view reducers. Importing this module loads no training/runtime deps.

This M0 interface is provisional. Mathematical state is bounded; callers own
partition identity, ordered coverage and duplicate delivery.
"""
from __future__ import annotations

import atexit
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import threading
from typing import Any

MAX_REQUEST_BYTES = 262_144
MAX_BATCH_VALUES = 1024
MAX_SAFE_INTEGER = 2**53 - 1


class ReducerError(ValueError):
    """Invalid reducer input, state, identity or exhausted execution budget."""


def assets():
    """Return package resources for offline/static serving (no network fetch)."""
    return files(__package__).joinpath("assets")


def descriptor() -> dict[str, Any]:
    return json.loads(assets().joinpath("reducer.json").read_text(encoding="utf-8"))


def _close_compiled(module, engine):
    # Cache ownership outlives individual requests. Release native resources
    # before Python clears Wasmtime's FFI globals during interpreter teardown.
    try:
        module.close()
    finally:
        try:
            engine.close()
        finally:
            _compiled.cache_clear()


@lru_cache(maxsize=1)
def _compiled():
    try:
        import wasmtime
    except ImportError as error:
        raise RuntimeError("Shared view reduction requires hypergan[reducers]; raw event readers do not.") from error
    spec = descriptor()
    if spec.get("abi") != 1 or spec.get("max_memory_bytes") != 16_777_216 or spec.get("max_request_bytes") != MAX_REQUEST_BYTES or spec.get("max_batch_values") != MAX_BATCH_VALUES:
        raise ReducerError("Unsupported bundled reducer manifest limits or ABI")
    binary = assets().joinpath("reducer.wasm").read_bytes()
    if hashlib.sha256(binary).hexdigest() != spec["sha256"]:
        raise ReducerError("Bundled reducer module digest mismatch")
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    module = wasmtime.Module(engine, binary)
    if module.imports:
        raise ReducerError("Bundled reducer must have no host imports")
    atexit.register(_close_compiled, module, engine)
    return wasmtime, spec, engine, module


class Reducer:
    """One bounded WASM instance; operations are serialized across Python threads."""

    def __init__(self, *, fuel: int = 50_000_000):
        if type(fuel) is not int or not 1 <= fuel <= 1_000_000_000:
            raise ValueError("fuel must be an integer in 1..1000000000")
        self._runtime, self.spec, engine, module = _compiled()
        self._store = self._runtime.Store(engine)
        self._store.set_limits(memory_size=self.spec["max_memory_bytes"], memories=1, instances=1)
        # Construction/ABI discovery is distinct from the caller's per-request
        # fuel budget. Even fuel=1 must construct before its first request traps.
        self._store.set_fuel(50_000_000)
        self._exports = self._runtime.Instance(self._store, module, []).exports(self._store)
        if self._exports["abi_version"](self._store) != 1:
            raise ReducerError("Unsupported reducer ABI")
        self._lock = threading.Lock()
        self._fuel = fuel
        self._failed = False

    @property
    def module_sha256(self) -> str:
        return self.spec["sha256"]

    @property
    def memory_bytes(self) -> int:
        return self._exports["memory"].data_len(self._store)

    def request(self, request: dict[str, Any]) -> Any:
        try:
            encoded = json.dumps(request, allow_nan=False, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        except (ValueError, TypeError, RecursionError) as error:
            raise ReducerError(f"Request must contain finite JSON values: {error}") from error
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ReducerError("Request exceeds 262144 bytes")
        with self._lock:
            if self._failed:
                raise ReducerError("Reducer instance trapped; create a new instance")
            store, exports = self._store, self._exports
            try:
                store.set_fuel(self._fuel)
                memory = exports["memory"]
                memory.write(store, encoded, exports["input_ptr"](store))
                length = exports["execute"](store, len(encoded))
                if not 0 < length <= 65_536:
                    raise ReducerError("Invalid reducer response length")
                start = exports["output_ptr"](store)
                response = json.loads(bytes(memory.read(store, start, start + length)))
            except self._runtime.Trap as error:
                self._failed = True
                raise ReducerError(f"Reducer execution budget exhausted or trapped: {error}") from error
            if "error" in response:
                raise ReducerError(response["error"])
            return response["ok"]

    def identity(self, reducer: str):
        return self.request({"op": "identity", "reducer": reducer})

    def add(self, state, values):
        return self.request({"op": "add", "state": state, "values": values})

    def merge(self, left, right):
        """Mathematical merge only: caller must establish disjoint coverage first."""
        return self.request({"op": "merge", "left": left, "right": right})

    def finalize(self, state):
        return self.request({"op": "finalize", "state": state})


_BOOTSTRAP_KEYS = {"identity", "reducer", "module_sha256", "start", "end", "state"}


def _offset(value):
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ReducerError("Coverage offset must be a nonnegative safe integer")


def validate_bootstrap(reducer: Reducer, bootstrap):
    if not isinstance(bootstrap, dict) or set(bootstrap) != _BOOTSTRAP_KEYS:
        raise ReducerError("Invalid bootstrap fields")
    _offset(bootstrap["start"])
    _offset(bootstrap["end"])
    if bootstrap["start"] > bootstrap["end"]:
        raise ReducerError("Reversed coverage range")
    if not isinstance(bootstrap["identity"], str) or not 1 <= len(bootstrap["identity"]) <= 256 or not bootstrap["identity"].isascii():
        raise ReducerError("Bootstrap identity must contain 1..256 ASCII characters")
    if bootstrap["module_sha256"] != reducer.module_sha256:
        raise ReducerError("Bootstrap module digest mismatch")
    if not isinstance(bootstrap["state"], dict) or bootstrap["state"].get("reducer") != bootstrap["reducer"]:
        raise ReducerError("Bootstrap reducer mismatch")
    reducer.finalize(bootstrap["state"])
    if bootstrap["start"] == bootstrap["end"] and bootstrap["state"] != reducer.identity(bootstrap["reducer"]):
        raise ReducerError("Empty coverage must contain identity state")


def bootstrap(reducer: Reducer, reducer_id: str, identity: str, *, start: int = 0):
    result = {"identity": identity, "reducer": reducer_id, "module_sha256": reducer.module_sha256,
              "start": start, "end": start, "state": reducer.identity(reducer_id)}
    validate_bootstrap(reducer, result)
    return result


def append_frame(reducer: Reducer, view, *, identity: str, start: int, end: int, values):
    """Commit a complete projection frame, including zero emissions, atomically.

    Already covered frames are replay and ignored. Partial overlaps and gaps fail.
    Source offsets refer to committed projection records, not plotted point count.
    """
    validate_bootstrap(reducer, view)
    _offset(start)
    _offset(end)
    if identity != view["identity"]:
        raise ReducerError("Frame identity mismatch")
    if start >= end:
        raise ReducerError("Frame must advance coverage")
    if start >= view["start"] and end <= view["end"]:
        return view
    if start != view["end"]:
        raise ReducerError("Coverage gap or partial overlap; request a new bootstrap")
    state = reducer.add(view["state"], values)
    return {**view, "end": end, "state": state}


def merge_bootstraps(reducer: Reducer, left, right):
    """Merge adjacent disjoint ranges in declared order; never average averages."""
    validate_bootstrap(reducer, left)
    validate_bootstrap(reducer, right)
    if any(left[key] != right[key] for key in ("identity", "reducer", "module_sha256")):
        raise ReducerError("Cannot merge incompatible view identities")
    if left["end"] != right["start"]:
        raise ReducerError("Only adjacent disjoint coverage can merge")
    return {**left, "end": right["end"], "state": reducer.merge(left["state"], right["state"])}
