"""Write a Python bootstrap fixture and optional measurements for the real browser proof.

Example: python scripts/reducer_proof.py --output /tmp/proof.json --benchmark
Serve a copied assets directory containing proof.json with python -m http.server.
"""
import argparse
import json
import math
from pathlib import Path
import time

from hypergan.metrics_reducer import Reducer, append_frame, bootstrap


def fixture():
    reducer = Reducer()
    values = [{"value": math.sin(i / 11), "position":[i, f"event-{i:04d}"]} for i in range(128)]
    cases = {}
    for kind in ["mean/v1", "envelope/v1"]:
        view = bootstrap(reducer,kind,"synthetic-run/generation-1/map-1/view-1/metric-1/bucket-0")
        view = append_frame(reducer,view,identity=view["identity"],start=0,end=96,values=values[:96])
        frames = [{"identity":view["identity"],"start":96,"end":128,"values":values[96:]}]
        expected = reducer.finalize(reducer.add(reducer.identity(kind),values))
        cases[kind] = {"bootstrap":view,"frames":frames,"expected":expected}
    return {"cases":cases,"plot":[v["value"] for v in values]}


def benchmark():
    start = time.perf_counter()
    reducer = Reducer()
    cold_ms = (time.perf_counter()-start)*1000
    values = [{"value":float(i),"position":[i,str(i)]} for i in range(1024)]
    result = {"construction_ms":cold_ms,"module_sha256":reducer.module_sha256,"runs":{}}
    for kind in ["mean/v1","envelope/v1"]:
        state = reducer.identity(kind)
        start = time.perf_counter()
        for _ in range(100): state = reducer.add(state,values)
        batched = time.perf_counter()-start
        single = reducer.identity(kind)
        start = time.perf_counter()
        for value in values: single = reducer.add(single,[value])
        singles = time.perf_counter()-start
        result["runs"][kind] = {"batch_1024_ms":batched*10,"values_per_second":102400/batched,
            "single_1024_calls_ms":singles*1000,"state_bytes":len(json.dumps(state)),"wasm_memory_bytes":reducer.memory_bytes}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmark",action="store_true")
    args = parser.parse_args()
    if args.benchmark: print(json.dumps(benchmark(),indent=2))
    args.output.write_text(json.dumps(fixture(),indent=2)+"\n")
