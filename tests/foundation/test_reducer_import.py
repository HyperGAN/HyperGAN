"""Base imports and missing optional runtime errors must stay lightweight."""
import subprocess
import sys


def test_shared_reducer_import_and_missing_runtime():
    result = subprocess.run([sys.executable, "-c", """
import sys
from hypergan.metrics_reducer import Reducer, descriptor
assert 'torch' not in sys.modules and 'wasmtime' not in sys.modules
assert descriptor()['abi'] == 1
sys.modules['wasmtime'] = None
try:
    Reducer()
except RuntimeError as error:
    assert 'hypergan[reducers]' in str(error)
else:
    raise AssertionError('Missing runtime must not silently fall back')
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
