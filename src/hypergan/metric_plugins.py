"""Bounded trusted Python metric factories; no tensor/runtime import in the host.

Each call gets a fresh group-free worker. The independent broker bounds startup,
execution and cleanup and reaps its worker when the coordinator dies. Plugin
allocations, hidden external state and plugin-created descendants are not a
security sandbox. No live trainer or sampler crosses this boundary.
"""
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time

MAX_PLUGIN_BYTES = 65536
MAX_CUSTOM_METRICS = 32
SCALAR_INPUTS = {'update.d_loss', 'update.g_loss', 'update.d_adversarial',
    'update.g_adversarial', 'update.d_adversarial_weighted', 'update.g_adversarial_weighted',
    'update.gradient_penalty', 'update.prior_loss', 'update.lr_scale', 'update.step', 'update.step_seconds'}


def finite_json(value):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded) > MAX_PLUGIN_BYTES:
        raise ValueError('Metric plugin input/output exceeds 64 KiB')
    return json.loads(encoded)


def validate_custom(specs):
    if not isinstance(specs, dict) or len(specs) > MAX_CUSTOM_METRICS:
        raise ValueError('metrics.custom must be a table with at most 32 metrics')
    for name, spec in specs.items():
        if not isinstance(name, str) or re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_./-]{0,127}', name) is None:
            raise ValueError('Custom metric IDs must be 1–128 letters, digits, dots, underscores, slashes or hyphens')
        allowed = {'factory', 'args', 'inputs', 'mode', 'every_steps', 'timeout', 'on_error', 'trigger', 'evaluation'}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError(f'Unknown or invalid custom metric fields: {name}')
        ref = spec.get('factory')
        if not isinstance(ref, str) or ref.count(':') != 1 or not all(re.fullmatch(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*', part) for part in ref.split(':')):
            raise ValueError(f'{name}.factory requires module:object')
        spec.setdefault('args', {})
        if not isinstance(spec['args'], dict):
            raise ValueError(f'{name}.args must be a table')
        finite_json(spec['args'])
        spec.setdefault('mode', 'scalar')
        if spec['mode'] not in ('scalar', 'snapshot'):
            raise ValueError(f'{name}.mode must be scalar or snapshot')
        spec.setdefault('timeout', 10.0 if spec['mode'] == 'scalar' else 120.0)
        if type(spec['timeout']) not in (float, int) or not math.isfinite(spec['timeout']) or not 0 < spec['timeout'] <= 3600:
            raise ValueError(f'{name}.timeout must be positive and at most 3600 seconds')
        spec.setdefault('on_error', 'disable' if spec['mode'] == 'scalar' else 'fail')
        if spec['on_error'] not in ('disable', 'fail'):
            raise ValueError(f'{name}.on_error must be disable or fail')
        inputs = spec.get('inputs')
        if not isinstance(inputs, dict) or not inputs or any(not isinstance(k, str) or not k.isidentifier() or k == 'context' or not isinstance(v, str) for k, v in inputs.items()):
            raise ValueError(f'{name}.inputs must bind non-context argument names to explicit sources')
        if spec['mode'] == 'scalar':
            spec.setdefault('every_steps', 1)
            if type(spec['every_steps']) is not int or spec['every_steps'] <= 0:
                raise ValueError(f'{name}.every_steps must be a positive integer')
            if set(inputs.values()) - SCALAR_INPUTS:
                raise ValueError(f'{name}.inputs must select supported update scalars')
            if 'trigger' in spec or 'evaluation' in spec:
                raise ValueError(f'{name}: scalar metrics do not accept snapshot options')
        else:
            if spec.get('trigger') != 'manual' or 'every_steps' in spec:
                raise ValueError(f'{name}: snapshot metrics currently require trigger="manual" and no every_steps; invoke evaluate explicitly')
            if set(inputs.values()) - {'evaluation.generated', 'evaluation.reference'}:
                raise ValueError(f'{name}.inputs must select evaluation.generated or evaluation.reference')
            evaluation = spec.get('evaluation')
            required = {'data', 'sample_count', 'batch_size', 'seed'}
            if not isinstance(evaluation, dict) or not required <= set(evaluation) or set(evaluation) - required - {'device'}:
                raise ValueError(f'{name}.evaluation requires explicit data, sample_count, batch_size and seed')
            evaluation.setdefault('device', 'cuda')
            if not isinstance(evaluation['device'], str) or re.fullmatch(r'cpu|cuda(?::(?:0|[1-9][0-9]*))?', evaluation['device']) is None:
                raise ValueError(f'{name}.evaluation.device requires cpu or cuda[:N]')
            for field, lower, upper in [('sample_count', 1, 1000000), ('batch_size', 1, 1024), ('seed', 0, 2**32-1)]:
                if type(evaluation[field]) is not int or not lower <= evaluation[field] <= upper:
                    raise ValueError(f'{name}.evaluation.{field} must be an integer in [{lower}, {upper}]')
            data = evaluation['data']
            if not isinstance(data, dict) or set(data) != {'factory', 'args'} or not isinstance(data['factory'], str) or not isinstance(data['args'], dict):
                raise ValueError(f'{name}.evaluation.data requires factory and args')
            if data['factory'] not in ('gaussian_grid', 'paired_linear', 'image_folder') and (data['factory'].count(':') != 1 or not all(data['factory'].split(':'))):
                raise ValueError(f'{name}.evaluation.data.factory requires built-in data or module:object')
        finite_json(spec)


def _factory(reference, args):
    module_name, qualname = reference.split(':')
    module = importlib.import_module(module_name)
    factory = module
    for part in qualname.split('.'):
        factory = getattr(factory, part)
    files = {}
    for candidate in (module, inspect.getmodule(factory)):
        path = Path(getattr(candidate, '__file__', ''))
        if not path.is_file():
            raise ValueError('Metric factories require inspectable Python source files')
        files[candidate.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
    instance = factory(**args)
    if not callable(getattr(instance, 'describe', None)) or not callable(getattr(instance, 'evaluate', None)):
        raise ValueError('Metric factory must provide describe() and evaluate()')
    descriptor = finite_json(instance.describe())
    allowed = {'kind', 'label', 'unit', 'direction', 'description'}
    if not isinstance(descriptor, dict) or set(descriptor) - allowed or descriptor.get('kind') not in ('scalar', 'histogram'):
        raise ValueError('Metric describe() requires scalar or histogram kind and presentation metadata only')
    if any(not isinstance(value, str) or len(value) > 1000 for value in descriptor.values()):
        raise ValueError('Metric descriptions must contain bounded text')
    descriptor.setdefault('label', qualname)
    descriptor.setdefault('unit', 'value')
    descriptor.setdefault('direction', 'none')
    if descriptor['direction'] not in ('none', 'minimize', 'maximize'):
        raise ValueError('Metric direction must be none, minimize or maximize')
    return instance, {'descriptor': descriptor, 'factory_sources': files, 'protocol': 'hypergan-metric/v1'}


def cpu_observation_resources():
    # This is the disposable observation worker, never the training process.
    # These are resource defaults for trusted code, not a security sandbox.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    if hasattr(os, 'nice'):
        os.nice(10)
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(1)


def _worker_factory(rank, world_size, spec, expected, inputs, context):
    cpu_observation_resources()
    instance, description = _factory(spec['factory'], spec['args'])
    if expected is not None and description != expected:
        raise ValueError('Metric factory source or descriptor changed since preflight')
    return instance, description, inputs, context


def _worker_command(state, operation, payload):
    instance, description, inputs, context = state
    if operation == 'describe':
        return description
    if operation != 'scalar':
        raise ValueError('Unsupported metric worker operation')
    result = instance.evaluate(context=context, **inputs)
    if type(result) not in (int, float) or not math.isfinite(result):
        raise ValueError('Scalar metric evaluate() must return a finite Python number')
    return {'value': result}


def invoke(spec, operation, *, expected=None, inputs=None, context=None, cancellation_event=None):
    from .cpu_worker_service import CPUWorkerService
    inputs, context = finite_json(inputs or {}), finite_json(context or {})
    service = CPUWorkerService(_worker_factory, _worker_command,
        args=(spec, expected, inputs, context), run_id='metric', attempt_id='metric',
        world_size=1, initialize_process_group=False, startup_timeout=spec['timeout'],
        command_timeout=spec['timeout'], collective_timeout=spec['timeout'], total_timeout=spec['timeout'],
        cancellation_event=cancellation_event)
    with service:
        return finite_json(service.command(operation)['results'][0])


def enabled_custom(config):
    settings = config['metrics']
    return {name: spec for name, spec in settings['custom'].items()
            if name not in settings['disable'] and settings['overrides'].get(name, {}).get('enabled', True)}


def prepare_custom(config):
    """Resolve trusted factory descriptions in workers before creating an attempt."""
    descriptions = {}
    for name, spec in enabled_custom(config).items():
        description = invoke(spec, 'describe')
        if spec['mode'] == 'scalar' and description['descriptor']['kind'] != 'scalar':
            raise ValueError('Primitive update metrics require a scalar descriptor')
        descriptions[name] = description
    config['_metric_runtime'] = descriptions


class ScalarMetrics:
    """Lossy, bounded submission; factory work never runs on the training thread.

    At most one job per metric is outstanding, including unconsumed results.
    One background dispatcher serializes disposable CPU workers. Deadlines begin
    at admission, including queue time, so overload cannot accumulate stale work.
    Results retain their source context; callers publish them at that step.
    Only final ``close`` waits for observations or worker cleanup.
    """
    def __init__(self, config):
        self.specs = {name: spec for name, spec in enabled_custom(config).items() if spec['mode'] == 'scalar'}
        self.descriptions = config.get('_metric_runtime', {})
        self.disabled = {}
        self._pending = set()
        self._jobs = queue.Queue(maxsize=MAX_CUSTOM_METRICS)
        self._results = queue.Queue(maxsize=MAX_CUSTOM_METRICS)
        self._cancel = threading.Event()
        self._thread = None
        self._closed = False
        self._last_deadline = 0.0

    def start(self):
        """Start before the update loop; no process is created on this thread."""
        if self._closed:
            raise RuntimeError('Scalar metrics are closed')
        if self.specs and self._thread is None:
            self._thread = threading.Thread(target=self._dispatch,
                name='hypergan-scalar-metrics', daemon=True)
            self._thread.start()

    def _dispatch(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            name, spec, inputs, context, deadline = job
            try:
                if self._cancel.is_set():
                    raise RuntimeError('Metric observation cancelled at attempt shutdown')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Metric deadline expired while queued')
                result = invoke(dict(spec, timeout=remaining), 'scalar',
                    expected=self.descriptions[name], inputs=inputs, context=context,
                    cancellation_event=self._cancel)
                outcome = {'context': context, 'metrics': {name: result['value']}, 'measurement_status': {}}
            except BaseException as error:
                reason = f'{type(error).__name__}: {error}'[:1000]
                outcome = {'context': context, 'metrics': {},
                    'measurement_status': {name: {'status': 'failed' if spec['on_error'] == 'fail' else 'disabled',
                                                  'reason': reason}}}
            self._results.put_nowait((name, outcome))

    def evaluate(self, row, context):
        if self._closed:
            raise RuntimeError('Scalar metrics are closed')
        if self.specs and self._thread is None:
            raise RuntimeError('Scalar metrics must be started before training')
        statuses = {}
        for name, spec in self.specs.items():
            if name in self.disabled:
                statuses[name] = {'status': 'disabled', 'reason': self.disabled[name]}
                continue
            if context['step'] % spec['every_steps']:
                continue
            if name in self._pending:
                statuses[name] = {'status': 'dropped', 'reason': 'Previous observation is still outstanding'}
                continue
            inputs = {key: row[path.split('.', 1)[1]] for key, path in spec['inputs'].items()}
            # Copy only primitive, bounded data, never a live trainer/tensor.
            inputs, source = finite_json(inputs), finite_json(context)
            deadline = time.monotonic() + spec['timeout']
            self._jobs.put_nowait((name, spec, inputs, source, deadline))
            self._pending.add(name)
            self._last_deadline = max(self._last_deadline, deadline)
            statuses[name] = {'status': 'queued'}
        return {}, statuses

    def poll(self):
        outcomes = []
        failure = None
        for _ in range(MAX_CUSTOM_METRICS):
            try:
                name, outcome = self._results.get_nowait()
            except queue.Empty:
                break
            self._pending.remove(name)
            status = outcome['measurement_status'].get(name)
            if status:
                self.disabled[name] = status['reason']
                if status['status'] == 'failed':
                    failure = RuntimeError(f"Required metric {name} failed: {status['reason']}")
            outcomes.append(outcome)
        if failure is not None:
            # Attach evidence so a controller may publish the failure before
            # propagating it. It must never complete the run successfully.
            failure.metric_outcomes = outcomes
            raise failure
        return outcomes

    def close(self, *, drain=True):
        if not self._closed:
            self._closed = True
            if not drain:
                self._cancel.set()
            if self._thread is not None:
                # All <=32 outstanding observations already have deadlines.
                # A sentinel may need one slot while the worker consumes its
                # first job; waiting is confined to terminal cleanup.
                self._jobs.put(None)
                timeout = max(0.0, self._last_deadline - time.monotonic()) + 8.0 if drain else 8.0
                self._thread.join(timeout)
                if self._thread.is_alive():
                    self._cancel.set()
                    self._thread.join(8.0)
                    if self._thread.is_alive():
                        raise RuntimeError('Metric dispatcher has not completed worker cleanup')
        return self.poll() if drain else []
