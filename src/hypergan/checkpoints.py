"""Full trusted training checkpoints, separate from inference bundles."""
from collections import defaultdict
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import re
import shutil
import uuid

from .run_state import atomic_json, sync_directory, validate_event_boundary

SCHEMA = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_rng():
    import numpy as np
    import torch
    numpy = np.random.get_state()
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]]}
    # CPU execution must not initialize CUDA merely to observe random state.
    if torch.cuda.is_initialized():
        result['cuda'] = torch.cuda.get_rng_state_all()
        result['cuda_device'] = torch.cuda.current_device()
    return result


def restore_rng(state):
    import numpy as np
    import torch
    if 'cuda' in state:
        if (not torch.cuda.is_available() or type(state['cuda']) is not list
                or len(state['cuda']) != torch.cuda.device_count()
                or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != 'cpu' for value in state['cuda'])
                or type(state.get('cuda_device')) is not int or not 0 <= state['cuda_device'] < torch.cuda.device_count()):
            raise ValueError('Checkpoint CUDA RNG device inventory differs from the runtime')
        torch.cuda.set_rng_state_all(state['cuda'])
        torch.cuda.set_device(state['cuda_device'])
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))


def data_contract(data, specification):
    has_save, has_load = callable(getattr(data, 'state_dict', None)), callable(getattr(data, 'load_state_dict', None))
    if has_save != has_load:
        raise ValueError('Data recovery requires both state_dict and load_state_dict')
    builtin = specification['factory'] in ('gaussian_grid', 'paired_linear')
    stateless = builtin or getattr(data, 'resume_stateless', False) is True
    identity_fn = getattr(data, 'resume_identity', None)
    identity = identity_fn() if callable(identity_fn) else {'specification': specification}
    # Require a stable portable descriptor, not repr(object) or pickle identity.
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    return {'supported': bool(has_save or stateless), 'stateful': has_save, 'identity': identity}


def _portable(value):
    import torch
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, torch.Tensor):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _portable(item)
        return
    if isinstance(value, dict) and all(type(key) in (str, int) for key in value):
        for item in value.values():
            _portable(item)
        return
    raise ValueError(f'Checkpoint state must contain tensors and primitive containers, got {type(value).__name__}')


def trainer_state(trainer, last_batch):
    rng = capture_rng()
    try:
        modules = ('graph', 'prior', 'ema_graph', 'ema_prior')
        state = {name: getattr(trainer, name).state_dict() for name in modules}
        state.update(optimizers=[trainer.opt_g.state_dict(), trainer.opt_d.state_dict()],
                     base_lrs=trainer.base_lrs, step=trainer.step,
                     streams={name: stream.get_state() for name, stream in trainer.streams.items()},
                     rng=rng, data=trainer.data.state_dict() if callable(getattr(trainer.data, 'state_dict', None)) else None,
                     buffers={name: dict(getattr(trainer, name).named_buffers()) for name in modules},
                     trainable={name: {key: value.requires_grad for key, value in getattr(trainer, name).named_parameters()} for name in modules},
                     modes={name: {key: module.training for key, module in getattr(trainer, name).named_modules()} for name in modules},
                     last_batch=last_batch)
        _portable(state)
        return state
    finally:
        restore_rng(rng)


def _restore_trainer(trainer, state):
    import numpy as np
    import torch
    required = {'graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                'step', 'streams', 'rng', 'data', 'modes', 'buffers', 'trainable', 'last_batch'}
    if not isinstance(state, dict) or not required <= set(state) <= required | {'aggregation'}:
        raise ValueError('Checkpoint training state fields are incomplete or unsupported')
    if getattr(trainer, 'device', torch.device('cpu')).type == 'cuda':
        rng = state['rng']
        if not isinstance(rng, dict) or 'cuda' not in rng or 'cuda_device' not in rng:
            raise ValueError('CUDA training checkpoint is missing complete CUDA RNG state')
        if type(rng['cuda']) is not list or len(rng['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG device inventory differs from the runtime')
        for index, saved in enumerate(rng['cuda']):
            if (not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8 or saved.device.type != 'cpu'
                    or saved.shape != torch.cuda.get_rng_state(index).shape):
                raise ValueError('Checkpoint CUDA RNG tensor is incompatible')
    if type(state['step']) is not int or not 0 <= state['step'] <= trainer.config['training']['steps']:
        raise ValueError('Checkpoint step is outside the configured training schedule')
    if len(state['optimizers']) != 2 or len(state['base_lrs']) != 2:
        raise ValueError('Checkpoint requires both optimizer states and base learning rates')
    if set(state['streams']) != set(trainer.streams):
        raise ValueError('Checkpoint named RNG streams differ from trainer')
    # Validate consumed RNG streams before any model or optimizer is restored.
    # Local generators leave the process-global streams untouched on failure.
    for name, stream in trainer.streams.items():
        torch.Generator(device=stream.device).set_state(state['streams'][name])
    torch.Generator(device='cpu').set_state(state['rng']['torch'])
    random.Random().setstate(state['rng']['python'])
    numpy_state = state['rng']['numpy']
    np.random.RandomState().set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if set(state['modes']) != {'graph', 'prior', 'ema_graph', 'ema_prior'}:
        raise ValueError('Checkpoint module mode inventory differs from trainer')
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        module = getattr(trainer, name)
        if set(state[name]) != set(module.state_dict()):
            raise ValueError('Checkpoint model state inventory differs from trainer')
        for key, value in module.state_dict().items():
            if isinstance(value, torch.Tensor) and (not isinstance(state[name][key], torch.Tensor) or state[name][key].shape != value.shape or state[name][key].dtype != value.dtype):
                raise ValueError('Checkpoint model tensor shape or dtype differs from trainer')
        if set(state['buffers'][name]) != dict(module.named_buffers()).keys() or set(state['trainable'][name]) != dict(module.named_parameters()).keys():
            raise ValueError('Checkpoint buffer/parameter inventory differs from trainer')
        for key, value in module.named_buffers():
            saved = state['buffers'][name][key]
            if saved.shape != value.shape or saved.dtype != value.dtype:
                raise ValueError('Checkpoint buffer shape or dtype differs from trainer')
        if set(state['modes'][name]) != dict(getattr(trainer, name).named_modules()).keys():
            raise ValueError('Checkpoint nested module mode inventory differs from trainer')
        if any(type(value) is not bool for value in (*state['modes'][name].values(), *state['trainable'][name].values())):
            raise ValueError('Checkpoint module modes and trainability must be boolean')
    for optimizer, saved, rates in zip((trainer.opt_g, trainer.opt_d), state['optimizers'], state['base_lrs']):
        # The recipe's optimizers keep their regularization state (EMA critic,
        # LR record, latent-damping history) under 'regularizer'.
        if set(saved) != set(optimizer.state_dict()) or len(saved['param_groups']) != len(optimizer.param_groups) or len(rates) != len(optimizer.param_groups):
            raise ValueError('Checkpoint optimizer parameter groups differ from trainer')
        for actual, prior in zip(optimizer.param_groups, saved['param_groups']):
            if len(actual['params']) != len(prior['params']):
                raise ValueError('Checkpoint optimizer parameter inventory differs from trainer')
        validate = getattr(optimizer, 'validate_state_dict', None)
        if validate is not None:
            validate(saved)
    if state['base_lrs'] != trainer.base_lrs:
        raise ValueError('Checkpoint base learning rates differ from original configuration')
    for name in ('graph', 'prior', 'ema_graph', 'ema_prior'):
        module = getattr(trainer, name)
        module.load_state_dict(state[name], strict=True)
        with torch.no_grad():
            for key, buffer in module.named_buffers():
                buffer.copy_(state['buffers'][name][key])
        for key, parameter in module.named_parameters():
            parameter.requires_grad_(state['trainable'][name][key])
        for key, child in module.named_modules():
            child.training = state['modes'][name][key]
    for optimizer, saved in zip((trainer.opt_g, trainer.opt_d), state['optimizers']):
        optimizer.load_state_dict(saved)
    trainer.base_lrs = state['base_lrs']
    trainer.step = state['step']
    for name, stream in trainer.streams.items():
        stream.set_state(state['streams'][name])
    if state['data'] is not None:
        trainer.data.load_state_dict(state['data'])
    if 'aggregation' in state:
        strategy = getattr(trainer, 'aggregation_strategy', getattr(trainer, 'aggregation', None))
        if strategy is not None and hasattr(strategy, 'load_state_dict') and callable(strategy.load_state_dict):
            strategy.load_state_dict(state['aggregation'])
    # Constructors and custom load hooks can consume global randomness.
    restore_rng(state['rng'])
    from .recipes import move_tensors
    return move_tensors(state['last_batch'], getattr(trainer, 'device', 'cpu'))


def restore_trainer(trainer, state):
    try:
        return _restore_trainer(trainer, state)
    except (KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f'Invalid training checkpoint state: {exc}') from exc


class _HashingWriter:
    """Digest bytes as torch.save streams them, so the payload is never re-read."""
    def __init__(self, output):
        self.output, self.digest = output, hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.output.write(data)

    def flush(self):
        self.output.flush()


_PENDING_CHECKPOINT = re.compile(r'\.pending-.+-step-\d+-[0-9a-f]{12}')


def _sweep_pending(root):
    """Remove pending checkpoint writes that a killed writer never cleaned up.

    The caller is the run-lock owner, so no other writer is mid-checkpoint.
    """
    for path in root.iterdir():
        if _PENDING_CHECKPOINT.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def write_checkpoint(run_dir, trainer, last_batch, metadata):
    import torch
    root = Path(run_dir) / 'checkpoints'
    root.mkdir(exist_ok=True)
    sync_directory(root.parent)
    _sweep_pending(root)
    name = f"{metadata['attempt_id']}-step-{trainer.step:08d}-{uuid.uuid4().hex[:12]}"
    temporary, target = root / ('.pending-' + name), root / name
    temporary.mkdir()
    try:
        state = trainer_state(trainer, last_batch)
        with (temporary / 'state.pt').open('wb') as output:
            writer = _HashingWriter(output)
            torch.save(state, writer)
            output.flush()
            os.fsync(output.fileno())
        digest = writer.digest.hexdigest()
        info = dict(metadata, schema_version=SCHEMA, kind='hypergan-training-checkpoint', step=trainer.step, state_sha256=digest)
        atomic_json(temporary / 'manifest.json', info)
        sync_directory(temporary)
        temporary.rename(target)
        sync_directory(root)
        atomic_json(root / 'latest.json', {'schema_version': SCHEMA, 'checkpoint': target.name, 'step': trainer.step})
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_checkpoint(run_dir, checkpoint=None):
    import torch
    root = (Path(run_dir) / 'checkpoints').resolve()
    if checkpoint is None:
        pointer = root / 'latest.json'
        if not pointer.exists():
            raise ValueError('No full training checkpoint exists; an inference model is not resumable')
        latest = json.loads(pointer.read_text())
        if not isinstance(latest, dict) or latest.get('schema_version') != SCHEMA or not isinstance(latest.get('checkpoint'), str):
            raise ValueError('Invalid training checkpoint pointer')
        checkpoint = root / latest['checkpoint']
    target = Path(checkpoint)
    if not target.is_absolute():
        choices = (target, Path(run_dir) / target, root / target)
        target = next((choice for choice in choices if choice.is_dir()), root / target)
    target = target.resolve()
    if not target.is_relative_to(root) or not target.is_dir() or target.name.startswith('.'):
        raise ValueError('Checkpoint must be a completed checkpoint directory inside this run')
    info = json.loads((target / 'manifest.json').read_text())
    required = {'schema_version', 'kind', 'run_id', 'attempt_id', 'step', 'config', 'config_sha256', 'runtime', 'data_contract', 'implementation', 'state_sha256', 'next_sample_sequence'}
    if not isinstance(info, dict) or not required.issubset(info):
        raise ValueError('Invalid training checkpoint metadata')
    if info.get('schema_version') != SCHEMA or info.get('kind') != 'hypergan-training-checkpoint':
        raise ValueError('Unsupported full training checkpoint schema')
    from .checkpoint_compatibility import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(info)
    validate_event_boundary(run_dir, info)
    if file_sha256(target / 'state.pt') != info['state_sha256']:
        raise ValueError('Training checkpoint digest mismatch')
    try:
        state = torch.load(target / 'state.pt', map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError('Invalid training checkpoint payload; safe tensor loading failed') from exc
    if not isinstance(state, dict) or 'step' not in state:
        raise ValueError('Invalid training checkpoint state fields')
    if state['step'] != info['step']:
        raise ValueError('Checkpoint step metadata mismatch')
    return target, info, state


def read_checkpoint(run_dir, checkpoint=None):
    try:
        return _read_checkpoint(run_dir, checkpoint)
    except (KeyError, TypeError, OSError, EOFError) as exc:
        raise ValueError(f'Invalid or incomplete training checkpoint: {exc}') from exc


from .model_file import DEFAULT_CHECKPOINT_POLICY, normalize_checkpoint_policy


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _durable_checkpoints(ckpt_root):
    """Completed checkpoint directories with their step and owning attempt, oldest first."""
    checkpoints = []
    if not Path(ckpt_root).is_dir():
        return checkpoints
    for d in Path(ckpt_root).iterdir():
        if not d.is_dir() or d.name.startswith('.'):
            continue
        info = _read_json(d / 'manifest.json') or {}
        step = info.get('step')
        if type(step) is not int:
            m = re.search(r'-step-(\d+)-', d.name)
            step = int(m.group(1)) if m else None
        if step is None:
            continue
        attempt = info.get('attempt_id')
        checkpoints.append({'name': d.name, 'path': d, 'step': step,
                            'attempt_id': attempt if isinstance(attempt, str) else None})
    checkpoints.sort(key=lambda c: (c['step'], c['name']))
    return checkpoints


def _run_policy(run_dir):
    manifest = _read_json(Path(run_dir) / 'manifest.json') or {}
    config = manifest.get('config')
    return config.get('checkpoints') if isinstance(config, dict) else None


def _rank_best(run_dir, evaluations, policy=None):
    from .summary import rank_evaluations
    return rank_evaluations(evaluations, _run_policy(run_dir) if policy is None else policy)


def _checkpoints_for_records(checkpoints, records):
    """Checkpoints holding each evaluated snapshot: same step and, when both are known, same attempt."""
    matches = {}
    for record in records:
        attempt = record.get('attempt_id')
        found = [c for c in checkpoints if c['step'] == record['step']
                 and (attempt is None or c['attempt_id'] is None or c['attempt_id'] == attempt)]
        matches[id(record)] = sorted(found, key=lambda c: c['name'])
    return matches


def _live_branch(run_dir, checkpoints, tip):
    """Names of checkpoints on the lineage that ends at ``tip``, or None when unknown.

    A resumed attempt continues from its ``recovery_parent``; checkpoints of an
    ancestor attempt after the step it was resumed from belong to an abandoned
    branch. Each attempt manifest records its own parent; a missing link makes
    the lineage unknown rather than cutting the branch short.
    """
    if tip is None or tip['attempt_id'] is None:
        return None
    parents, known = {}, set()
    manifests = [Path(run_dir) / 'manifest.json']
    attempts = Path(run_dir) / 'attempts'
    if attempts.is_dir():
        manifests.extend(path / 'manifest.json' for path in attempts.iterdir() if path.is_dir())
    for path in manifests:
        info = _read_json(path) or {}
        if isinstance(info.get('attempt_id'), str):
            known.add(info['attempt_id'])
        parent = info.get('recovery_parent')
        if (isinstance(info.get('attempt_id'), str) and isinstance(parent, dict)
                and isinstance(parent.get('attempt_id'), str) and type(parent.get('step')) is int):
            parents[info['attempt_id']] = (parent['attempt_id'], parent['step'])
    branch, seen = set(), set()
    attempt, limit = tip['attempt_id'], tip['step']
    while attempt is not None and attempt not in seen:
        if attempt not in known:
            return None
        seen.add(attempt)
        branch.update(c['name'] for c in checkpoints
                      if c['attempt_id'] in (attempt, None) and c['step'] <= limit)
        attempt, limit = parents.get(attempt, (None, None))
    return branch


def resolve_checkpoint(run_dir: Path | str, target: str | int | None = "latest") -> Path:
    """Resolve a checkpoint target ('latest', 'best', integer step, or path) to a checkpoint directory."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory not found: {run_dir}")
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise ValueError(f"No checkpoints directory found in {run_dir}")

    target_str = str(target).strip() if target is not None else "latest"

    # 1. Target: "latest" (or None)
    if target is None or target_str in ("latest", "None", ""):
        latest_file = ckpt_root / "latest.json"
        if latest_file.is_file():
            try:
                latest = json.loads(latest_file.read_text(encoding="utf-8"))
                if isinstance(latest, dict) and latest.get("checkpoint"):
                    candidate = ckpt_root / latest["checkpoint"]
                    if candidate.is_dir():
                        return candidate.resolve()
            except Exception:
                pass
        # Fallback to max step on disk
        candidates = []
        for p in ckpt_root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                match = re.search(r"-step-(\d+)-", p.name)
                step_val = None
                if match:
                    step_val = int(match.group(1))
                else:
                    mf = p / "manifest.json"
                    if mf.is_file():
                        try:
                            m_info = json.loads(mf.read_text(encoding="utf-8"))
                            step_val = m_info.get("step")
                        except Exception:
                            pass
                if step_val is not None:
                    candidates.append((step_val, p))
        if not candidates:
            raise ValueError(f"No training checkpoints found in {ckpt_root}")
        candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        return candidates[0][1].resolve()

    # 2. Target: "best": the best-ranked primary-metric evaluation that has a checkpoint
    if target_str == "best":
        from .summary import collect_evaluations
        evaluations = collect_evaluations(run_dir)
        if not evaluations:
            raise ValueError("No evaluation records available to determine best checkpoint; pass explicit step or latest")
        metric, _, ranked = _rank_best(run_dir, evaluations)
        if metric is None:
            raise ValueError("No primary metric defines the best checkpoint: set checkpoints.keep_best.metric "
                             "to an evaluated metric, or pass an explicit step or latest")
        by_record = _checkpoints_for_records(_durable_checkpoints(ckpt_root), ranked)
        for record in ranked:
            if by_record.get(id(record)):
                return by_record[id(record)][-1]['path'].resolve()
        raise ValueError(f"No checkpoint exists at any evaluated step of {metric} in {ckpt_root}")

    # 3. Target: STEP (integer or numeric digits)
    if isinstance(target, int) or target_str.isdigit():
        step = int(target)
        pattern = f"-step-{step:08d}-"
        matches = [p for p in ckpt_root.iterdir() if p.is_dir() and not p.name.startswith(".") and pattern in p.name]
        if not matches:
            fallback_pattern = f"-step-{step}-"
            matches = [p for p in ckpt_root.iterdir() if p.is_dir() and not p.name.startswith(".") and fallback_pattern in p.name]
        if not matches:
            for p in ckpt_root.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    mf = p / "manifest.json"
                    if mf.is_file():
                        try:
                            m_info = json.loads(mf.read_text(encoding="utf-8"))
                            if m_info.get("step") == step:
                                matches.append(p)
                        except Exception:
                            pass
        if not matches:
            raise ValueError(f"Checkpoint for step {step} not found in {ckpt_root}")
        matches.sort(key=lambda p: p.name, reverse=True)
        return matches[0].resolve()

    # 4. Target: PATH
    cand_path = Path(target_str)
    choices = [cand_path, run_dir / cand_path, ckpt_root / cand_path]
    for c in choices:
        if c.is_file() and c.name in ("state.pt", "model.pt"):
            return c.parent.resolve()
        if c.is_dir() and ((c / "state.pt").is_file() or (c / "manifest.json").is_file() or (c / "model.pt").is_file()):
            return c.resolve()

    raise ValueError(f"Checkpoint path not found: {target}")


def extract_ema_bundle(checkpoint_dir: Path | str, manifest: dict | None = None) -> dict:
    """Extract an in-memory EMA inference bundle conforming to kind='ema-inference' from a checkpoint."""
    import torch
    from .config import config_values, sampling_bindings, evaluation_bindings
    path = Path(checkpoint_dir).resolve()
    if path.is_file() and path.name == "model.pt":
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(bundle, dict) and bundle.get("kind") == "ema-inference":
            return bundle
        state_file = path
        ckpt_dir = path.parent
    elif path.is_file() and path.name == "state.pt":
        state_file = path
        ckpt_dir = path.parent
    elif path.is_dir():
        ckpt_dir = path
        state_file = ckpt_dir / "state.pt"
        if not state_file.is_file() and (ckpt_dir / "model.pt").is_file():
            bundle = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
            if isinstance(bundle, dict) and bundle.get("kind") == "ema-inference":
                return bundle
    else:
        raise ValueError(f"Checkpoint path not found: {checkpoint_dir}")

    if not state_file.is_file():
        raise ValueError(f"Checkpoint state file missing: {state_file}")

    if manifest is None:
        manifest_file = ckpt_dir / "manifest.json"
        if not manifest_file.is_file():
            if (ckpt_dir.parent / "manifest.json").is_file():
                manifest_file = ckpt_dir.parent / "manifest.json"
            else:
                raise ValueError(f"Checkpoint manifest missing: {ckpt_dir / 'manifest.json'}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    state = torch.load(state_file, map_location="cpu", weights_only=True)

    if not isinstance(state, dict):
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")
    if "ema_graph" not in state or "ema_prior" not in state:
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")
    if state["ema_graph"] is None or state["ema_prior"] is None:
        raise ValueError("Checkpoint state missing EMA weights; cannot construct EMA inference bundle")

    config = manifest.get("config")
    if not isinstance(config, dict) or "components" not in config:
        raise ValueError("Checkpoint manifest missing valid configuration components")

    needed_components = set()
    called_components = set()

    def include(name):
        if name in called_components or name not in config["components"]:
            return
        called_components.add(name)
        needed_components.add(name)
        comp_spec = config["components"][name]
        if "reuse" in comp_spec:
            needed_components.add(comp_spec["reuse"])
        for input_path in comp_spec.get("inputs", {}).values():
            if isinstance(input_path, str) and input_path.startswith("components."):
                include(input_path.split(".")[1])

    include("generator")
    for binding in sampling_bindings(config.get("sampling", {})):
        if binding.startswith("components."):
            include(binding.split(".")[1])
    sampling_components = set(called_components)
    if isinstance(config.get("metrics"), dict) and isinstance(config["metrics"].get("custom"), dict):
        for binding in evaluation_bindings(config):
            if binding.startswith("components."):
                include(binding.split(".")[1])

    specs = {name: spec for name, spec in config["components"].items() if name in needed_components}
    needed_inputs = {
        path.split(".")[1]
        for name in sampling_components
        for path in specs[name].get("inputs", {}).values()
        if isinstance(path, str) and path.startswith("batch.")
    }

    raw_ema_graph = state["ema_graph"]
    raw_buffers = state.get("buffers", {}).get("ema_graph", {}) if isinstance(state.get("buffers"), dict) else {}

    model_states = {}
    model_buffers = {}
    for name, spec in specs.items():
        if "reuse" not in spec:
            prefix = f"models.{name}."
            matching_weights = {
                k[len(prefix):]: v for k, v in raw_ema_graph.items() if k.startswith(prefix)
            }
            if not matching_weights and name in raw_ema_graph and isinstance(raw_ema_graph[name], dict):
                matching_weights = raw_ema_graph[name]
            model_states[name] = matching_weights

            matching_buffers = {
                k[len(prefix):]: v for k, v in raw_buffers.items() if k.startswith(prefix)
            }
            if not matching_buffers and name in raw_buffers and isinstance(raw_buffers[name], dict):
                matching_buffers = raw_buffers[name]
            model_buffers[name] = matching_buffers

    prior_state = state["ema_prior"]
    prior_buffers = state.get("buffers", {}).get("ema_prior", {}) if isinstance(state.get("buffers"), dict) else {}

    batch = state.get("last_batch") or {}
    example_inputs = {k: v for k, v in batch.items() if k in needed_inputs} if isinstance(batch, dict) else {}

    identity = {
        "run_id": manifest.get("run_id", ""),
        "attempt_id": manifest.get("attempt_id", ""),
        "source_step": state.get("step", manifest.get("step", 0)),
    }
    if "identity" in manifest and isinstance(manifest["identity"], dict):
        identity.update(manifest["identity"])

    bundle = {
        "schema_version": 1,
        "kind": "ema-inference",
        "resume_supported": False,
        "step": state.get("step", manifest.get("step", 0)),
        "config": config_values(config),
        "components": specs,
        "model_states": model_states,
        "prior": prior_state,
        "example_inputs": example_inputs,
        "model_buffers": model_buffers,
        "prior_buffers": prior_buffers,
        "identity": identity,
    }
    return bundle


def export_checkpoint_bundle(checkpoint_dir: Path | str, output_dir: Path | str) -> tuple[Path, Path]:
    """Export an EMA inference bundle (model.pt + model.sha256) to output directory or file path."""
    import torch
    ckpt_dir = Path(checkpoint_dir).resolve()
    bundle = extract_ema_bundle(ckpt_dir)

    dest = Path(output_dir).resolve()
    if dest.is_dir() or (not dest.exists() and dest.suffix == ""):
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / "model.pt"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    temp_pt = dest.with_suffix(".pt.tmp") if dest.suffix == ".pt" else dest.with_name(dest.name + ".tmp")
    with temp_pt.open("wb") as stream:
        torch.save(bundle, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp_pt.replace(dest)

    digest = file_sha256(dest)
    checksum_path = dest.with_name("model.sha256") if dest.name == "model.pt" else dest.with_suffix(".sha256")
    temp_sha = checksum_path.with_suffix(".sha256.tmp") if checksum_path.suffix == ".sha256" else checksum_path.with_name(checksum_path.name + ".tmp")
    with temp_sha.open("w", encoding="utf-8") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp_sha.replace(checksum_path)
    sync_directory(dest.parent)
    return dest, checksum_path


def _prune_report(run_dir, dry_run, retained=(), deleted=()):
    retained, deleted = list(retained), list(deleted)
    reclaimed = sum(item.get('size', 0) for item in deleted)
    return {
        'run_dir': str(run_dir),
        'dry_run': dry_run,
        'retained_count': len(retained),
        'deleted_count': len(deleted),
        'retained': retained,
        'deleted': deleted,
        'reclaimed_bytes': reclaimed,
        # aliases:
        'pruned': deleted,
        'pruned_count': len(deleted),
        'bytes_reclaimed': reclaimed,
    }


def prune_checkpoints(
    run_dir: Path | str,
    policy: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Evaluate and apply checkpoint retention policy to a run directory adhering to Invariants S1-S5.

    ``keep_last`` counts checkpoints on the live branch, the lineage that ends
    at ``latest.json``. After a resume from an older checkpoint, each abandoned
    attempt keeps its newest checkpoint, so the old branch stays resumable;
    its other checkpoints keep only milestone, best, step-zero and
    active-evaluation protection.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        return _prune_report(run_dir, dry_run)

    ckpt_root = run_dir / 'checkpoints'
    if not ckpt_root.is_dir():
        child_runs = [d for d in run_dir.iterdir() if d.is_dir() and (d / 'checkpoints').is_dir()]
        total_retained, total_deleted = [], []
        for crun in sorted(child_runs, key=lambda p: p.name):
            sub_res = prune_checkpoints(crun, policy=policy, dry_run=dry_run)
            total_retained.extend(sub_res.get('retained', []))
            total_deleted.extend(sub_res.get('deleted', []))
        return _prune_report(run_dir, dry_run, total_retained, total_deleted)

    # 1. Resolve Policy
    if policy is None:
        policy = _run_policy(run_dir)
    policy = normalize_checkpoint_policy(policy)

    # 2. Discover durable checkpoints
    checkpoints = _durable_checkpoints(ckpt_root)
    for c in checkpoints:
        size = 0
        try:
            for item in c['path'].rglob('*'):
                if item.is_file():
                    size += item.stat().st_size
        except OSError:
            pass
        c['size'] = size

    if len(checkpoints) <= 1:
        retained = [{'step': c['step'], 'name': c['name'], 'reasons': ['single_checkpoint']} for c in checkpoints]
        return _prune_report(run_dir, dry_run, retained)

    # 3. Retain map: name -> set of reasons
    retained_reasons = defaultdict(set)

    # S1: Never delete latest. Without a pointer, the highest step stands in
    # and the branch is unknown, so every checkpoint counts as live.
    latest_name = (_read_json(ckpt_root / 'latest.json') or {}).get('checkpoint')
    tip = next((c for c in checkpoints if c['name'] == latest_name), None)
    branch = _live_branch(run_dir, checkpoints, tip)
    max_step = max(c['step'] for c in checkpoints)
    for c in checkpoints:
        if c is tip or (tip is None and c['step'] == max_step):
            retained_reasons[c['name']].add('latest')
    live = [c for c in checkpoints if branch is None or c['name'] in branch]
    live_names = {c['name'] for c in live}

    # S2: Pin active interval evaluations
    pinned_steps = set()
    m_data = _read_json(run_dir / 'manifest.json') or {}
    schedule = m_data.get('evaluation_schedule')
    for item in (schedule.values() if isinstance(schedule, dict) else []):
        if isinstance(item, dict) and item.get('status') == 'running' and type(item.get('source_step')) is int:
            pinned_steps.add(item['source_step'])

    eval_root = run_dir / 'metrics' / 'evaluations'
    if eval_root.is_dir():
        for r_file in eval_root.glob('*/receipt.json'):
            receipt = _read_json(r_file) or {}
            if receipt.get('status') == 'running':
                s = receipt.get('source_step') if receipt.get('source_step') is not None else receipt.get('step')
                if type(s) is int:
                    pinned_steps.add(s)

    for c in checkpoints:
        if c['step'] in pinned_steps:
            retained_reasons[c['name']].add('active_evaluation')

    # S3: Retain step 0 (initial state)
    for c in checkpoints:
        if c['step'] == 0:
            retained_reasons[c['name']].add('step_zero')

    # S4: Keep milestones (keep_every)
    keep_every = policy.get('keep_every')
    if keep_every and keep_every > 0:
        for c in checkpoints:
            if c['step'] % keep_every == 0:
                retained_reasons[c['name']].add(f'milestone(step={c["step"]})')

    # S5: Keep the best checkpoints of the run's one primary metric, ranked
    # exactly as summary "best" and resolve_checkpoint('best') rank them.
    keep_best = policy.get('keep_best')
    if isinstance(keep_best, dict) and keep_best.get('count', 0) > 0:
        from .summary import collect_evaluations
        metric, _, ranked = _rank_best(run_dir, collect_evaluations(run_dir), policy)
        matches = _checkpoints_for_records(checkpoints, ranked)
        remaining = keep_best['count']
        for record in ranked:
            names = [c['name'] for c in matches[id(record)] if f'best({metric})' not in retained_reasons[c['name']]]
            if not names:
                continue
            for name in names:
                retained_reasons[name].add(f'best({metric})')
            remaining -= 1
            if remaining == 0:
                break

    # Keep last N checkpoints of the live branch
    keep_last = policy.get('keep_last', 3)
    if keep_last > 0:
        last_steps = set(sorted({c['step'] for c in live})[-keep_last:])
        recovery_ckpt = (m_data.get('recovery_parent') or {}).get('checkpoint_id')
        for c in checkpoints:
            if (c['name'] in live_names and c['step'] in last_steps) or (recovery_ckpt and c['name'] == recovery_ckpt):
                retained_reasons[c['name']].add('keep_last')

    # The newest checkpoint of each attempt off the live branch stays as that
    # abandoned branch's tip; it no longer counts toward keep_last.
    abandoned = {}
    for c in checkpoints:
        if c['name'] not in live_names:
            abandoned[c['attempt_id']] = c
    for c in abandoned.values():
        retained_reasons[c['name']].add('abandoned_tip')

    # S3: Require a newer durable checkpoint before deleting older. Off the
    # live branch, the retained live tip is that newer durable checkpoint.
    deleted_candidates = []
    for c in checkpoints:
        if c['name'] in retained_reasons:
            continue
        has_newer = c['name'] not in live_names or any(other['step'] > c['step'] for other in live)
        if not has_newer:
            retained_reasons[c['name']].add('s3_no_newer_durable')
        else:
            deleted_candidates.append(c)

    # Execute deletion if not dry_run
    if not dry_run:
        for c in deleted_candidates:
            target_dir = c['path']
            (target_dir / 'state.pt').unlink(missing_ok=True)
            (target_dir / 'manifest.json').unlink(missing_ok=True)
            shutil.rmtree(target_dir, ignore_errors=True)
        if deleted_candidates:
            sync_directory(ckpt_root)

    retained = [
        {'step': c['step'], 'name': c['name'], 'reasons': sorted(list(retained_reasons[c['name']]))}
        for c in checkpoints if c['name'] in retained_reasons
    ]
    deleted = [
        {'step': c['step'], 'name': c['name'], 'size': c['size']}
        for c in deleted_candidates
    ]
    return _prune_report(run_dir, dry_run, retained, deleted)
