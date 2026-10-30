"""Public train/resume routing with torch-free validation before local viewing.

Recipes select the native device. Explicit profiles select fixed-topology
replication; deadlines are attempt policy, never numerical checkpoint identity.
Use a Python main guard when selecting a replicated profile.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .config import config_values, fingerprint, load_config, resolve_config, resume_compatible
from .model_file import ConfigError
from .previews import sample_name
from .execution_profiles import load_execution_profile, profile_from_execution, resolve_execution_profile
from .main_guard import require_main_guard
from .metric_plugins import enabled_custom

PROFILE_NAMES = ('cpu-single', 'cpu-replicated-gloo', 'cuda-replicated-nccl')
SERVICE_TIMEOUTS = ('startup_timeout', 'command_timeout', 'collective_timeout',
                    'total_timeout', 'observer_timeout', 'preview_timeout')
MAX_METADATA_BYTES = 16 * 1024 * 1024


def _read_metadata(path):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_METADATA_BYTES + 1)
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError('Run/checkpoint metadata exceeds its byte bound')
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('Run/checkpoint metadata must be a JSON object')
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _same_topology(exec_a, exec_b):
    if exec_a is None and exec_b is None:
        return True
    if exec_a is None or exec_b is None:
        return False
    keys = set(exec_a.keys()) | set(exec_b.keys())
    for k in keys:
        if k in ('node_rank', 'rendezvous'):
            continue
        if not _same(exec_a.get(k), exec_b.get(k)):
            return False
    return True


def _differences(current, original):
    """Name the resolved top-level sections that differ, including defaulted fields."""
    keys = sorted(set(current) | set(original))
    return [key for key in keys if not _same(current.get(key), original.get(key))]


def _training_config(config_path, steps=None):
    config = load_config(config_path)
    if steps is not None:
        # load_config re-resolves and keeps the retention, launch and source tables.
        config = load_config({**config, 'training': {**config['training'], 'steps': steps}})
    return config


def _profile(value, config):
    if value is None:
        return None
    if isinstance(value, str) and value in PROFILE_NAMES:
        return resolve_execution_profile({'schema_version': 1, 'execution': {'name': value}}, config)
    if isinstance(value, (str, Path)):
        return load_execution_profile(value, config)
    from .execution_preflight import _resolve_profile
    return _resolve_profile(value, config)


def _execution(profile):
    if profile is None or profile['execution']['name'] == 'cpu-single':
        return None
    return profile['execution']


def _policy(profile, service_policy):
    if service_policy is not None and not isinstance(service_policy, dict):
        raise ValueError('service_policy must be a dictionary of replicated deadlines')
    if _execution(profile) is None:
        if service_policy:
            raise ValueError('Service timeout options require a replicated execution profile')
        return None
    from .replicated_execution import _policy as resolve_policy
    return resolve_policy(profile, service_policy)


def _checkpoint(run_dir, checkpoint, manifest, execution, total_steps):
    """Pin a complete generation and reject routing/config conflicts without torch.

    The adapter validates all payload bytes, runtime, data and implementation
    during restore, under the run lock and before publishing a new attempt.
    """
    root = (run_dir / ('distributed-checkpoints' if execution else 'checkpoints')).resolve()
    pointer = None
    if checkpoint is None:
        if not (root / 'latest.json').exists():
            raise ValueError('No full training checkpoint exists; an inference model is not resumable')
        pointer = _read_metadata(root / 'latest.json')
        if type(pointer.get('schema_version')) is not int or pointer['schema_version'] != 1:
            raise ValueError('Unsupported training checkpoint pointer schema')
        if execution and pointer.get('kind') != 'hypergan-distributed-training-checkpoint':
            raise ValueError('Invalid distributed checkpoint pointer kind')
        name = pointer.get('checkpoint')
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Invalid training checkpoint pointer path')
        checkpoint = root / name
    target = Path(checkpoint)
    if not target.is_absolute():
        choices = (target, run_dir / target, root / target)
        target = next((choice for choice in choices if choice.is_dir()), root / target)
    if target.is_symlink():
        raise ValueError('Checkpoint must be an ordinary completed directory inside this run')
    target = target.resolve()
    if target.parent != root or target.name.startswith('.') or not target.is_dir():
        raise ValueError('Checkpoint must be a completed checkpoint directory inside this run')
    info = _read_metadata(target / 'manifest.json')
    if type(info.get('schema_version')) is not int or info['schema_version'] != 1:
        raise ValueError('Unsupported training checkpoint schema')
    if type(info.get('step')) is not int or not 0 <= info['step'] <= total_steps:
        raise ValueError('Checkpoint step is outside the original schedule')
    if pointer is not None and (type(pointer.get('step')) is not int or pointer['step'] != info['step']):
        raise ValueError('Checkpoint pointer step differs from generation')
    expected = 'hypergan-distributed-training-checkpoint' if execution else 'hypergan-training-checkpoint'
    if info.get('kind') != expected:
        raise ValueError(f'Selected execution requires checkpoint kind {expected}')
    if info.get('run_id') != manifest['run_id']:
        raise ValueError('Checkpoint belongs to a different run')
    identity = info.get('identity', {}) if execution else info
    if not isinstance(identity, dict):
        raise ValueError('Resume checkpoint configuration differs from the run manifest')
    if identity.get('config_sha256') != manifest['config_sha256']:
        saved_config = identity.get('config')
        if (not resume_compatible(manifest['config'], saved_config)
                or fingerprint(saved_config) != identity.get('config_sha256')):
            raise ValueError('Resume checkpoint configuration differs from the run manifest')
    from .checkpoint_compatibility import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(identity)
    if execution:
        topology = identity.get('topology', {})
        # The checkpoint writer excludes these per-launch fields from shared
        # numerical topology. Match that contract while retaining every batch,
        # accumulation, node-count and aggregation check.
        if not isinstance(topology, dict) or any(not _same(topology.get(key), value)
                for key, value in execution.items() if key not in ('node_rank', 'rendezvous')):
            raise ValueError('Resume checkpoint numerical execution identity differs from the run manifest')
    if info.get('event_boundary') is None:
        raise ValueError('Controller checkpoint requires a durable event boundary')
    from .run_state import validate_event_boundary
    validate_event_boundary(run_dir, info)
    return target


def detect_cluster_environment() -> dict:
    """Auto-detect multi-node cluster settings from Slurm and torchrun environment variables."""
    detected = {}
    # Nodes: SLURM_NNODES, NNODES
    nodes_raw = os.environ.get("SLURM_NNODES") or os.environ.get("NNODES")
    if nodes_raw:
        try:
            val = int(nodes_raw)
            if val >= 1:
                detected["nodes"] = val
        except ValueError:
            pass

    # Node Rank: SLURM_NODEID, NODE_RANK, GROUP_RANK
    rank_raw = os.environ.get("SLURM_NODEID") or os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK")
    if rank_raw:
        try:
            val = int(rank_raw)
            if val >= 0:
                detected["node_rank"] = val
        except ValueError:
            pass

    # Rendezvous: MASTER_ADDR and MASTER_PORT
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        master_port = os.environ.get("MASTER_PORT", "29500")
        detected["rendezvous"] = f"tcp://{master_addr}:{master_port}"

    return detected


def _is_follower(profile=None):
    if isinstance(profile, dict):
        exec_cfg = profile.get('execution', {})
        if isinstance(exec_cfg, dict) and (exec_cfg.get('node_rank', 0) > 0 or exec_cfg.get('rank', 0) > 0):
            return True
    for env_key in ('NODE_RANK', 'GROUP_RANK', 'SLURM_NODEID'):
        val = os.environ.get(env_key)
        if val is not None:
            try:
                if int(val) > 0:
                    return True
            except ValueError:
                pass
    return False


def _apply_execution_options(
    devices=None,
    nodes=None,
    node_rank=None,
    rendezvous=None,
    strategy=None,
    sync_every=None,
    config=None,
    profile=None,
    context_path=None,
):
    cli_nodes = nodes
    cli_node_rank = node_rank
    cli_rendezvous = rendezvous
    cli_strategy = strategy
    cli_sync_every = sync_every
    from .workspace import find_workspace, load_workspace, resolve_device_aliases
    env_ws = os.environ.get("HYPERGAN_WORKSPACE")
    source_p = config.get("_source_path") if isinstance(config, dict) else None
    ws_path = None
    for cand in (env_ws, source_p, context_path, Path.cwd()):
        if cand is not None:
            try:
                p = Path(cand)
                if p.is_file() and p.name == "hypergan.toml":
                    ws_path = p
                else:
                    ws_path = find_workspace(p)
                if ws_path:
                    break
            except Exception:
                pass
    ws_data = load_workspace(ws_path) if ws_path else {}

    # 4-tier precedence: CLI > Env > TOML > Defaults
    cluster_env = detect_cluster_environment()

    # 1. nodes
    if nodes is None:
        nodes = cluster_env.get("nodes")
    if nodes is None and isinstance(config, dict):
        nodes = config.get("run", {}).get("nodes")
    if nodes is None and ws_data:
        nodes = ws_data.get("run", {}).get("nodes")
    if nodes is None:
        nodes = 1
    else:
        nodes = int(nodes)

    # 2. node_rank
    if node_rank is None:
        node_rank = cluster_env.get("node_rank")
    if node_rank is None and isinstance(config, dict):
        node_rank = config.get("run", {}).get("node_rank")
    if node_rank is None and ws_data:
        node_rank = ws_data.get("run", {}).get("node_rank")
    if node_rank is None:
        node_rank = 0
    else:
        node_rank = int(node_rank)

    # 3. rendezvous
    if rendezvous is None:
        rendezvous = cluster_env.get("rendezvous")
    if rendezvous is None and isinstance(config, dict):
        rendezvous = config.get("run", {}).get("rendezvous")
    if rendezvous is None and ws_data:
        rendezvous = ws_data.get("run", {}).get("rendezvous")
    if rendezvous is None and nodes == 1:
        rendezvous = "file://"

    # 4. strategy
    if strategy is None and isinstance(config, dict):
        strategy = config.get("aggregation", {}).get("strategy", config.get("run", {}).get("strategy"))
    if strategy is None and ws_data:
        strategy = ws_data.get("aggregation", {}).get("strategy", ws_data.get("run", {}).get("strategy"))
    if strategy is None:
        strategy = "gradient_allreduce"

    # 5. sync_every
    if sync_every is None and isinstance(config, dict):
        sync_every = (
            config.get("aggregation", {}).get("sync_every_steps")
            or config.get("aggregation", {}).get("sync_every")
            or config.get("run", {}).get("sync_every_steps")
            or config.get("run", {}).get("sync_every")
        )
    if sync_every is None and ws_data:
        sync_every = (
            ws_data.get("aggregation", {}).get("sync_every_steps")
            or ws_data.get("aggregation", {}).get("sync_every")
            or ws_data.get("run", {}).get("sync_every_steps")
            or ws_data.get("run", {}).get("sync_every")
        )
    if sync_every is None:
        sync_every = 1
    else:
        sync_every = int(sync_every)

    # 6. devices
    if devices is None:
        if isinstance(config, dict):
            devices = config.get("run", {}).get("devices")
    if devices is None and ws_data:
        devices = ws_data.get("run", {}).get("devices")

    dev_list = []
    if devices is not None:
        resolved = resolve_device_aliases(devices, ws_data)
        dev_list = [d.strip() for d in (resolved if isinstance(resolved, list) else resolved.split(",")) if d.strip()]
        dev_list = [d[5:] if d.lower().startswith("cuda:") else d for d in dev_list]

    local_ranks = len(dev_list) if dev_list else 1

    if len(dev_list) > 1:
        _set_visible_devices(",".join(dev_list))
        if isinstance(config, dict) and "training" in config:
            config["training"]["device"] = "cuda"
    elif len(dev_list) == 1:
        if dev_list[0].lower() == "cpu":
            _set_visible_devices("", order=False)
            if isinstance(config, dict) and "training" in config:
                config["training"]["device"] = "cpu"
        else:
            _set_visible_devices(dev_list[0])
            if isinstance(config, dict) and "training" in config:
                config["training"]["device"] = "cuda"

    # Determine runtime name
    is_cpu = (
        (dev_list and dev_list[0].lower() == "cpu")
        or (isinstance(config, dict) and config.get("training", {}).get("device") == "cpu")
    )
    name = "cpu-replicated-gloo" if is_cpu else "cuda-replicated-nccl"

    # Dynamic profile synthesis if profile is None and multi-node or multi-GPU or DiLoCo is active
    is_replicated_needed = (nodes > 1 or local_ranks > 1 or strategy != "gradient_allreduce" or sync_every > 1)

    if profile is None and is_replicated_needed:
        agg_dict = {
            "strategy": strategy,
            "sync_every_steps": sync_every,
            "outer_lr": float(config.get("aggregation", {}).get("outer_lr", 1.0)) if isinstance(config, dict) else 1.0,
            "outer_momentum": float(config.get("aggregation", {}).get("outer_momentum", 0.0)) if isinstance(config, dict) else 0.0,
        }
        exec_dict = {
            "name": name,
            "world_size": local_ranks * nodes,
            "nodes": nodes,
            "node_rank": node_rank,
            "ranks_per_node": local_ranks,
            "accumulation_steps": 1,
            "aggregation": agg_dict,
        }
        if rendezvous is not None:
            exec_dict["rendezvous"] = rendezvous
        profile = {
            "schema_version": 1,
            "execution": exec_dict,
            "aggregation": agg_dict,
        }
        if isinstance(config, dict) and "training" in config:
            config["training"]["device"] = "cpu" if is_cpu else "cuda"
    elif profile is not None:
        has_cli_or_env_override = (
            cli_nodes is not None
            or cluster_env.get("nodes") is not None
            or cli_node_rank is not None
            or cluster_env.get("node_rank") is not None
            or cli_rendezvous is not None
            or cluster_env.get("rendezvous") is not None
            or len(dev_list) > 1
            or cli_strategy is not None
            or cli_sync_every is not None
        )
        if has_cli_or_env_override:
            if isinstance(profile, str) and profile in PROFILE_NAMES:
                profile = {
                    "schema_version": 1,
                    "execution": {
                        "name": profile,
                        "world_size": local_ranks * nodes,
                        "nodes": nodes,
                        "node_rank": node_rank,
                        "ranks_per_node": local_ranks,
                        "accumulation_steps": 1,
                        "aggregation": {
                            "strategy": strategy,
                            "sync_every_steps": sync_every,
                            "outer_lr": float(config.get("aggregation", {}).get("outer_lr", 1.0)) if isinstance(config, dict) else 1.0,
                            "outer_momentum": float(config.get("aggregation", {}).get("outer_momentum", 0.0)) if isinstance(config, dict) else 0.0,
                        },
                    },
                }
                if rendezvous is not None:
                    profile["execution"]["rendezvous"] = rendezvous
            elif isinstance(profile, dict) and "execution" in profile:
                profile = deepcopy(profile)
                if cli_nodes is not None or cluster_env.get("nodes") is not None:
                    profile["execution"]["nodes"] = nodes
                if cli_node_rank is not None or cluster_env.get("node_rank") is not None:
                    profile["execution"]["node_rank"] = node_rank
                if cli_rendezvous is not None or cluster_env.get("rendezvous") is not None:
                    profile["execution"]["rendezvous"] = rendezvous
                if cli_strategy is not None or cli_sync_every is not None:
                    agg = profile.get("aggregation") or profile["execution"].get("aggregation") or {}
                    agg = deepcopy(agg)
                    if cli_strategy is not None:
                        agg["strategy"] = cli_strategy
                    if cli_sync_every is not None:
                        agg["sync_every_steps"] = cli_sync_every
                    if "aggregation" in profile:
                        profile["aggregation"] = agg
                    if "aggregation" in profile.get("execution", {}):
                        profile["execution"]["aggregation"] = deepcopy(agg)

    return profile, devices


def _apply_devices(devices, config, profile, context_path):
    return _apply_execution_options(devices=devices, config=config, profile=profile, context_path=context_path)


def _set_visible_devices(value, *, order=True):
    """Mask GPUs for this process; refuse once CUDA has read the old mask."""
    if value and os.environ.get("CUDA_VISIBLE_DEVICES") != value:
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            raise ValueError(f"Cannot select devices {value!r}: CUDA is already initialized in this "
                             "process, so CUDA_VISIBLE_DEVICES no longer applies. Pass devices before any "
                             "CUDA use, or set CUDA_VISIBLE_DEVICES before starting Python.")
    if order:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = value


@dataclass(frozen=True)
class PreparedExecution:
    """Validated routing; running rechecks config and checkpoint under controller ownership."""
    operation: str
    config: dict
    run_dir: Path
    config_path: object
    checkpoint: object
    steps: object
    profile: object
    service_policy: object
    controls: dict
    repeat_train: bool = False
    devices: object = None
    nodes: object = None
    node_rank: object = None
    rendezvous: object = None
    strategy: object = None
    sync_every: object = None

    def run(self, *, on_event=None):
        exec_cfg = _execution(self.profile)
        if exec_cfg is not None and exec_cfg.get('node_rank', 0) > 0:
            from .replicated_execution import run_follower
            return run_follower(self.config, self.run_dir, self.steps,
                                profile=self.profile, service_policy=self.service_policy)

        # Revalidate files/options before dispatch: preparation is not a lock and
        # caller code may have changed the config while starting the viewer.
        if self.operation == 'train':
            current = prepare_train(self.config_path, self.run_dir, self.steps,
                profile=self.profile, service_policy=self.service_policy, devices=self.devices,
                nodes=self.nodes, node_rank=self.node_rank, rendezvous=self.rendezvous,
                strategy=self.strategy, sync_every=self.sync_every, **self.controls)
        else:
            current = prepare_resume(self.run_dir, self.checkpoint, self.config_path,
                steps=self.steps, _repeat_train=self.repeat_train,
                profile=self.profile, service_policy=self.service_policy, devices=self.devices,
                nodes=self.nodes, node_rank=self.node_rank, rendezvous=self.rendezvous,
                strategy=self.strategy, sync_every=self.sync_every, **self.controls)
        if current.operation != self.operation:
            raise FileExistsError(f'Run directory appeared after training was prepared: {self.run_dir}')
        if fingerprint(current.config) != fingerprint(self.config) or not _same(_execution(current.profile), _execution(self.profile)):
            raise ValueError('Prepared numerical configuration or execution identity changed')
        if self.repeat_train and not _same(config_values(current.config), config_values(self.config)):
            raise ValueError('Prepared training configuration changed')
        if self.controls.get('preview_every') or enabled_custom(current.config):
            # Fail before step 0 rather than at the first preview or metric job.
            require_main_guard('Previews and custom metrics')
        if _execution(self.profile) is None:
            from .training import train as run_train, resume as run_resume
            options = {}
        else:
            from .replicated_execution import run_train, run_resume
            options = {'profile': self.profile, 'service_policy': self.service_policy}
        if self.devices is None:
            # The controller reloads config_path under the run lock and rejects late edits.
            config, steps = self.config_path, self.steps
        else:
            # A device override exists only in the prepared configuration, so train
            # exactly that (steps included); reloading config_path would drop it.
            config, steps = _launch_config(current.config, self.config_path), None
        if self.operation == 'train':
            return run_train(config, self.run_dir, steps,
                             on_event=on_event, **options, **self.controls)
        return run_resume(self.run_dir, self.checkpoint, config,
                          steps=steps, require_same_config=self.repeat_train,
                          on_event=on_event, **options, **self.controls)


def _launch_config(config, config_path):
    """The resolved configuration plus the source path its provenance snapshot reads."""
    config = dict(config)
    source = config_path.get('_source_path') if isinstance(config_path, dict) else config_path
    if '_source_path' not in config and isinstance(source, (str, Path)):
        config['_source_path'] = str(source)
    return config


# Written into a new run directory before its first manifest: the CLI's
# re-exec launch record and recipe, and the run lock.
_PRE_MANIFEST_ENTRIES = {'.run.lock', 'launch', 'config-source', 'recipe.toml'}


def _is_run(run_dir):
    """Whether an existing run_dir is a run to continue (True) or a new run's unstarted launch (False).

    Anything else is refused: training never writes into a directory that
    HyperGAN did not create.
    """
    if (run_dir / 'manifest.json').is_file():
        return True
    if not run_dir.is_dir():
        raise ValueError(f'Run directory path exists and is not a directory: {run_dir}')
    entries = sorted(p.name for p in run_dir.iterdir())
    if entries and set(entries) <= _PRE_MANIFEST_ENTRIES:
        return False  # A launch that stopped before its first manifest; start it again.
    found = ('it is empty' if not entries else 'it contains '
             + ', '.join(entries[:5]) + (', ...' if len(entries) > 5 else ''))
    raise ValueError(f'{run_dir} exists but is not a HyperGAN run (no manifest.json; {found}); '
                     'choose a new run directory or remove this one')


def prepare_train(config_path, run_dir, steps=None, *, profile=None, service_policy=None,
                  devices=None, nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
                  checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                  preview_every=None, preview_keep=None, preview_keep_source=None,
                  preview_name=None):
    """Create a run or resume its latest full checkpoint with the same configuration."""
    from .run_controller import _controls, resolve_preview_keep
    run_dir = Path(run_dir).resolve() if run_dir is not None else None
    config = _training_config(config_path, steps)
    profile, devices = _apply_execution_options(
        devices=devices, nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
        strategy=strategy, sync_every=sync_every, config=config, profile=profile,
        context_path=run_dir if run_dir else config_path
    )
    profile = _profile(profile, config)
    if run_dir is not None and not _is_follower(profile) and run_dir.exists() and os.environ.get("_HYPERGAN_REEXEC") != "1" and _is_run(run_dir):
        return prepare_resume(run_dir, config_path=config_path, steps=steps, _repeat_train=True,
            profile=profile, service_policy=service_policy, devices=devices,
            nodes=nodes, node_rank=node_rank, rendezvous=rendezvous, strategy=strategy, sync_every=sync_every,
            checkpoint_every=checkpoint_every, max_seconds=max_seconds,
            stop_after_steps=stop_after_steps, preview_every=preview_every,
            preview_keep=preview_keep, preview_keep_source=preview_keep_source,
            preview_name=preview_name)
    checkpoint_every = 1000 if checkpoint_every is None else checkpoint_every
    preview_every = 0 if preview_every is None else preview_every
    preview_keep, preview_keep_source = resolve_preview_keep({}, preview_keep, preview_keep_source)
    preview_name = sample_name(preview_name)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    policy = _policy(profile, service_policy)
    if _execution(profile) is not None:
        from .replicated_execution import MAX_SAMPLE_COUNT
        if config['sampling']['count'] > MAX_SAMPLE_COUNT:
            raise ValueError(f'Replicated final inference sample count exceeds {MAX_SAMPLE_COUNT}')
    controls = dict(checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps, preview_every=preview_every,
                    preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                    preview_name=preview_name)
    return PreparedExecution('train', config, run_dir, config_path, None, steps, profile, policy, controls, False,
                             devices, nodes, node_rank, rendezvous, strategy, sync_every)


def prepare_resume(run_dir, checkpoint=None, config_path=None, *, steps=None, _repeat_train=False,
                   profile=None, service_policy=None, devices=None,
                   nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
                   checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                   preview_every=None, preview_keep=None, preview_keep_source=None,
                   preview_name=None):
    """Infer persisted numerical routing and pin an earlier/latest complete snapshot."""
    from .run_controller import _controls, resolve_preview_keep
    run_dir = Path(run_dir).resolve() if run_dir is not None else None
    is_follower = _is_follower(profile) or (node_rank is not None and node_rank > 0)
    if not is_follower and config_path is not None:
        try:
            cand_config = load_config(config_path)
            if isinstance(cand_config, dict):
                cand_rank = cand_config.get('run', {}).get('node_rank')
                if cand_rank is not None and int(cand_rank) > 0:
                    is_follower = True
        except Exception:
            pass
    manifest = {}
    saved = None
    if not is_follower:
        if run_dir is None or not run_dir.is_dir():
            raise ValueError(f'Run directory does not exist or is not a directory: {run_dir}')
        if not (run_dir / 'manifest.json').is_file():
            raise ValueError(f'{run_dir} has no manifest.json, so it is not a resumable HyperGAN run '
                             '(its first attempt never started); train into it again or choose another run directory')
        manifest = _read_metadata(run_dir / 'manifest.json')
        required = {'schema_version', 'run_id', 'config', 'config_sha256', 'next_sample_sequence'}
        if not required.issubset(manifest) or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
            raise ValueError('Run manifest has no supported full recovery contract')
        config = _training_config(config_path, steps) if config_path is not None else resolve_config(manifest['config'])
        if config_path is None and steps is not None:
            raise ValueError('A training step override requires an explicit configuration')
        saved = manifest.get('execution')
        if 'execution' in manifest and (not isinstance(saved, dict) or saved.get('name') not in PROFILE_NAMES[1:]):
            raise ValueError('Run has no supported numerical execution identity')
        if profile is None and saved is not None:
            profile = profile_from_execution(saved)
    else:
        if config_path is not None:
            config = _training_config(config_path, steps)
        elif run_dir is not None and (run_dir / 'manifest.json').is_file():
            manifest = _read_metadata(run_dir / 'manifest.json')
            config = resolve_config(manifest['config'])
            saved = manifest.get('execution')
        elif run_dir is not None and (run_dir / 'recipe.toml').is_file():
            config = _training_config(run_dir / 'recipe.toml', steps)
        elif run_dir is not None and (run_dir / 'model.toml').is_file():
            config = _training_config(run_dir / 'model.toml', steps)
        else:
            raise ValueError('Follower node resume without shared filesystem requires an explicit configuration (pass --config or recipe.toml)')

    profile, devices = _apply_execution_options(
        devices=devices, nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
        strategy=strategy, sync_every=sync_every, config=config, profile=profile,
        context_path=run_dir
    )
    if not is_follower and isinstance(profile, dict) and _execution(profile) and _execution(profile).get('node_rank', 0) > 0:
        is_follower = True
    if not is_follower:
        if _repeat_train:
            original = config_values(resolve_config(manifest['config']))
            changed = _differences(config_values(config), original)
            if changed and not resume_compatible(config, original, include_observation=True):
                # Resolved defaults are part of the recorded configuration, so a default
                # that changed between releases surfaces here instead of silently applying.
                remedy = ('`hypergan resume RUN --config CONFIG` continues this run with changed '
                          'observation settings' if set(changed) <= {'metrics', 'samplers'} else
                          'use a new run directory for a different numerical configuration')
                raise ConfigError('Training configuration differs from the original run in '
                                  + ', '.join(changed) + '; ' + remedy)
        if fingerprint(config) != manifest['config_sha256']:
            compat = resume_compatible(config, manifest['config'])
            if (not compat
                    or fingerprint(manifest['config']) != manifest['config_sha256']):
                diffs = getattr(compat, 'differences', ())
                diff_str = f" in {', '.join(diffs)}" if diffs else ""
                raise ConfigError(f'Resume configuration differs from the original run{diff_str}; only an increased '
                                  'training.steps with unchanged constant learning rate (lr_floor=1), or enabling '
                                  'pinned training-image skipping, or changing its pixel cache, is allowed')
    profile = _profile(profile, config)
    execution = _execution(profile)
    is_follower = is_follower or (execution is not None and execution.get('node_rank', 0) > 0)
    if (not is_follower or saved is not None) and not _same_topology(execution, saved):
        raise ValueError('Resume numerical execution identity differs from the run manifest')
    policy = _policy(profile, service_policy)
    checkpoint_every = manifest.get('checkpoint_every', 1000) if checkpoint_every is None else checkpoint_every
    preview_every = manifest.get('preview_every', 0) if preview_every is None else preview_every
    preview_keep, preview_keep_source = resolve_preview_keep(manifest, preview_keep, preview_keep_source)
    preview_name = sample_name(manifest.get('preview_name') if preview_name is None else preview_name)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    checkpoint = _checkpoint(run_dir, checkpoint, manifest, execution, config['training']['steps']) if not is_follower else None
    controls = dict(checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps, preview_every=preview_every,
                    preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                    preview_name=preview_name)
    return PreparedExecution('resume', config, run_dir, config_path, checkpoint, steps, profile, policy, controls, _repeat_train,
                             devices, nodes, node_rank, rendezvous, strategy, sync_every)


def train(config_path, run_dir, steps=None, *, on_event=None, **options):
    """Create or resume a run, permitting constant-rate target extensions."""
    return prepare_train(config_path, run_dir, steps, **options).run(on_event=on_event)


def resume(run_dir, checkpoint=None, config_path=None, *, on_event=None, **options):
    """Resume using the saved profile; explicit profile changes must match exactly."""
    return prepare_resume(run_dir, checkpoint, config_path, **options).run(on_event=on_event)
