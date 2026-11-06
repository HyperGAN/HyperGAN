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


def _validate_saved_checkpoint_topology(topology):
    if not isinstance(topology, dict) or topology.get('name') not in PROFILE_NAMES[1:]:
        raise ValueError('Resume checkpoint numerical execution identity differs from valid topology schema')
    for key in ('world_size', 'accumulation_steps', 'nodes', 'ranks_per_node',
                'global_batch_size', 'local_batch_size', 'microbatch_size'):
        if key in topology:
            val = topology[key]
            if type(val) is not int or isinstance(val, bool) or val < 1:
                raise ValueError('Resume checkpoint numerical execution identity differs from valid topology schema')


def _checkpoint(run_dir, checkpoint, manifest, execution, total_steps, *, current_config=None):
    """Pin a complete generation across checkpoints/ and distributed-checkpoints/ without torch.

    The adapter validates all payload bytes, runtime, data and implementation
    during restore, under the run lock and before publishing a new attempt.
    """
    from .checkpoints import resolve_resume_checkpoint_dir
    target, pointer = resolve_resume_checkpoint_dir(run_dir, checkpoint, run_manifest=manifest)
    info = _read_metadata(target / 'manifest.json')
    if type(info.get('schema_version')) is not int or info['schema_version'] != 1:
        raise ValueError('Unsupported training checkpoint schema')
    if type(info.get('step')) is not int or not 0 <= info['step'] <= total_steps:
        raise ValueError('Checkpoint step is outside the original schedule')
    if pointer is not None and (type(pointer.get('step')) is not int or pointer['step'] != info['step']):
        raise ValueError('Checkpoint pointer step differs from generation')
    expected_root_kind = (
        'hypergan-distributed-training-checkpoint'
        if target.parent.name == 'distributed-checkpoints'
        else 'hypergan-training-checkpoint'
    )
    if info.get('kind') != expected_root_kind:
        raise ValueError(f'Unsupported or mismatched training checkpoint kind {info.get("kind")!r}')
    if info.get('run_id') != manifest['run_id']:
        raise ValueError('Checkpoint belongs to a different run')
    is_dist_ckpt = info['kind'] == 'hypergan-distributed-training-checkpoint'
    identity = info.get('identity', {}) if is_dist_ckpt else info
    if not isinstance(identity, dict):
        raise ValueError('Resume checkpoint configuration differs from the run manifest')
    if identity.get('config_sha256') != manifest['config_sha256']:
        saved_config = identity.get('config')
        ref_config = (
            manifest['config']
            if isinstance(manifest.get('config'), dict) and manifest['config']
            else current_config
        )
        if (
            not isinstance(saved_config, dict)
            or not isinstance(ref_config, dict)
            or not resume_compatible(
                ref_config,
                saved_config,
                allow_device_change=True,
                allow_batch_size_change=True,
            )
            or fingerprint(saved_config) != identity.get('config_sha256')
        ):
            raise ValueError('Resume checkpoint configuration differs from the run manifest')
    from .checkpoint_compatibility import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(identity)
    if is_dist_ckpt:
        _validate_saved_checkpoint_topology(identity.get('topology'))
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


def _sync_prior_device(config):
    if not isinstance(config, dict):
        return
    dev = config.get("training", {}).get("device")
    if isinstance(dev, str) and dev:
        prior_args = config.get("prior", {}).get("args")
        if isinstance(prior_args, dict) and "device" in prior_args:
            prior_args["device"] = dev


def _requests_single_device(*, devices=None, device=None, nodes=None, strategy=None, sync_every=None, context_path=None):
    """True when explicit CLI/API arguments request single-device execution rather than inheriting saved replication."""
    cluster_env = detect_cluster_environment()
    eff_nodes = int(nodes) if nodes is not None else int(cluster_env.get("nodes") or 1)
    if eff_nodes > 1 or strategy is not None or (sync_every is not None and int(sync_every) > 1):
        return False
    if devices is not None:
        from .workspace import find_workspace, load_workspace, resolve_device_aliases
        ws_path = None
        for cand in (os.environ.get("HYPERGAN_WORKSPACE"), context_path, Path.cwd()):
            if cand is not None:
                try:
                    p = Path(cand)
                    ws_path = p if (p.is_file() and p.name == "hypergan.toml") else find_workspace(p)
                    if ws_path:
                        break
                except Exception:
                    pass
        ws_data = load_workspace(ws_path) if ws_path else {}
        resolved = resolve_device_aliases(devices, ws_data)
        dev_list = [d.strip() for d in (resolved if isinstance(resolved, list) else str(resolved).split(",")) if d.strip()]
        return len(dev_list) == 1
    if device is not None:
        return True
    return False


def _apply_execution_options(
    devices=None,
    device=None,
    nodes=None,
    node_rank=None,
    rendezvous=None,
    strategy=None,
    sync_every=None,
    config=None,
    profile=None,
    context_path=None,
):
    if device is not None:
        if isinstance(config, dict) and "training" in config:
            config["training"]["device"] = device
        elif isinstance(config, dict):
            config.setdefault("training", {})["device"] = device
        _sync_prior_device(config)
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
    if device is not None and devices is None:
        if isinstance(config, dict) and "training" in config:
            config["training"]["device"] = device

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
            _sync_prior_device(config)
    elif len(dev_list) == 1:
        if dev_list[0].lower() == "cpu":
            _set_visible_devices("", order=False)
            if isinstance(config, dict) and "training" in config:
                config["training"]["device"] = "cpu"
                _sync_prior_device(config)
        else:
            _set_visible_devices(dev_list[0])
            if isinstance(config, dict) and "training" in config:
                config["training"]["device"] = "cuda"
                _sync_prior_device(config)

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
            _sync_prior_device(config)
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
                old_exec = profile["execution"]
                old_nodes = old_exec.get("nodes", 1)
                old_world = old_exec.get("world_size", 1)
                old_rpn = old_exec.get("ranks_per_node")
                if type(old_rpn) is not int or old_rpn < 1:
                    old_rpn = (
                        max(1, old_world // max(1, old_nodes))
                        if type(old_world) is int and type(old_nodes) is int
                        else 1
                    )
                if cli_nodes is not None or cluster_env.get("nodes") is not None or len(dev_list) > 1:
                    new_nodes = nodes if (cli_nodes is not None or cluster_env.get("nodes") is not None) else old_nodes
                    new_rpn = len(dev_list) if len(dev_list) > 1 else old_rpn
                    new_world = new_rpn * new_nodes
                    profile["execution"]["nodes"] = new_nodes
                    profile["execution"]["ranks_per_node"] = new_rpn
                    profile["execution"]["world_size"] = new_world
                    if len(dev_list) > 1:
                        profile["execution"]["name"] = name
                    if (new_nodes, new_rpn, new_world) != (old_nodes, old_rpn, old_world):
                        for k in ("global_batch_size", "local_batch_size", "microbatch_size", "accumulation_algorithm"):
                            profile["execution"].pop(k, None)
                        profile.pop("preflight", None)
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

    prof_name = (
        profile
        if isinstance(profile, str)
        else (profile.get("execution", {}).get("name") if isinstance(profile, dict) and isinstance(profile.get("execution"), dict) else None)
    )
    if prof_name == "cuda-replicated-nccl" and isinstance(config, dict):
        cur_dev = str(config.get("training", {}).get("device") or "")
        if cur_dev.startswith("cuda:"):
            config["training"]["device"] = "cuda"
            _sync_prior_device(config)

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
    device: object = None
    allow_device_change: bool = False
    allow_batch_size_change: bool = False

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
                device=self.device,
                nodes=self.nodes, node_rank=self.node_rank, rendezvous=self.rendezvous,
                strategy=self.strategy, sync_every=self.sync_every, **self.controls)
        else:
            current = prepare_resume(self.run_dir, self.checkpoint, self.config_path,
                steps=self.steps, _repeat_train=self.repeat_train,
                _allow_device_change=self.allow_device_change,
                _allow_batch_size_change=self.allow_batch_size_change,
                profile=self.profile, service_policy=self.service_policy, devices=self.devices,
                device=self.device,
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
        if (
            self.devices is None
            and self.device is None
            and not self.allow_device_change
            and not self.allow_batch_size_change
        ):
            # The controller reloads config_path under the run lock and rejects late edits.
            config, steps = self.config_path, self.steps
        else:
            # A device or batch-size override exists in the prepared configuration, so run
            # exactly that (steps included); reloading config_path or manifest would drop it.
            config, steps = _launch_config(current.config, self.config_path), None
        if self.operation == 'train':
            return run_train(config, self.run_dir, steps,
                             on_event=on_event, **options, **self.controls)
        return run_resume(self.run_dir, self.checkpoint, config,
                          steps=steps, require_same_config=self.repeat_train,
                          allow_device_change=self.allow_device_change,
                          allow_batch_size_change=self.allow_batch_size_change,
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
                  devices=None, device=None, nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
                  checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                  preview_every=None, preview_keep=None, preview_keep_source=None,
                  preview_name=None):
    """Create a run or resume its latest full checkpoint with the same configuration."""
    from .run_controller import _controls, resolve_preview_keep
    raw_profile = profile
    raw_devices = devices
    run_dir = Path(run_dir).resolve() if run_dir is not None else None
    config = _training_config(config_path, steps)
    profile, devices = _apply_execution_options(
        devices=devices, device=device, nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
        strategy=strategy, sync_every=sync_every, config=config, profile=profile,
        context_path=run_dir if run_dir else config_path
    )
    profile = _profile(profile, config)
    if run_dir is not None and not _is_follower(profile) and run_dir.exists() and os.environ.get("_HYPERGAN_REEXEC") != "1" and _is_run(run_dir):
        return prepare_resume(run_dir, config_path=config_path, steps=steps, _repeat_train=True,
            profile=raw_profile, service_policy=service_policy, devices=raw_devices, device=device,
            nodes=nodes, node_rank=node_rank, rendezvous=rendezvous, strategy=strategy, sync_every=sync_every,
            checkpoint_every=checkpoint_every, max_seconds=max_seconds,
            stop_after_steps=stop_after_steps, preview_every=preview_every,
            preview_keep=preview_keep, preview_keep_source=preview_keep_source,
            preview_name=preview_name)
    run_cfg = config.get('run', {}) if isinstance(config, dict) else {}
    if checkpoint_every is None:
        checkpoint_every = run_cfg.get('checkpoint_every', 1000)
    if preview_every is None:
        preview_every = run_cfg.get('preview_every', 0)
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
                             devices, nodes, node_rank, rendezvous, strategy, sync_every, device)


def prepare_resume(run_dir, checkpoint=None, config_path=None, *, steps=None, _repeat_train=False,
                   _allow_device_change=False, _allow_batch_size_change=False,
                   profile=None, service_policy=None, devices=None, device=None,
                   nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
                   checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                   preview_every=None, preview_keep=None, preview_keep_source=None,
                   preview_name=None):
    """Infer persisted numerical routing or validate an elastic cross-topology migration."""
    from .run_controller import _controls, resolve_preview_keep
    explicit_device_requested = bool(devices is not None or device is not None or _allow_device_change)
    single_device_requested = _requests_single_device(
        devices=devices, device=device, nodes=nodes, strategy=strategy, sync_every=sync_every, context_path=run_dir
    )
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
        if saved is not None:
            saved_profile = resolve_execution_profile(profile_from_execution(saved), config) if False else profile_from_execution(saved)
            # Validate saved execution types strictly even when scaling down to single-device.
            resolve_execution_profile(deepcopy(saved_profile), resolve_config({
                **config_values(config),
                'training': {
                    **config_values(config)['training'],
                    'device': 'cpu' if saved.get('name') == 'cpu-replicated-gloo' else 'cuda',
                    'batch_size': saved.get('global_batch_size', config['training']['batch_size'])
                    if type(saved.get('global_batch_size')) is int and saved.get('global_batch_size') >= 1
                    else config['training']['batch_size'],
                },
            }))
            if profile is None and not single_device_requested:
                profile = saved_profile
    else:
        if config_path is not None:
            config = _training_config(config_path, steps)
        elif run_dir is not None and (run_dir / 'manifest.json').is_file():
            manifest = _read_metadata(run_dir / 'manifest.json')
            config = resolve_config(manifest['config'])
            saved = manifest.get('execution')
            if isinstance(saved, dict) and saved.get('name') in PROFILE_NAMES[1:] and profile is None and not single_device_requested:
                profile = profile_from_execution(saved)
        elif run_dir is not None and (run_dir / 'recipe.toml').is_file():
            config = _training_config(run_dir / 'recipe.toml', steps)
        elif run_dir is not None and (run_dir / 'model.toml').is_file():
            config = _training_config(run_dir / 'model.toml', steps)
        else:
            raise ValueError('Follower node resume without shared filesystem requires an explicit configuration (pass --config or recipe.toml)')

    profile, devices = _apply_execution_options(
        devices=devices, device=device, nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
        strategy=strategy, sync_every=sync_every, config=config, profile=profile,
        context_path=run_dir
    )
    if not is_follower and isinstance(profile, dict) and _execution(profile) and _execution(profile).get('node_rank', 0) > 0:
        is_follower = True
    saved_dev = (
        str(manifest.get('config', {}).get('training', {}).get('device') or '')
        if isinstance(manifest.get('config'), dict) and isinstance(manifest['config'].get('training'), dict)
        else ''
    )
    cur_dev = str(config.get('training', {}).get('device') or '') if isinstance(config, dict) else ''
    same_cuda_family = saved_dev.split(':')[0] == 'cuda' and cur_dev.split(':')[0] == 'cuda'
    allow_device_change = bool(explicit_device_requested or same_cuda_family)
    allow_batch_size_change = bool(_allow_batch_size_change)
    if not is_follower:
        if _repeat_train:
            original = config_values(resolve_config(manifest['config']))
            changed = _differences(config_values(config), original)
            if changed and not resume_compatible(
                config,
                original,
                include_observation=True,
                allow_device_change=allow_device_change,
                allow_batch_size_change=allow_batch_size_change,
            ):
                # Resolved defaults are part of the recorded configuration, so a default
                # that changed between releases surfaces here instead of silently applying.
                remedy = ('`hypergan resume RUN --config CONFIG` continues this run with changed '
                          'observation settings' if set(changed) <= {'metrics', 'samplers'} else
                          'use a new run directory for a different numerical configuration')
                raise ConfigError('Training configuration differs from the original run in '
                                  + ', '.join(changed) + '; ' + remedy)
        if fingerprint(config) != manifest['config_sha256']:
            compat = resume_compatible(
                config,
                manifest['config'],
                allow_device_change=allow_device_change,
                allow_batch_size_change=allow_batch_size_change,
            )
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
    policy = _policy(profile, service_policy)
    run_cfg = (config.get('run', {}) if isinstance(config, dict) else {}) or (manifest.get('config', {}).get('run', {}) if isinstance(manifest.get('config'), dict) else {})
    if checkpoint_every is None:
        checkpoint_every = manifest.get('checkpoint_every', run_cfg.get('checkpoint_every', 1000))
    if preview_every is None:
        preview_every = manifest.get('preview_every', run_cfg.get('preview_every', 0))
    preview_keep, preview_keep_source = resolve_preview_keep(manifest, preview_keep, preview_keep_source)
    preview_name = sample_name(manifest.get('preview_name') if preview_name is None else preview_name)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    checkpoint = (
        _checkpoint(run_dir, checkpoint, manifest, execution, config['training']['steps'], current_config=config)
        if not is_follower
        else None
    )
    controls = dict(checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps, preview_every=preview_every,
                    preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                    preview_name=preview_name)
    return PreparedExecution('resume', config, run_dir, config_path, checkpoint, steps, profile, policy, controls, _repeat_train,
                             devices, nodes, node_rank, rendezvous, strategy, sync_every,
                             device, allow_device_change, allow_batch_size_change)


def train(config_path, run_dir, steps=None, *, on_event=None, **options):
    """Create or resume a run, permitting constant-rate target extensions."""
    return prepare_train(config_path, run_dir, steps, **options).run(on_event=on_event)


def resume(run_dir, checkpoint=None, config_path=None, *, on_event=None, **options):
    """Resume using the saved profile; explicit profile changes must match exactly."""
    return prepare_resume(run_dir, checkpoint, config_path, **options).run(on_event=on_event)
