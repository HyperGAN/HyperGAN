"""HyperGAN-owned, bounded single-process CPU reference loop.

CPU recovery is supported; image quality and DDP require separate qualification.
"""
import copy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import random
import uuid
import warnings
import inspect

import numpy as np

import torch
from particlegan import GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale

from .config import config_values, fingerprint, load_config, resolve_config
from .recipes import ComponentGraph, construct, detach, make_prior
from .checkpoints import capture_rng, restore_rng, data_contract, read_checkpoint, restore_trainer, write_checkpoint
from .run_state import atomic_json, repair_event_tail, run_lock, sync_directory


@torch.no_grad()
def update_ema(average, current, decay):
    for target, source in zip(average.parameters(), current.parameters()):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(average.buffers(), current.buffers()):
        target.copy_(source)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown (source import without distribution metadata)"


def runtime_info():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.system(), "machine": platform.machine(), "threads": 1, "particlegan": _version("particlegan"), "hypergan": _version("hypergan"), "device": "cpu", "dtype": "float32", "world_size": 1, "default_dtype": str(torch.get_default_dtype()), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def source_info():
    # The integration reference SHA is not a claim about an installed wheel's source.
    root = Path(__file__).resolve().parents[2]
    result = {"integration_reference": {"repository": "https://github.com/255BITS/ParticleGAN", "commit": "f946b4ed468ff3b3eae5a3bca11411d5725f1181"}, "particlegan_distribution_version": _version("particlegan"), "particlegan_distribution_commit": None, "hypergan_commit": None}
    result["distribution_records"] = {}
    for name in ("hypergan", "particlegan"):
        try:
            distribution = importlib.metadata.distribution(name)
            record = distribution.read_text("RECORD")
            result["distribution_records"][name] = {"record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None}
        except importlib.metadata.PackageNotFoundError:
            result["distribution_records"][name] = None
    if (root / ".git").exists():
        try:
            result["hypergan_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            result["hypergan_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


class ReferenceTrainer:
    """Small inspectable state machine; future distributed strategies wrap this contract."""
    def __init__(self, config):
        self.config = config
        settings = config["training"]
        torch.manual_seed(settings["seed"])
        random.seed(settings["seed"])
        np.random.seed(settings["seed"] % (2 ** 32))
        self.graph = ComponentGraph(config["components"]).float()
        self.prior = make_prior(config["prior"]).float()
        self.data = construct(config["data"])
        self.gan = GANLoss(**{k: v for k, v in config["adversarial"].items() if k != "weight"})
        self.penalty = GradientPenalty(**config["gradient_penalty"])
        self.spread = ParticleRegularizer(**{k: v for k, v in config["prior_regularizer"].items() if k != "rows"})
        self.objectives = [construct(term) for term in config["objectives"]]
        if any(isinstance(term, torch.nn.Module) and any(p.requires_grad for p in term.parameters()) for term in self.objectives):
            raise ValueError("Objective constructors must not own trainable parameters; declare trainable transforms as components and bind their outputs into an objective")
        if any(isinstance(term, torch.nn.Module) and list(term.buffers()) for term in self.objectives):
            raise ValueError("Stateful objective buffers are not supported by this reference loop; declare stateful transforms as components")
        opt = config["optimizer"]
        groups = [{"params": self.graph.generator_parameters(), "lr": opt["lr"]}]
        prior_parameters = [p for p in self.prior.parameters() if p.requires_grad]
        if prior_parameters:
            groups.append({"params": prior_parameters, "lr": opt["lr"] * opt["prior_lr_mult"], "betas": tuple(opt["prior_betas"])})
        d_parameters = [p for p in self.graph.models["discriminator"].parameters() if p.requires_grad]
        if not groups[0]["params"] or not d_parameters:
            raise ValueError("The reference adversarial loop requires trainable generator and discriminator parameters")
        self.opt_g = torch.optim.Adam(groups, betas=tuple(opt["betas"]))
        self.opt_d = torch.optim.Adam(d_parameters, lr=opt["lr"] * opt["d_lr_mult"], betas=tuple(opt["betas"]))
        self.base_lrs = [[g["lr"] for g in optimizer.param_groups] for optimizer in (self.opt_g, self.opt_d)]
        self.ema_graph = copy.deepcopy(self.graph).eval().requires_grad_(False)
        self.ema_prior = copy.deepcopy(self.prior).eval().requires_grad_(False)
        self.streams = {"data": torch.Generator().manual_seed(settings["seed"] + 1), "prior": torch.Generator().manual_seed(settings["seed"] + 2), "penalty": torch.Generator().manual_seed(settings["seed"] + 3)}
        self.step = 0

    def batch(self):
        batch = self.data(self.config["training"]["batch_size"], generator=self.streams["data"])
        if not isinstance(batch, dict) or not isinstance(batch.get("real"), torch.Tensor):
            raise ValueError("Data constructor must return a callable producing a dict containing tensor 'real'")
        if batch["real"].device.type != "cpu" or not batch["real"].is_floating_point():
            raise ValueError("Reference real data must be floating-point CPU tensors")
        return batch

    def update(self, batch=None, latent_draw=None):
        """Execute one D update followed by G/prior/aux update and matched EMA.

        Explicit batch and (latent, indices) enable controlled numerical comparisons.
        """
        cfg = self.config
        settings = cfg["training"]
        step = self.step + 1
        scale = learning_rate_scale(step - 1, settings["steps"], start=settings["lr_anneal_start"], floor=settings["lr_floor"])
        for optimizer, rates in zip((self.opt_g, self.opt_d), self.base_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * scale
        batch = self.batch() if batch is None else batch
        if len(batch["real"]) != settings["batch_size"]:
            raise ValueError("Data batch length must match training.batch_size")
        z, ids = self.prior.sample(len(batch["real"]), generator=self.streams["prior"]) if latent_draw is None else latent_draw
        context = self.graph.generate(z, batch)
        fake, real = context["generated"], batch["real"]
        if not isinstance(fake, torch.Tensor) or fake.shape != real.shape:
            raise ValueError(f"Generator output must match real data shape {tuple(real.shape)}")
        critic = lambda x: self.graph.critic(x, context)
        self.opt_d.zero_grad(set_to_none=True)
        d_adversarial = self.gan.d_loss(critic(real), critic(fake.detach()))
        d_penalty = self.penalty(critic, real, fake.detach(), step=step, generator=self.streams["penalty"])
        d_loss = cfg["adversarial"]["weight"] * d_adversarial + d_penalty
        if not torch.isfinite(d_loss).all():
            raise ValueError("Nonfinite discriminator loss; run stopped")
        d_loss.backward()
        self._check_gradients(self.graph.models["discriminator"].parameters(), "discriminator")
        self.opt_d.step()
        discriminator = self.graph.models["discriminator"]
        flags = [p.requires_grad for p in discriminator.parameters()]
        discriminator.requires_grad_(False)
        try:
            self.opt_g.zero_grad(set_to_none=True)
            g_adversarial = self.gan.g_loss(critic(fake), critic(real).detach())
            if ids is None:
                prior_loss = fake.new_zeros(())
            else:
                rows = self.prior.z if cfg["prior_regularizer"]["rows"] == "full" else self.prior.z[ids.unique()]
                prior_loss = self.spread(rows)
            objective_losses = []
            for term, objective in zip(cfg["objectives"], self.objectives):
                inputs = {arg: self.graph.resolve(path, context) for arg, path in term["inputs"].items()}
                for arg in term["detach"]:
                    inputs[arg] = detach(inputs[arg])
                value = objective(**inputs)
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError("Each objective must return one scalar tensor")
                objective_losses.append(term["weight"] * value)
            g_loss = cfg["adversarial"]["weight"] * g_adversarial + prior_loss + sum(objective_losses)
            if not torch.isfinite(g_loss).all():
                raise ValueError("Nonfinite generator loss; run stopped")
            g_loss.backward()
            self._check_gradients(self.graph.generator_parameters(), "generator/auxiliary")
            self._check_gradients(self.prior.parameters(), "prior")
            self.opt_g.step()
        finally:
            for parameter, flag in zip(discriminator.parameters(), flags):
                parameter.requires_grad_(flag)
        update_ema(self.ema_graph, self.graph, settings["ema"])
        update_ema(self.ema_prior, self.prior, settings["ema"])
        self.step = step
        return {"event": "train", "step": step, "d_loss": float(d_loss.detach()), "g_loss": float(g_loss.detach()), "g_adversarial": float(g_adversarial.detach()), "prior_loss": float(prior_loss.detach()), "gradient_penalty": float(d_penalty.detach()), "objectives": [float(x.detach()) for x in objective_losses], "lr_scale": scale}, detach(batch)

    @staticmethod
    def _check_gradients(parameters, name):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f"Nonfinite {name} gradient; run stopped")


def _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every=0, preview_keep=3):
    if type(preview_every) is not int or preview_every < 0:
        raise ValueError("preview_every must be a nonnegative integer; zero disables previews")
    if type(preview_keep) is not int or not 1 <= preview_keep <= 100:
        raise ValueError("preview_keep must be between 1 and 100")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    if max_seconds is not None and (type(max_seconds) not in (int, float) or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be a finite positive number')
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError('stop_after_steps must be a positive integer')


def _implementation(trainer):
    """Hash imported implementation bytes without depending on checkout location."""
    import hypergan.checkpoints
    import hypergan.config
    import hypergan.recipes
    objects = [hypergan.checkpoints, hypergan.config, hypergan.recipes, ReferenceTrainer,
               GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale,
               type(trainer.data), type(trainer.prior), *[type(x) for x in trainer.graph.modules()],
               *[x if inspect.isfunction(x) else type(x) for x in trainer.objectives]]
    specifications = [trainer.config['data'], *trainer.config['components'].values(), *trainer.config['objectives']]
    for specification in specifications:
        factory = specification['factory']
        if ':' in factory:
            objects.append(importlib.import_module(factory.split(':', 1)[0]))
    result = {}
    for obj in objects:
        module = inspect.getmodule(obj)
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            result[module.__name__] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def _recovery_contract(trainer):
    contract = data_contract(trainer.data, trainer.config['data'])
    reasons = []
    if not contract['supported']:
        reasons.append('Custom data must declare resume_stateless=True or paired state_dict/load_state_dict methods for recovery')
    for spec, objective in zip(trainer.config['objectives'], trainer.objectives):
        if spec['factory'] not in ('mse', 'l1') and getattr(objective, 'resume_stateless', False) is not True:
            reasons.append('Custom objectives must declare resume_stateless=True for recovery; move mutable state into registered component buffers')
    return contract, reasons


def _new_attempt(run_dir):
    root = run_dir / 'attempts'
    root.mkdir(exist_ok=True)
    indexes = [int(p.name.split('-')[0]) for p in root.iterdir() if p.is_dir() and p.name.split('-')[0].isdigit()]
    index = max(indexes, default=0) + 1
    identity = f'{index:04d}-{uuid.uuid4().hex}'
    path = root / identity
    path.mkdir()
    sync_directory(root)
    return index, identity, path


def train(config_path, run_dir, steps=None, *, checkpoint_every=100, max_seconds=None,
          stop_after_steps=None, on_event=None, preview_every=0, preview_keep=3):
    """Create a run; budgets stop only at complete D/G/EMA update boundaries."""
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep)
    config = load_config(config_path)
    if steps is not None:
        raw = config_values(config)
        raw['training']['steps'] = steps
        config = resolve_config(raw)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    qualification = dict(config['qualification'])
    qualification['status'] = 'reference-only' if qualification['recipe_match'] else 'unqualified'
    qualification['runtime_qualification'] = 'not-certified; run numerical parity CI for this exact runtime'
    manifest = {'schema_version': 1, 'status': 'initializing', 'run_id': uuid.uuid4().hex,
                'run_dir': str(run_dir), 'config': config_values(config), 'config_sha256': fingerprint(config),
                'runtime': runtime_info(), 'source': source_info(), 'qualification': qualification,
                'warnings': list(config['warnings']), 'steps': 0, 'total_steps': config['training']['steps'],
                'global_batch_size': config['training']['batch_size'], 'resume_supported': False,
                'last_durable_step': None, 'checkpoint_path': None, 'next_sample_sequence': 1,
                'preview_every': preview_every, 'preview_keep': preview_keep, 'previews': [], 'observation_errors': [],
                'rng_streams': {name: config['training']['seed'] + offset for name, offset in [('data', 1), ('prior', 2), ('penalty', 3)]}}
    manifest['rng_streams']['sampling'] = config['sampling']['seed']
    with run_lock(run_dir):
        return _execute(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps, on_event)


def resume(run_dir, checkpoint=None, config_path=None, *, checkpoint_every=None,
           max_seconds=None, stop_after_steps=None, on_event=None, preview_every=None, preview_keep=None):
    """Resume a full checkpoint from this run, retaining the original schedule."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f'Run directory does not exist: {run_dir}')
    with run_lock(run_dir):
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        required = {'schema_version', 'run_id', 'config', 'config_sha256', 'next_sample_sequence'}
        if not isinstance(manifest, dict) or not required.issubset(manifest) or manifest['schema_version'] != 1:
            raise ValueError('Run manifest has no supported full recovery contract')
        checkpoint_every = manifest.get('checkpoint_every', 100) if checkpoint_every is None else checkpoint_every
        preview_every = manifest.get('preview_every', 0) if preview_every is None else preview_every
        preview_keep = manifest.get('preview_keep', 3) if preview_keep is None else preview_keep
        _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep)
        config = load_config(config_path) if config_path is not None else resolve_config(manifest['config'])
        target, info, state = read_checkpoint(run_dir, checkpoint)
        if info['run_id'] != manifest['run_id']:
            raise ValueError('Checkpoint belongs to a different run')
        if fingerprint(config) != info['config_sha256'] or fingerprint(config) != manifest['config_sha256']:
            raise ValueError('Resume configuration differs from checkpoint; total training schedule cannot change')
        if runtime_info() != info['runtime']:
            raise ValueError('Resume runtime/topology differs from checkpoint')
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            trainer = ReferenceTrainer(config)
            contract, reasons = _recovery_contract(trainer)
            if reasons:
                raise ValueError('Recovery unsupported: ' + '; '.join(reasons))
            if contract != info['data_contract']:
                raise ValueError('Resume data identity or state protocol differs from checkpoint')
            if _implementation(trainer) != info['implementation']:
                raise ValueError('Resume implementation differs from checkpoint')
            batch = restore_trainer(trainer, state)
            manifest.update(preview_every=preview_every, preview_keep=preview_keep,
                            checkpoint_path=str(target), last_durable_step=trainer.step,
                            resumed_from=str(target), steps=trainer.step)
            return _execute(config, run_dir, manifest, checkpoint_every, max_seconds,
                            stop_after_steps, on_event, trainer=trainer, last_batch=batch)
        finally:
            torch.set_num_threads(previous_threads)


def _execute(config, run_dir, manifest, checkpoint_every, max_seconds, stop_after_steps,
             on_event, trainer=None, last_batch=None):
    previous_threads = torch.get_num_threads()
    started = time.monotonic()
    index, attempt_id, attempt_dir = _new_attempt(run_dir)
    manifest.update(attempt_id=attempt_id, attempt_index=index, attempt_dir=str(attempt_dir),
                    status='initializing', checkpoint_every=checkpoint_every, stop_reason=None,
                    possible_lost_steps=0)
    for key in ('error', 'sample_path', 'bundle_path'):
        manifest.pop(key, None)
    atomic_json(run_dir / 'manifest.json', manifest)
    repair_event_tail(run_dir / 'events.jsonl')
    sequence = 0

    def emit(event, **values):
        nonlocal sequence
        sequence += 1
        row = dict(values, schema_version=1, event=event, run_id=manifest['run_id'],
                   attempt_id=attempt_id, sequence=sequence,
                   step=manifest['steps'], seconds=time.monotonic() - started)
        with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
        if on_event is not None:
            rng = capture_rng()
            try:
                on_event(dict(row))
            except Exception as exc:
                try:
                    warnings.warn(f'Run event observer failed: {exc}', RuntimeWarning)
                except Warning:
                    pass
            finally:
                restore_rng(rng)
        return row

    def publish():
        manifest['seconds'] = time.monotonic() - started
        atomic_json(run_dir / 'manifest.json', manifest)
        atomic_json(attempt_dir / 'manifest.json', manifest)

    try:
        torch.set_num_threads(1)
        trainer = trainer if trainer is not None else ReferenceTrainer(config)
        contract, reasons = _recovery_contract(trainer)
        manifest.update(resume_supported=not reasons, resume_unsupported_reasons=reasons,
                        data_identity=contract['identity'], status='running')
        manifest['qualification']['resume'] = False
        manifest['qualification']['recovery_scope'] = 'CPU full-state protocol; custom hidden state is author responsibility'
        for reason in reasons:
            if reason not in manifest['warnings']:
                manifest['warnings'].append(reason)
            warnings.warn(reason, RuntimeWarning)
        metadata = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                    'config': config_values(config), 'config_sha256': fingerprint(config),
                    'runtime': runtime_info(), 'data_contract': contract,
                    'implementation': _implementation(trainer),
                    'next_sample_sequence': manifest['next_sample_sequence']}

        def checkpoint_now(request_ids=None, observer=False):
            if not manifest['resume_supported']:
                return
            metadata['next_sample_sequence'] = manifest['next_sample_sequence']
            metadata['request_ids'] = list(request_ids or [])
            try:
                path = write_checkpoint(run_dir, trainer, last_batch, metadata)
            except Exception as exc:
                if observer:
                    return None, exc
                raise
            manifest.update(checkpoint_path=str(path), last_durable_step=trainer.step,
                            possible_lost_steps=0)
            publish()
            emit('checkpoint', checkpoint_path=str(path), request_ids=list(request_ids or []))
            return path, None

        def observer_error(source, error):
            record = {'source': source, 'step': trainer.step, 'attempt_id': attempt_id,
                      'error': f'{type(error).__name__}: {error}'[:1000]}
            manifest['observation_errors'] = [*manifest.get('observation_errors', []), record][-16:]
            publish()
            emit('observer_error', **{key: value for key, value in record.items() if key not in ('step', 'attempt_id')})

        def preview_now():
            from .previews import publish_preview
            identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                        'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence']}
            manifest['next_sample_sequence'] += 1
            publish()  # Reserve before rendering: failed or killed attempts never reuse a sequence.
            try:
                record, preview_index, errors = publish_preview(run_dir, trainer, last_batch, identity,
                                                                keep=manifest['preview_keep'])
            except Exception as exc:
                observer_error('preview', exc)
                return
            manifest['previews'] = preview_index['previews']
            manifest['preview_path'] = record['path']
            publish()
            emit('preview', preview=record)
            for error in errors:
                observer_error('preview_retention', RuntimeError(error))

        def poll_requests():
            from .run_requests import pending_requests, acknowledge_request
            try:
                pending = pending_requests(run_dir)
            except RuntimeError:
                return  # A producer holds the short queue lock; retry next boundary.
            except Exception as exc:
                observer_error('checkpoint_requests', exc)
                return
            matching = []
            def acknowledge(request, status, path=None, error=None, saved_step=None):
                try:
                    receipt = acknowledge_request(run_dir, request['request_id'], status=status,
                                                  attempt_id=attempt_id, checkpoint_path=str(path) if path else None,
                                                  step=(trainer.step if saved_step is None else saved_step) if path else None, error=error)
                except Exception as exc:
                    observer_error('checkpoint_acknowledgement', exc)
                    return
                emit('checkpoint_request', receipt=receipt)
            for request in pending:
                if request['run_id'] != manifest['run_id'] or request['attempt_id'] != attempt_id:
                    acknowledge(request, 'rejected', error='Request targets a different run or attempt; it cannot be applied after resume')
                elif not manifest['resume_supported']:
                    acknowledge(request, 'rejected', error='Full training checkpoints are unsupported for this configuration')
                else:
                    matching.append(request)
            if matching and manifest.get('checkpoint_path'):
                # A lost acknowledgement must not repeat a still-identifiable save.
                try:
                    saved_path = Path(manifest['checkpoint_path'])
                    saved = json.loads((saved_path / 'manifest.json').read_text(encoding='utf-8'))
                    completed_ids = set(saved.get('request_ids', [])) if saved.get('attempt_id') == attempt_id else set()
                except Exception as exc:
                    observer_error('checkpoint_request_reconciliation', exc)
                    return
                remaining = []
                for request in matching:
                    if request['request_id'] in completed_ids:
                        acknowledge(request, 'succeeded', path=saved_path, saved_step=saved['step'])
                    else:
                        remaining.append(request)
                matching = remaining
            if matching:
                path, error = checkpoint_now([request['request_id'] for request in matching], observer=True)
                for request in matching:
                    acknowledge(request, 'succeeded' if path else 'rejected', path=path,
                                error=f'{type(error).__name__}: {error}'[:1000] if error else None)
                if error:
                    observer_error('manual_checkpoint', error)

        publish()  # A start/resume observer can immediately submit an attempt-bound request.
        emit('resume' if manifest.get('resumed_from') else 'start', config_sha256=fingerprint(config))
        if manifest['last_durable_step'] is None or manifest.get('resumed_from'):
            # Accepting an older recovery point must also move the default pointer,
            # even if this attempt stops before another update.
            checkpoint_now()
        publish()
        poll_requests()
        attempt_steps = 0
        while trainer.step < config['training']['steps']:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                manifest['stop_reason'] = 'max_seconds'
                break
            if stop_after_steps is not None and attempt_steps >= stop_after_steps:
                manifest['stop_reason'] = 'stop_after_steps'
                break
            row, last_batch = trainer.update()
            attempt_steps += 1
            manifest['steps'] = trainer.step
            durable = manifest['last_durable_step']
            manifest['possible_lost_steps'] = trainer.step - durable if durable is not None else trainer.step
            emit('train', **{key: value for key, value in row.items() if key not in ('event', 'step')})
            if trainer.step % checkpoint_every == 0:
                checkpoint_now()
            poll_requests()
            if manifest['preview_every'] and trainer.step % manifest['preview_every'] == 0:
                preview_now()
            publish()
        if manifest['last_durable_step'] != trainer.step:
            checkpoint_now()
        if last_batch is not None:
            from .artifacts import save_bundle, sample
            bundle_dir = attempt_dir / 'inference'
            bundle_dir.mkdir()
            sync_directory(attempt_dir)
            trainer.artifact_identity = {'run_id': manifest['run_id'], 'attempt_id': attempt_id,
                                         'attempt_index': index, 'sample_sequence': manifest['next_sample_sequence']}
            manifest['next_sample_sequence'] += 1
            publish()
            save_bundle(bundle_dir, trainer, last_batch)
            sample_path = sample(bundle_dir, count=config['sampling']['count'], seed=config['sampling']['seed'])
            manifest.update(bundle_path=str(bundle_dir / 'model.pt'), sample_path=str(sample_path))
        manifest['status'] = 'complete' if trainer.step == config['training']['steps'] else 'stopped'
        publish()
        emit(manifest['status'], stop_reason=manifest['stop_reason'], checkpoint_path=manifest['checkpoint_path'])
        return manifest
    except BaseException as exc:
        # Live trainer may contain a half update: NEVER checkpoint in this handler.
        manifest.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        publish()
        emit(manifest['status'], error=manifest['error'], checkpoint_path=manifest['checkpoint_path'])
        raise
    finally:
        torch.set_num_threads(previous_threads)
