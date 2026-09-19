"""HyperGAN-owned, bounded single-process CPU reference loop.

This establishes numerical integration, not image quality, resume, or DDP support.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from particlegan import GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale

from .config import config_values, fingerprint, load_config, resolve_config
from .recipes import ComponentGraph, construct, detach, make_prior


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
    return {"python": platform.python_version(), "torch": torch.__version__, "particlegan": _version("particlegan"), "hypergan": _version("hypergan"), "device": "cpu", "dtype": "float32", "world_size": 1}


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


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(config_path, run_dir, steps=None):
    config = load_config(config_path)
    if steps is not None:
        raw = config_values(config)
        raw["training"]["steps"] = steps
        config = resolve_config(raw)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime = runtime_info()
    qualification = dict(config["qualification"])
    # This revision introduces the fixture. A manifest records observed checks rather
    # than promoting an arbitrary installed wheel/runtime to a certified profile.
    qualification["status"] = "reference-only" if qualification["recipe_match"] else "unqualified"
    qualification["runtime_qualification"] = "not-certified; run numerical parity CI for this exact runtime"
    manifest = {"schema_version": 1, "status": "running", "run_dir": str(run_dir), "config": config_values(config), "config_sha256": fingerprint(config), "runtime": runtime, "source": source_info(), "qualification": qualification, "warnings": config["warnings"], "steps": 0, "global_batch_size": config["training"]["batch_size"], "rng_streams": {"data": config["training"]["seed"] + 1, "prior": config["training"]["seed"] + 2, "penalty": config["training"]["seed"] + 3, "sampling": config["sampling"]["seed"]}, "resume_supported": False}
    _json(run_dir / "manifest.json", manifest)
    previous_threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        trainer = ReferenceTrainer(config)
        with (run_dir / "events.jsonl").open("x") as events:
            events.write(json.dumps({"event": "start", "config_sha256": manifest["config_sha256"], "qualification": qualification}) + "\n")
            events.flush()
            for _ in range(config["training"]["steps"]):
                row, batch = trainer.update()
                row["seconds"] = time.monotonic() - started
                events.write(json.dumps(row, allow_nan=False) + "\n")
                events.flush()
                manifest["steps"] = trainer.step
            from .artifacts import save_bundle, sample
            save_bundle(run_dir, trainer, batch)
            sample_path = sample(run_dir, count=config["sampling"]["count"], seed=config["sampling"]["seed"])
            manifest.update(status="complete", sample_path=str(sample_path), bundle_path=str(run_dir / "model.pt"), seconds=time.monotonic() - started)
            events.write(json.dumps({"event": "complete", "step": trainer.step, "sample_path": str(sample_path)}) + "\n")
        _json(run_dir / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", error=f"{type(exc).__name__}: {exc}")
        _json(run_dir / "manifest.json", manifest)
        with (run_dir / "events.jsonl").open("a") as events:
            events.write(json.dumps({"event": manifest["status"], "step": manifest["steps"], "error": manifest["error"]}) + "\n")
        raise
    finally:
        torch.set_num_threads(previous_threads)
