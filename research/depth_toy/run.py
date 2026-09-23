"""Depth-only adaptation of ParticleGAN's native 100gaussians training loop."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn

SEED = 1234  # Original example seed; deliberately no seed-sweep interface.


def load_source(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import particlegan
    from lib import toy_models
    # Avoid silently using a different installed ParticleGAN.
    if not Path(particlegan.__file__).resolve().is_relative_to(root):
        raise RuntimeError("ParticleGAN was already imported from another checkout")
    return particlegan, toy_models


def digest(tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def initialize(pg, models, depth, steps):
    """Construct native baseline first so depth cannot shift D/prior draws."""
    recipe = pg.get_recipe("100gaussians", total_steps=steps)
    torch.manual_seed(SEED)
    prior = recipe.make_prior()
    source = models.SimpleMLPGenerator(z_dim=recipe.z_dim)
    critic = models.SimpleMLPDiscriminator(in_dim=2, fourier=2)
    for module in list(source.modules()) + list(critic.modules()):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    generator = source
    if depth != 3:
        # One fixed extra-layer initialization stream, shared across depth cases.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 100)
            generator = models.SimpleMLPGenerator(z_dim=recipe.z_dim, n_hidden=depth)
            for index, module in enumerate(generator.modules()):
                if isinstance(module, nn.Linear):
                    torch.manual_seed(SEED + 100 + index)
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for i in range(3):
            generator.net[2 * i].load_state_dict(source.net[2 * i].state_dict())
        generator.net[-1].load_state_dict(source.net[-1].state_dict())
    shared = {f"hidden{i}.{k}": v for i in range(3)
              for k, v in generator.net[2 * i].state_dict().items()}
    shared.update({f"output.{k}": v for k, v in generator.net[-1].state_dict().items()})
    return recipe, generator, critic, prior, {
        "generator": digest(generator.state_dict()), "shared_generator": digest(shared),
        "critic": digest(critic.state_dict()), "prior": digest(prior.state_dict()),
    }


def moments(value):
    value = value.detach().double()
    energy = value.square().mean()
    shared = value.mean(0).square().mean()
    return {"rms": energy.sqrt().item(),
            "between_sample_rms": value.var(0, unbiased=False).mean().sqrt().item(),
            "shared_energy_fraction": (shared / energy).item() if energy > 0 else None}


def movement(request, actual):
    request, actual = request.detach().double(), actual.detach().double()
    denom = request.norm() * actual.norm()
    return {"requested": moments(request), "actual": moments(actual),
            "cosine": ((request * actual).sum() / denom).item() if denom > 0 else None,
            "gain": (actual.norm() / request.norm()).item() if request.norm() > 0 else None}


@torch.no_grad()
def distribution(fake, real):
    coords = torch.arange(10, dtype=fake.dtype) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    distances, nearest = torch.cdist(fake, centers).min(1)
    hq = distances <= .09
    counts = torch.bincount(nearest[hq], minlength=100)
    # Fixed, evenly spaced directions: no evaluation RNG enters training.
    angles = torch.arange(64, dtype=fake.dtype) * torch.pi / 64
    directions = torch.stack((angles.cos(), angles.sin()))
    sw1 = ((fake @ directions).sort(0).values - (real @ directions).sort(0).values).abs().mean()
    return {"modes": int((counts >= 10).sum()), "hq": float(hq.float().mean()),
            "sliced_w1": float(sw1), "nearest_center_rms": float(distances.square().mean().sqrt()),
            "nearest_mode_mass": (torch.bincount(nearest, minlength=100) / len(fake)).tolist(),
            **moments(fake)}


def train_case(pg, models, depth, steps=7000, observe_every=500, output=None,
               diagnostics=True, progress=True):
    recipe, G, D, prior, identity = initialize(pg, models, depth, steps)
    ema_G, ema_prior = copy.deepcopy(G), copy.deepcopy(prior)
    ema_G.requires_grad_(False)
    ema_prior.requires_grad_(False)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    spread = recipe.make_prior_regularizer(weight=1.)
    opt_g, opt_d = recipe.make_optimizers(G, D, fused=False)
    opt_p = torch.optim.Adam(prior.parameters(), lr=recipe.lr * recipe.prior_lr_mult,
                             betas=recipe.betas, fused=False)
    optimizers = (opt_g, opt_d, opt_p)
    base_rates = [[group['lr'] for group in opt.param_groups] for opt in optimizers]
    data_rng = torch.Generator().manual_seed(SEED)
    latent_rng = torch.Generator().manual_seed(SEED + 2)
    penalty_rng = torch.Generator().manual_seed(SEED + 3)
    eval_rng = torch.Generator().manual_seed(SEED + 999)
    ids = torch.randint(recipe.num_particles, (20000,), generator=eval_rng)
    real = models.sample_100gaussians(20000, torch.device('cpu'), generator=eval_rng)
    initial_z = prior(ids).detach().clone()
    identity.update(monitor=digest({'ids': ids, 'real': real, 'initial_z': initial_z}))
    records = []
    started = time.perf_counter()

    def emit(row):
        records.append(row)
        if output:
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        if progress:
            shown = {k: row[k] for k in ('step', 'seconds', 'online', 'ema')}
            for key in ('online', 'ema'):
                shown[key] = {k: shown[key][k] for k in ('modes', 'hq', 'sliced_w1')}
            print(json.dumps({'depth': depth, **shown}), flush=True)

    def observe(step, update=None):
        with torch.no_grad():
            row = {'step': step, 'seconds': time.perf_counter() - started,
                   'online': distribution(G(prior(ids)), real),
                   'ema': distribution(ema_G(ema_prior(ids)), real),
                   'fixed_initial_latents': distribution(G(initial_z), real),
                   'prior_std': float(prior.z.std()), 'update': update}
        emit(row)

    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps({
            'depth': depth, 'seed': SEED, 'recipe': recipe.to_dict(),
            'identity': identity, 'parameters': sum(p.numel() for p in G.parameters()),
        }, indent=2) + '\n')
    if diagnostics:
        observe(0)
    for index in range(steps):
        step = index + 1
        due = diagnostics and (step in (1, 8, 32, 128) or step % observe_every == 0 or step == steps)
        anneal_from = recipe.lr_anneal_start * steps
        scale = pg.learning_rate_scale(index - anneal_from, max(1., steps - anneal_from), 0., recipe.lr_floor)
        for opt, rates in zip(optimizers, base_rates):
            for group, rate in zip(opt.param_groups, rates):
                group['lr'] = rate * scale
        D.train()
        G.eval()
        x_real = models.sample_100gaussians(recipe.batch_size, torch.device('cpu'), generator=data_rng)
        with torch.no_grad():
            z, _ = prior.sample(recipe.batch_size, generator=latent_rng)
            fake = G(z)
        d_loss = gan.d_loss(D(x_real), D(fake))
        penalty, _ = regularizer.penalty(D, x_real, fake, step, generator=penalty_rng, collect_stats=False)
        opt_d.zero_grad()
        (d_loss + penalty).backward()
        opt_d.step()

        D.eval()
        G.train()
        activations, handles = {}, []
        if due:
            def capture(name):
                def hook(module, inputs, out):
                    if torch.is_grad_enabled():
                        out.retain_grad()
                        activations[name] = out
                return hook
            for name, module in G.named_modules():
                if isinstance(module, nn.LeakyReLU):
                    handles.append(module.register_forward_hook(capture(name)))
        z, idx = prior.sample(recipe.batch_size, generator=latent_rng)
        fake = G(z)
        if due:
            fake.retain_grad()
            z.retain_grad()
        real_g = models.sample_100gaussians(recipe.batch_size, torch.device('cpu'), generator=data_rng)
        g_loss = gan.g_loss(D(fake), D(real_g))
        raw = prior.z if recipe.num_particles <= 1024 else prior.z[idx.unique()]
        p_loss = spread(raw)
        opt_g.zero_grad()
        opt_p.zero_grad()
        (g_loss + recipe.prior_reg * p_loss).backward()
        for handle in handles:
            handle.remove()
        if due:
            before = {name: p.detach().clone() for name, p in G.named_parameters()}
            layers = {name: {'activation': moments(a), 'gradient': moments(a.grad)}
                      for name, a in activations.items()}
        opt_g.step()
        update = None
        if due:
            with torch.no_grad():
                after_g = G(z.detach())
                update = {'g_only': movement(-fake.grad, after_g - fake.detach()),
                          'latent_adversarial_gradient': moments(z.grad), 'layers': layers,
                          'parameter_updates': {name: {
                              'gradient_rms': float(p.grad.square().mean().sqrt()),
                              'delta_rms': float((p - before[name]).square().mean().sqrt())}
                              for name, p in G.named_parameters()},
                          'd_loss': float(d_loss), 'g_loss': float(g_loss),
                          'penalty': float(penalty), 'lr_scale': scale}
        opt_p.step()
        with torch.no_grad():
            for target, current in zip(ema_G.parameters(), G.parameters()):
                target.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
            for target, current in zip(ema_prior.parameters(), prior.parameters()):
                target.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
            if due:
                update['g_and_prior'] = movement(-fake.grad, G(prior(idx)) - fake.detach())
        if due:
            observe(step, update)
    result = {'depth': depth, 'identity': identity, 'records': records,
              'seconds': time.perf_counter() - started}
    if output:
        (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result, {'G': G, 'D': D, 'prior': prior, 'ema_G': ema_G, 'ema_prior': ema_prior}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particlegan-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 8, 16])
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--observe-every', type=int, default=500)
    args = parser.parse_args()
    if args.steps < 1 or args.observe_every < 1 or any(d < 3 for d in args.depths) or len(set(args.depths)) != len(args.depths):
        parser.error('positive steps/interval and distinct depths >=3 required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pg, models = load_source(args.particlegan_root)
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = args.particlegan_root.resolve()
    paths = [source_root / 'examples/100gaussians.py', source_root / 'lib/toy_models.py',
             *sorted((source_root / 'particlegan').glob('*.py')), Path(__file__).resolve()]
    provenance = {'torch': torch.__version__, 'python': sys.version, 'seed': SEED,
                  'argv': sys.argv, 'device': 'cpu', 'threads': 1,
                  'source_commit': subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip(),
                  'runner_commit': subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip(),
                  'file_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    results = []
    for depth in args.depths:
        result, _ = train_case(pg, models, depth, args.steps, args.observe_every, args.output / f'depth-{depth}')
        results.append(result)
        (args.output / 'summary.json').write_text(json.dumps([
            {'depth': r['depth'], 'seconds': r['seconds'], 'final': r['records'][-1]}
            for r in results], indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
