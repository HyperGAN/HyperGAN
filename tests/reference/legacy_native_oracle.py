"""Frozen copy of the native D-then-G step, for equivalence tests only.

This is the update that lived on ``ReferenceTrainer`` before the compiled
program. It deliberately calls ``ComponentGraph.critic`` and
``generator_parameters`` so a bug in the executor cannot satisfy both sides.
Do not use it from the training runtime.
"""
import torch

from hypergan.recipes import detach
from hypergan.training import ScoredCritic, noise_levels, schedule_learning_rates, update_ema


def legacy_native_update(trainer, batch=None, latent_draw=None, *, generator_batch=None, generator_latent_draw=None):
    """One critic step, then one generator step, then EMA. Matches the historical native update."""
    cfg = trainer.config
    settings = cfg["training"]
    independent = settings["phase_draws"] == "independent"
    if not independent and (generator_batch is not None or generator_latent_draw is not None):
        raise ValueError("Explicit generator phase draws require training.phase_draws=independent")
    step = trainer.step + 1
    if noise_levels(trainer, step - 1) != (0, 0):
        raise ValueError("The oracle covers noise-free recipes only")
    scale = schedule_learning_rates(trainer, step - 1)
    if independent:
        with torch.no_grad():
            batch, ids, context = trainer._draw(batch, latent_draw)
    else:
        batch, ids, context = trainer._draw(batch, latent_draw)
    fake, real = context["generated"], batch["real"]
    critic = lambda x: trainer.graph.critic(x, context)
    trainer.opt_d.zero_grad(set_to_none=True)
    d_adversarial = trainer.gan.d_loss(critic(real), critic(fake.detach()))
    scored = ScoredCritic(trainer.graph.models["discriminator"],
                          lambda module, x: trainer.graph.critic(x, context, module=module))
    d_penalty = trainer.penalty(scored, real, fake.detach())
    d_adversarial_weighted = cfg["adversarial"]["weight"] * d_adversarial
    d_loss = d_adversarial_weighted + d_penalty
    d_loss.backward()
    trainer._refuse_nonfinite(d_loss, "Nonfinite discriminator loss; run stopped",
                              [("Nonfinite discriminator gradient; run stopped",
                                trainer.graph.models["discriminator"].parameters())])
    trainer.opt_d.step()
    discriminator = trainer.graph.models["discriminator"]
    flags = [p.requires_grad for p in discriminator.parameters()]
    discriminator.requires_grad_(False)
    try:
        if independent:
            batch, ids, context = trainer._draw(generator_batch, generator_latent_draw)
            fake, real = context["generated"], batch["real"]
        trainer.opt_g.zero_grad(set_to_none=True)
        g_adversarial = trainer.gan.g_loss(critic(fake), critic(real).detach())
        if ids is None:
            prior_loss = fake.new_zeros(())
        else:
            rows = trainer.prior.z if cfg["prior_regularizer"]["rows"] == "full" else trainer.prior.z[ids.unique()]
            prior_loss = trainer.spread(rows)
        objective_losses = []
        for term, objective in zip(cfg["objectives"], trainer.objectives):
            inputs = {arg: trainer.graph.resolve(path, context) for arg, path in term["inputs"].items()}
            for arg in term["detach"]:
                inputs[arg] = detach(inputs[arg])
            value = objective(**inputs)
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError("Each objective must return one scalar tensor")
            objective_losses.append(term["weight"] * value)
        g_adversarial_weighted = cfg["adversarial"]["weight"] * g_adversarial
        g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
        g_loss.backward()
        trainer._refuse_nonfinite(g_loss, "Nonfinite generator loss; run stopped",
                                  [("Nonfinite generator/auxiliary gradient; run stopped",
                                    trainer.graph.generator_parameters()),
                                   ("Nonfinite prior gradient; run stopped", trainer.prior.parameters())])
        trainer.opt_g.step()
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
    update_ema(trainer.ema_graph, trainer.graph, settings["ema"])
    update_ema(trainer.ema_prior, trainer.prior, settings["ema"])
    trainer.step = step
    values = trainer._metric_transfer([d_loss, d_adversarial, d_adversarial_weighted,
        g_adversarial_weighted, g_loss, g_adversarial, prior_loss, d_penalty,
        *objective_losses])
    row = dict(zip(("d_loss", "d_adversarial", "d_adversarial_weighted",
                    "g_adversarial_weighted", "g_loss", "g_adversarial",
                    "prior_loss", "gradient_penalty"), values[:8]))
    row.update(event="train", step=step, objectives=values[8:], lr_scale=scale)
    return row, detach(batch)
