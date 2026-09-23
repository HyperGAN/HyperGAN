"""Compiled native objective program.

Legacy recipes compile into one critic step and one generator step. The
executor reads sample bindings, detach policies, terms, and parameter
sequences from that program.
"""
from dataclasses import dataclass

import torch
from particlegan import GANLoss, GradientPenalty, learning_rate_scale

from .recipes import detach


@dataclass(frozen=True)
class SampleBinding:
    """Where one scored sample is read, and the two different cuts around it.

    ``detach_sample`` drops the graph before the critic forward.
    ``detach_score`` drops it after the forward. A policy that only detaches
    inputs cannot express both of today's cuts.
    """
    path: str
    detach_sample: bool
    detach_score: bool


@dataclass(frozen=True)
class PhaseSamples:
    real: SampleBinding
    fake: SampleBinding


@dataclass(frozen=True)
class InputRoute:
    """One critic input. Each phase records whether the resolved value is detached."""
    argument: str
    path: str
    detach_critic: bool
    detach_generator: bool


@dataclass(frozen=True)
class AdversarialTerm:
    module: object
    routes: tuple
    weight: float
    penalty: bool
    penalty_fn: object
    gan: object
    critic_phase: PhaseSamples
    generator_phase: PhaseSamples
    id: str


@dataclass(frozen=True)
class ObjectiveTerm:
    function: object
    bindings: tuple
    detach: tuple
    weight: float


@dataclass(frozen=True)
class ObjectiveProgram:
    schedule: str
    critic_parameters: tuple
    generator_parameters: tuple
    prior_parameters: tuple
    prior_checked: tuple
    adversarial_terms: tuple
    objectives: tuple
    prior_rows: str
    gan: object
    penalty: object
    spread: object


def _ordered_unique(parameters):
    """First-seen order. Ownership checks use the set of identities."""
    seen = set()
    ordered = []
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(parameter)
    return tuple(ordered)


def _phases(real, fake):
    # Critic step detaches the fake sample before the forward. Generator step
    # detaches the real score after the forward. Those cuts are not the same.
    critic = PhaseSamples(
        SampleBinding(real, detach_sample=False, detach_score=False),
        SampleBinding(fake, detach_sample=True, detach_score=False))
    generator = PhaseSamples(
        SampleBinding(real, detach_sample=False, detach_score=True),
        SampleBinding(fake, detach_sample=False, detach_score=False))
    return critic, generator


def _legacy_phases():
    return _phases("batch.real", "generated")


def _routes(inputs):
    routes = tuple(
        InputRoute(argument, path, path != "candidate", path != "candidate")
        for argument, path in inputs.items())
    if sum(route.path == "candidate" for route in routes) != 1:
        raise ValueError("The adversarial term requires exactly one candidate input")
    return routes


def _penalty_for_term(config, spec):
    """A new penalty object. The legacy penalty passed into the compiler is not mutated."""
    if not spec["penalty"]:
        return None
    options = dict(config["gradient_penalty"])
    options["coeff"] = spec["penalty_coeff"]
    return GradientPenalty(**options)


def compile_legacy_program(graph, prior, config, objectives, gan, penalty, spread):
    """Compile today's recipe into ``d-then-g-v1``.

    The implicit first term reads the discriminator and its inputs. Optional
    ``adversarial_terms`` append further terms with that same detach policy.
    These records are not another training method.
    """
    discriminator = graph.models["discriminator"]
    critic_phase, generator_phase = _legacy_phases()
    terms = [AdversarialTerm(
        discriminator, _routes(graph.specs["discriminator"]["inputs"]), config["adversarial"]["weight"],
        True, penalty, gan, critic_phase, generator_phase, "adversarial")]
    for spec in config.get("adversarial_terms") or ():
        module = graph.models[spec["component"]]
        extra_critic, extra_generator = _phases(spec["real"], spec["fake"])
        terms.append(AdversarialTerm(
            module, _routes(spec["inputs"]), spec["weight"], bool(spec["penalty"]),
            _penalty_for_term(config, spec), GANLoss(spec["loss_type"], spec["mode"]),
            extra_critic, extra_generator, spec["id"]))
    terms = tuple(terms)
    if len(terms) == 1:
        # Same objects and order as the single-term compiler. Do not filter here:
        # a discriminator parameter that appears in the generator sequence is an overlap.
        critic_parameters = _ordered_unique(parameter for parameter in discriminator.parameters() if parameter.requires_grad)
        generator_parameters = _ordered_unique(graph.generator_parameters())
    else:
        critic_parameters = _ordered_unique(
            parameter
            for term in terms
            for parameter in term.module.parameters()
            if parameter.requires_grad)
        # generator_parameters() already excludes the discriminator by name.
        # Extra critic modules still appear there; drop them so they belong only to the critic group.
        owned = set()
        for term in terms[1:]:
            if term.module is discriminator:
                continue
            for parameter in term.module.parameters():
                owned.add(id(parameter))
        generator_parameters = _ordered_unique(
            parameter for parameter in graph.generator_parameters() if id(parameter) not in owned)
    prior_checked = tuple(prior.parameters())
    prior_parameters = _ordered_unique(parameter for parameter in prior_checked if parameter.requires_grad)
    seen = {}
    for name, parameters in (("critic", critic_parameters), ("generator", generator_parameters), ("prior", prior_parameters)):
        for parameter in parameters:
            previous = seen.get(id(parameter))
            if previous is not None:
                raise ValueError(f"Parameter belongs to both the {previous} and {name} update groups")
            seen[id(parameter)] = name
    if not generator_parameters or not critic_parameters:
        raise ValueError("The reference adversarial loop requires trainable generator and discriminator parameters")
    compiled = tuple(
        ObjectiveTerm(function, tuple(term["inputs"].items()), tuple(term["detach"]), term["weight"])
        for term, function in zip(config["objectives"], objectives))
    return ObjectiveProgram(
        "d-then-g-v1", critic_parameters, generator_parameters, prior_parameters, prior_checked,
        terms, compiled, config["prior_regularizer"]["rows"], gan, penalty, spread)


def _phase_detaches(route, phase):
    try:
        return {"critic": route.detach_critic, "generator": route.detach_generator}[phase]
    except KeyError:
        raise ValueError(f"Unsupported adversarial phase {phase}") from None


def score_candidate(term, candidate, context, graph, phase):
    """Score one sample using that phase's route detach records.

    Detached routes resolve against a detached context, so a component-produced
    condition sees detached inputs, and the resolved value is detached again.
    Attached routes resolve against a live per-score view so gradients can flow
    through a fresh component forward. Detached routes keep the generation cache.
    """
    detached = detach(context)
    detached["candidate"] = candidate
    live = None
    kwargs = {}
    for route in term.routes:
        if _phase_detaches(route, phase):
            kwargs[route.argument] = detach(graph.resolve(route.path, detached))
            continue
        if live is None:
            # Recompute attached components. Reusing the generation cache would let
            # the critic backward free tensors the generator step still needs.
            live = dict(context)
            live["components"] = {}
            live["prior"] = dict(context["prior"])
            live["candidate"] = candidate
        kwargs[route.argument] = graph.resolve(route.path, live)
    return term.module(**kwargs)


def _sample_tensor(graph, context, binding):
    value = graph.resolve(binding.path, context)
    return detach(value) if binding.detach_sample else value


def _sample_context(context, phase_name):
    """Critic-phase component samples must not occupy the generator-step cache.

    An attached real sample can be a component output. Backward through that
    score frees the forward. The generator step resolves the same component
    again, so the critic read uses a private component and prior cache.
    """
    if phase_name != "critic":
        return context
    scratch = dict(context)
    scratch["components"] = {}
    scratch["prior"] = {}
    return scratch


def _bound_scores(term, context, graph, phase_name, phase, *, first):
    """Resolve both samples, then score them in the phase's historical order."""
    sample_context = _sample_context(context, phase_name)
    real = _sample_tensor(graph, sample_context, phase.real)
    fake = _sample_tensor(graph, sample_context, phase.fake)

    def score(sample, binding):
        value = score_candidate(term, sample, context, graph, phase_name)
        return detach(value) if binding.detach_score else value

    if first == "real":
        real_score = score(real, phase.real)
        fake_score = score(fake, phase.fake)
    elif first == "fake":
        fake_score = score(fake, phase.fake)
        real_score = score(real, phase.real)
    else:
        raise ValueError(f"Unsupported score order {first}")
    return real, fake, real_score, fake_score


def _sum_tensors(values):
    """Add tensors without starting from integer 0. A single value is returned as-is."""
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _generator_tail(trainer, program, context, ids, fake):
    if ids is None:
        prior_loss = fake.new_zeros(())
    else:
        rows = trainer.prior.z if program.prior_rows == "full" else trainer.prior.z[ids.unique()]
        prior_loss = program.spread(rows)
    objective_losses = []
    for objective in program.objectives:
        inputs = {arg: trainer.graph.resolve(path, context) for arg, path in objective.bindings}
        for arg in objective.detach:
            inputs[arg] = detach(inputs[arg])
        value = objective.function(**inputs)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("Each objective must return one scalar tensor")
        objective_losses.append(objective.weight * value)
    return prior_loss, objective_losses


def run_native_program(trainer, batch, latent_draw, generator_batch, generator_latent_draw):
    """Execute ``d-then-g-v1`` from the compiled program."""
    from .training import update_ema
    from .tuning_overrides import scheduled_generator_lr
    program = trainer.program
    if program.schedule != "d-then-g-v1":
        raise ValueError(f"Unsupported native schedule {program.schedule}")
    settings = trainer.config["training"]
    independent = settings["phase_draws"] == "independent"
    if not independent and (generator_batch is not None or generator_latent_draw is not None):
        raise ValueError("Explicit generator phase draws require training.phase_draws=independent")
    step = trainer.step + 1
    scale = learning_rate_scale(step - 1, settings["steps"], start=settings["lr_anneal_start"], floor=settings["lr_floor"])
    for optimizer_index, (optimizer, rates) in enumerate(zip((trainer.opt_g, trainer.opt_d), trainer.base_lrs)):
        for group_index, (group, rate) in enumerate(zip(optimizer.param_groups, rates)):
            if optimizer_index == group_index == 0:
                rate = scheduled_generator_lr(rate, getattr(trainer, 'g_lr_warmup', None), step)
            group["lr"] = rate * scale
    if independent:
        with torch.no_grad():
            batch, ids, context = trainer._draw(batch, latent_draw)
    else:
        batch, ids, context = trainer._draw(batch, latent_draw)
    terms = program.adversarial_terms
    observer = getattr(trainer, '_update_response_observer', None)
    trainer.opt_d.zero_grad(set_to_none=True)
    if observer is not None:
        observer('before_d', step=step, batch=batch, ids=ids, context=context)
    if len(terms) == 1:
        term = terms[0]
        real, fake, real_score, fake_score = _bound_scores(
            term, context, trainer.graph, "critic", term.critic_phase, first="real")
        d_adversarial = term.gan.d_loss(real_score, fake_score)
        if term.penalty:
            d_penalty = term.penalty_fn(
                lambda value: score_candidate(term, value, context, trainer.graph, "critic"),
                real, fake, step=step, generator=trainer.streams["penalty"])
        else:
            d_penalty = fake.new_zeros(())
        d_adversarial_weighted = term.weight * d_adversarial
        d_loss = d_adversarial_weighted + d_penalty
    else:
        unweighted = []
        weighted = []
        penalties = []
        fake = None
        for term in terms:
            real, fake, real_score, fake_score = _bound_scores(
                term, context, trainer.graph, "critic", term.critic_phase, first="real")
            loss = term.gan.d_loss(real_score, fake_score)
            unweighted.append(loss)
            weighted.append(term.weight * loss)
            if term.penalty_fn is not None:
                penalties.append(term.penalty_fn(
                    lambda value, term=term: score_candidate(term, value, context, trainer.graph, "critic"),
                    real, fake, step=step, generator=trainer.streams["penalty"]))
        d_adversarial = _sum_tensors(unweighted)
        d_adversarial_weighted = _sum_tensors(weighted)
        d_penalty = _sum_tensors(penalties) if penalties else fake.new_zeros(())
        d_loss = d_adversarial_weighted + d_penalty
    d_loss.backward()
    if len(terms) == 1:
        critic_checked = terms[0].module.parameters()
    else:
        critic_checked = []
        seen_parameters = set()
        for term in terms:
            for parameter in term.module.parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                critic_checked.append(parameter)
    trainer._refuse_nonfinite(d_loss, "Nonfinite discriminator loss; run stopped",
                              [("Nonfinite discriminator gradient; run stopped", critic_checked)])
    trainer.opt_d.step()
    if observer is not None:
        observer('after_d', step=step, batch=batch, ids=ids, context=context)
    saved_flags = []
    seen_modules = set()
    for term in terms:
        if id(term.module) in seen_modules:
            continue
        seen_modules.add(id(term.module))
        flags = [parameter.requires_grad for parameter in term.module.parameters()]
        term.module.requires_grad_(False)
        saved_flags.append((term.module, flags))
    try:
        if independent:
            batch, ids, context = trainer._draw(generator_batch, generator_latent_draw)
        trainer.opt_g.zero_grad(set_to_none=True)
        if observer is not None:
            observer('before_g', step=step, batch=batch, ids=ids, context=context)
        if len(terms) == 1:
            term = terms[0]
            real, fake, real_score, fake_score = _bound_scores(
                term, context, trainer.graph, "generator", term.generator_phase, first="fake")
            g_adversarial = term.gan.g_loss(fake_score, real_score)
            prior_loss, objective_losses = _generator_tail(trainer, program, context, ids, fake)
            g_adversarial_weighted = term.weight * g_adversarial
        else:
            unweighted = []
            weighted = []
            fake = None
            for term in terms:
                real, fake, real_score, fake_score = _bound_scores(
                    term, context, trainer.graph, "generator", term.generator_phase, first="fake")
                loss = term.gan.g_loss(fake_score, real_score)
                unweighted.append(loss)
                weighted.append(term.weight * loss)
            g_adversarial = _sum_tensors(unweighted)
            prior_loss, objective_losses = _generator_tail(trainer, program, context, ids, fake)
            g_adversarial_weighted = _sum_tensors(weighted)
        g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
        g_loss.backward()
        trainer._refuse_nonfinite(g_loss, "Nonfinite generator loss; run stopped",
                                  [("Nonfinite generator/auxiliary gradient; run stopped", program.generator_parameters),
                                   ("Nonfinite prior gradient; run stopped", program.prior_checked)])
        trainer.opt_g.step()
        if observer is not None:
            observer('after_g', step=step, batch=batch, ids=ids, context=context)
    finally:
        for module, flags in saved_flags:
            for parameter, flag in zip(module.parameters(), flags):
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
