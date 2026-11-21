"""Compiled native objective program.

Every configuration compiles into one critic step and one generator step. The
executor reads sample bindings, detach policies, terms, and parameter
sequences from that program.
"""
from dataclasses import dataclass
import math

import torch

from .recipes import detach


@dataclass(frozen=True)
class SampleBinding:
    """Where one scored sample is read, and the two different cuts around it.

    ``detach_sample`` drops the graph before the critic forward.
    ``detach_score`` drops it after the forward. A policy that only detaches
    inputs cannot express both cuts.
    """
    path: str
    detach_sample: bool
    detach_score: bool
    use_draw_cache: bool = False


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
    critic_weight: float
    generator_weight: float
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


def _phases(real, fake, *, shared_fake=False):
    # Critic step detaches the fake sample before the forward. Generator step
    # detaches the real score after the forward. Those cuts are not the same.
    critic = PhaseSamples(
        SampleBinding(real, detach_sample=False, detach_score=False),
        SampleBinding(fake, detach_sample=True, detach_score=False, use_draw_cache=shared_fake))
    generator = PhaseSamples(
        SampleBinding(real, detach_sample=False, detach_score=True),
        SampleBinding(fake, detach_sample=False, detach_score=False))
    return critic, generator


def _routes(inputs):
    routes = tuple(
        InputRoute(argument, path, path != "candidate", path != "candidate")
        for argument, path in inputs.items())
    if sum(route.path == "candidate" for route in routes) != 1:
        raise ValueError("The adversarial term requires exactly one candidate input")
    return routes


def compile_objective_program(graph, prior, config, objectives, gan, penalty, spread, extra_term=None):
    """Compile the configured objectives into the ``d-then-g-v1`` program the trainer runs.

    The implicit first term reads the discriminator and its inputs. Optional
    ``adversarial_terms`` append further terms with that same detach policy.
    These records are not another training method.
    """
    discriminator = graph.models["discriminator"]
    primary = config["adversarial"]
    extra_specs = config.get("adversarial_terms") or ()
    shared_fake = bool(config.get("recipe")) and any(spec["component"] != "discriminator" for spec in extra_specs)

    def compile_term(spec, module, routes, loss, penalty_fn):
        critic_phase, generator_phase = _phases(
            spec.get("real", "batch.real"), spec.get("fake", "generated"), shared_fake=shared_fake)
        return AdversarialTerm(
            module=module, routes=_routes(routes), weight=spec["weight"],
            critic_weight=spec.get("critic_weight", spec["weight"]),
            generator_weight=spec.get("generator_weight", spec["weight"]),
            penalty=penalty_fn is not None, penalty_fn=penalty_fn, gan=loss,
            critic_phase=critic_phase, generator_phase=generator_phase, id=spec.get("id", "adversarial"))

    terms = [compile_term(primary, discriminator, graph.specs["discriminator"]["inputs"], gan, penalty)]
    for spec in extra_specs:
        loss, term_penalty = extra_term(spec)
        terms.append(compile_term(spec, graph.models[spec["component"]], spec["inputs"], loss, term_penalty))
    terms = tuple(terms)
    if len(terms) == 1:
        # The discriminator's own parameter objects, in module order. Do not filter
        # here: a discriminator parameter that appears in the generator sequence is
        # an overlap, which the ownership check below rejects.
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


def score_candidate(term, candidate, context, graph, phase, *, module=None, noise=(0, None)):
    """Score one sample using that phase's route detach records.

    ``module`` replaces the term's critic (the penalty passes its EMA critic).
    ``noise`` is the critic input noise ``(std, generator)`` added to the
    candidate inside the critic, so penalties differentiate the clean input.

    Detached routes resolve against a detached context, so a component-produced
    condition sees detached inputs, and the resolved value is detached again.
    Attached routes resolve against a live per-score view so gradients can flow
    through a fresh component forward. Detached routes keep the generation cache.
    """
    from .training import with_noise
    candidate = with_noise(candidate, *noise)
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
    from .recipes import _call_component
    return _call_component(term.module if module is None else module, kwargs)


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


def _bound_scores(term, context, graph, phase_name, phase, *, first, noise):
    """Resolve both samples, then score them in the phase's historical order.

    ``noise`` holds the ``(std, generator)`` pairs of the generator output
    noise, added to the fake sample, and the critic input noise.
    """
    from .training import with_noise
    output_noise, input_noise = noise
    sample_context = _sample_context(context, phase_name)
    real = _sample_tensor(graph, sample_context, phase.real)
    # Independent critics reading a generated mask must see the same mask,
    # including stochastic component outputs. The critic detaches this sample.
    fake_context = context if phase.fake.use_draw_cache else sample_context
    fake = with_noise(_sample_tensor(graph, fake_context, phase.fake), *output_noise)

    def score(sample, binding):
        value = score_candidate(term, sample, context, graph, phase_name, noise=input_noise)
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


def _graph_generator_uses_batch(graph):
    """Return True if resolving ``generator`` transitively reads ``batch.*``."""
    visited = set()

    def walk(name):
        if name in visited or name not in graph.specs:
            return False
        visited.add(name)
        for path in graph.specs[name].get("inputs", {}).values():
            if path == "batch" or path.startswith("batch."):
                return True
            if path.startswith("components."):
                sub = path.split(".", 2)[1]
                if walk(sub):
                    return True
        return False

    return walk("generator")


def _rhbi_generator_context(trainer, terms):
    """Return ``(rhbi_module, is_discrete, use_simplex)`` for the current generator step."""
    from particlegan import ParticlePrior

    if len(terms) != 1:
        return None, False, False
    rhbi = getattr(terms[0].gan, "t1c", None) or getattr(terms[0].gan, "rhbi", None)
    if rhbi is None or not hasattr(rhbi, "bind_discrete_prior_step"):
        return None, False, False
    is_discrete = type(trainer.prior) is ParticlePrior
    use_simplex = bool(
        is_discrete
        and trainer.prior.z.requires_grad
        and len(trainer.prior.z) >= 2
        and len(terms[0].routes) == 1
        and getattr(trainer.recipe, "rhbi_simplex_weight", 0.0) > 0.0
        and not _graph_generator_uses_batch(trainer.graph)
    )
    return rhbi, is_discrete, use_simplex


def _rhbi_simplex_loss(trainer, rhbi, selected_ids, real_batch):
    """Evaluate the RHBI simplicial manifold regularizer on an unconditional discrete ParticlePrior."""
    raw_base = trainer.prior.z
    base_table = raw_base if len(raw_base) <= 1024 or selected_ids is None else raw_base[selected_ids]
    return trainer.recipe.rhbi_simplex_weight * rhbi.simplicial_manifold_loss(
        base_table,
        lambda z_q: trainer.graph.generate(z_q, {}, prior=trainer.prior)["generated"],
        real_batch,
        record=getattr(trainer.opt_d, "record", None),
        chart_weight=trainer.recipe.rhbi_simplex_chart_weight,
        macro_settle=trainer.recipe.rhbi_simplex_macro_settle,
        curve_settle=trainer.recipe.rhbi_simplex_curve_settle,
    )


def _generator_tail(trainer, program, context, ids, fake):
    if ids is None or not trainer.prior.z.requires_grad:
        prior_loss = fake.new_zeros(())
    else:
        rows = trainer.prior.z if program.prior_rows == "full" else trainer.prior.z[ids.unique()]
        prior_loss = trainer.recipe.prior_regularization(rows, regularizer=program.spread, prior=trainer.prior)
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
    from .training import ScoredCritic, noise_levels, schedule_learning_rates, update_ema
    program = trainer.program
    if program.schedule != "d-then-g-v1":
        raise ValueError(f"Unsupported native schedule {program.schedule}")
    settings = trainer.config["training"]
    independent = settings["phase_draws"] == "independent"
    if not independent and (generator_batch is not None or generator_latent_draw is not None):
        raise ValueError("Explicit generator phase draws require training.phase_draws=independent")
    step = trainer.step + 1
    scale = schedule_learning_rates(trainer, step - 1)
    input_std, output_std = noise_levels(trainer, step - 1)
    stream = trainer.streams["noise"]
    noise = ((output_std, stream), (input_std, stream))

    from .particlegan_runtime import invoke_critic_penalty

    def penalty(term, real, fake):
        critic = ScoredCritic(term.module, lambda module, value: score_candidate(
            term, value, context, trainer.graph, "critic", module=module, noise=(input_std, stream)))
        return invoke_critic_penalty(term.penalty_fn, critic, real, fake)
    thermal_coordinator = getattr(trainer, "thermal_coordinator", None)
    if thermal_coordinator is not None:
        thermal_coordinator.synchronize()
    if independent:
        with torch.no_grad():
            batch, ids, context = trainer._draw(batch, latent_draw)
    else:
        batch, ids, context = trainer._draw(batch, latent_draw)
    terms = program.adversarial_terms
    for optimizer in trainer.critic_optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    per_term_records = []
    if len(terms) == 1:
        term = terms[0]
        real, fake, real_score, fake_score = _bound_scores(
            term, context, trainer.graph, "critic", term.critic_phase, first="real", noise=noise)
        d_adversarial = term.gan.d_loss(real_score, fake_score)
        if term.penalty:
            d_penalty = penalty(term, real, fake)
        else:
            d_penalty = fake.new_zeros(())
        d_adversarial_weighted = term.critic_weight * d_adversarial
        d_loss = d_adversarial_weighted + d_penalty
        per_term_records.append((term.id, d_adversarial_weighted, d_penalty, d_loss, real_score, fake_score))
        d_real_score, d_fake_score, d_real_samples, d_fake_samples = real_score, fake_score, real, fake
    else:
        unweighted = []
        weighted = []
        penalties = []
        d_real_score = d_fake_score = d_real_samples = d_fake_samples = None
        for index, term in enumerate(terms):
            real, fake, real_score, fake_score = _bound_scores(
                term, context, trainer.graph, "critic", term.critic_phase, first="real", noise=noise)
            loss = term.gan.d_loss(real_score, fake_score)
            weighted_t = term.critic_weight * loss
            unweighted.append(loss)
            weighted.append(weighted_t)
            if term.penalty_fn is not None:
                pen_t = penalty(term, real, fake)
                penalties.append(pen_t)
            else:
                pen_t = fake.new_zeros(())
            tot_t = weighted_t + pen_t
            per_term_records.append((term.id, weighted_t, pen_t, tot_t, real_score, fake_score))
            if index == 0:
                d_real_score, d_fake_score, d_real_samples, d_fake_samples = real_score, fake_score, real, fake
        d_adversarial = _sum_tensors(unweighted)
        d_adversarial_weighted = _sum_tensors(weighted)
        d_penalty = _sum_tensors(penalties) if penalties else d_fake_samples.new_zeros(())
        d_loss = d_adversarial_weighted + d_penalty
    d_loss.backward()
    if len(terms) == 1:
        critic_checked = list(terms[0].module.parameters())
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
    with torch.no_grad():
        zero_scalar = torch.zeros((), dtype=torch.float32, device=trainer.device)
        one_scalar = torch.ones((), dtype=torch.float32, device=trainer.device)
        half_scalar = torch.full((), 0.5, dtype=torch.float32, device=trainer.device)
        try:
            c_grads = [p.grad.detach() for p in critic_checked if p.grad is not None]
            critic_grad_norm_t = torch.sqrt(sum(g.norm(2) ** 2 for g in c_grads)).reshape(()) if c_grads else zero_scalar
        except Exception:
            critic_grad_norm_t = zero_scalar
        try:
            rs = d_real_score.detach()
            fs = d_fake_score.detach()
            critic_real_mean_t = rs.double().mean().reshape(())
            critic_fake_mean_t = fs.double().mean().reshape(())
            critic_accuracy_t = (rs > fs).float().mean().reshape(())
        except Exception:
            critic_real_mean_t = zero_scalar
            critic_fake_mean_t = zero_scalar
            critic_accuracy_t = half_scalar
        per_term_tensors = []
        for _, w_t, p_t, tot_t, r_sc_t, f_sc_t in per_term_records:
            per_term_tensors.extend((
                w_t.detach().reshape(()),
                p_t.detach().reshape(()),
                tot_t.detach().reshape(()),
                r_sc_t.detach().double().mean().reshape(()),
                f_sc_t.detach().double().mean().reshape(()),
            ))
    explicit_guard = getattr(trainer.opt_d, 'd_guard_ratio', None)
    for optimizer in trainer.critic_optimizers.values():
        optimizer.step()
    with torch.no_grad():
        if explicit_guard is None and getattr(trainer.opt_d, 'guard', None) is not None:
            try:
                post_c_grads = [p.grad.detach() for p in critic_checked if p.grad is not None]
                post_critic_grad_norm_t = torch.sqrt(sum(g.norm(2) ** 2 for g in post_c_grads)).reshape(()) if post_c_grads else zero_scalar
            except Exception:
                post_critic_grad_norm_t = zero_scalar
        else:
            post_critic_grad_norm_t = zero_scalar
    saved_flags = []
    seen_modules = set()
    for term in terms:
        if id(term.module) in seen_modules:
            continue
        seen_modules.add(id(term.module))
        flags = [parameter.requires_grad for parameter in term.module.parameters()]
        term.module.requires_grad_(False)
        saved_flags.append((term.module, flags))
    rhbi_mod, is_discrete_prior, use_rhbi_simplex = _rhbi_generator_context(trainer, terms)
    import contextlib
    thermal_scope = thermal_coordinator.step_scope() if thermal_coordinator is not None else contextlib.nullcontext()
    try:
        with thermal_scope:
            if independent:
                batch, ids, context = trainer._draw(generator_batch, generator_latent_draw)
            trainer.opt_g.zero_grad(set_to_none=True)
            if rhbi_mod is not None:
                simplicial_active = bool(
                    use_rhbi_simplex
                    and getattr(trainer.recipe, "rhbi_simplex_chart_weight", 0.0) > 0.0
                    and getattr(trainer.recipe, "rhbi_simplex_macro_settle", False)
                )
                rhbi_mod.bind_discrete_prior_step(is_discrete_prior, simplicial_active=simplicial_active)
            simplex_loss = None
            if len(terms) == 1:
                term = terms[0]
                real, fake, real_score, fake_score = _bound_scores(
                    term, context, trainer.graph, "generator", term.generator_phase, first="fake", noise=noise)
                g_adversarial = term.gan.g_loss(fake_score, real_score)
                prior_loss, objective_losses = _generator_tail(trainer, program, context, ids, fake)
                if use_rhbi_simplex and rhbi_mod is not None:
                    unique_ids = ids.unique() if ids is not None else None
                    simplex_loss = _rhbi_simplex_loss(trainer, rhbi_mod, unique_ids, real)
                g_adversarial_weighted = term.generator_weight * g_adversarial
            else:
                unweighted = []
                weighted = []
                fake = None
                for term in terms:
                    real, fake, real_score, fake_score = _bound_scores(
                        term, context, trainer.graph, "generator", term.generator_phase, first="fake", noise=noise)
                    loss = term.gan.g_loss(fake_score, real_score)
                    unweighted.append(loss)
                    weighted.append(term.generator_weight * loss)
                g_adversarial = _sum_tensors(unweighted)
                prior_loss, objective_losses = _generator_tail(trainer, program, context, ids, fake)
                g_adversarial_weighted = _sum_tensors(weighted)
            g_loss = g_adversarial_weighted + prior_loss + sum(objective_losses)
            if simplex_loss is not None:
                g_loss = g_loss + simplex_loss
            from particlegan.optim.constraint_geometry import constraint_geometry_backward
            set_rows = getattr(trainer.opt_g, 'set_sampled_rows', None)
            if set_rows is not None and ids is not None and program.prior_parameters:
                set_rows(trainer.prior.z, ids)
            constraint_geometry_backward(g_loss, trainer.opt_g, (g_adversarial_weighted,))
            gen_checked = list(program.generator_parameters)
            prior_checked = list(program.prior_checked) if program.prior_checked else []
            trainer._refuse_nonfinite(g_loss, "Nonfinite generator loss; run stopped",
                                      [("Nonfinite generator/auxiliary gradient; run stopped", gen_checked),
                                       ("Nonfinite prior gradient; run stopped", prior_checked)])
            with torch.no_grad():
                try:
                    g_grads = [p.grad.detach() for p in gen_checked if p.grad is not None]
                    gen_grad_norm_t = torch.sqrt(sum(g.norm(2) ** 2 for g in g_grads)).reshape(()) if g_grads else zero_scalar
                except Exception:
                    gen_grad_norm_t = zero_scalar
                try:
                    p_grads = [p.grad.detach() for p in prior_checked if p.grad is not None]
                    prior_grad_norm_t = torch.sqrt(sum(g.norm(2) ** 2 for g in p_grads)).reshape(()) if p_grads else zero_scalar
                except Exception:
                    prior_grad_norm_t = zero_scalar
            trainer.opt_g.step()
    finally:
        if rhbi_mod is not None:
            rhbi_mod.bind_discrete_prior_step(False)
        for module, flags in saved_flags:
            for parameter, flag in zip(module.parameters(), flags):
                parameter.requires_grad_(flag)
    update_ema(trainer.ema_graph, trainer.graph, settings["ema"])
    update_ema(trainer.ema_prior, trainer.prior, settings["ema"])
    trainer.step = step
    with torch.no_grad():
        try:
            if hasattr(trainer, 'prior') and hasattr(trainer.prior, 'z') and isinstance(trainer.prior.z, torch.Tensor):
                particle_rms_t = torch.sqrt(torch.mean(trainer.prior.z.detach() ** 2)).reshape(())
            else:
                particle_rms_t = zero_scalar
        except Exception:
            particle_rms_t = zero_scalar
        try:
            if d_fake_samples is not None and d_real_samples is not None:
                diversity_ratio_t = (d_fake_samples.detach().std() / (d_real_samples.detach().std() + 1e-8)).reshape(())
            else:
                diversity_ratio_t = one_scalar
        except Exception:
            diversity_ratio_t = one_scalar

    n_objectives = len(objective_losses)
    values = trainer._metric_transfer([d_loss, d_adversarial, d_adversarial_weighted,
        g_adversarial_weighted, g_loss, g_adversarial, prior_loss, d_penalty,
        critic_real_mean_t, critic_fake_mean_t, critic_accuracy_t,
        critic_grad_norm_t, post_critic_grad_norm_t, gen_grad_norm_t, prior_grad_norm_t,
        particle_rms_t, diversity_ratio_t,
        *objective_losses,
        *per_term_tensors])
    row = dict(zip(("d_loss", "d_adversarial", "d_adversarial_weighted",
                    "g_adversarial_weighted", "g_loss", "g_adversarial",
                    "prior_loss", "gradient_penalty"), values[:8]))
    (raw_real_mean, raw_fake_mean, raw_acc,
     raw_gn_critic, raw_post_gn_critic, raw_gn_gen, raw_gn_prior,
     raw_particle_rms, raw_diversity_ratio) = values[8:17]
    critic_real_mean = float(raw_real_mean) if math.isfinite(raw_real_mean) else 0.0
    critic_fake_mean = float(raw_fake_mean) if math.isfinite(raw_fake_mean) else 0.0
    critic_margin = critic_real_mean - critic_fake_mean
    if not math.isfinite(critic_margin):
        critic_margin = 0.0
    critic_accuracy = float(raw_acc) if math.isfinite(raw_acc) else 0.5
    critic_grad_norm = float(raw_gn_critic) if math.isfinite(raw_gn_critic) else 0.0
    gen_grad_norm = float(raw_gn_gen) if math.isfinite(raw_gn_gen) else 0.0
    prior_grad_norm = float(raw_gn_prior) if math.isfinite(raw_gn_prior) else 0.0
    particle_rms = float(raw_particle_rms) if math.isfinite(raw_particle_rms) else 0.0
    diversity_ratio = float(raw_diversity_ratio) if math.isfinite(raw_diversity_ratio) else 1.0
    if explicit_guard is not None:
        d_guard_ratio = float(explicit_guard or 1.0)
    elif getattr(trainer.opt_d, 'guard', None) is not None and critic_grad_norm > 0.0:
        d_guard_ratio = float(raw_post_gn_critic) / critic_grad_norm if math.isfinite(raw_post_gn_critic) else 1.0
    else:
        d_guard_ratio = 1.0

    row.update(
        event="train", step=step, objectives=values[17:17 + n_objectives], lr_scale=scale,
        critic_real_mean=critic_real_mean,
        critic_fake_mean=critic_fake_mean,
        critic_margin=critic_margin,
        critic_accuracy=critic_accuracy,
        grad_norm_critic=critic_grad_norm,
        grad_norm_generator=gen_grad_norm,
        grad_norm_prior=prior_grad_norm,
        d_guard_ratio=d_guard_ratio,
        particle_rms=particle_rms,
        diversity_ratio=diversity_ratio,
    )
    row["critic/real_mean"] = critic_real_mean
    row["critic/fake_mean"] = critic_fake_mean
    row["critic/margin"] = critic_margin
    row["critic/accuracy"] = critic_accuracy
    row["critic/thermal_scale"] = (
        thermal_coordinator.effective_scale() if thermal_coordinator is not None else 1.0
    )
    row["grad_norm/critic"] = critic_grad_norm
    row["grad_norm/generator"] = gen_grad_norm
    row["grad_norm/prior"] = prior_grad_norm
    row["optimizer/d_guard_ratio"] = d_guard_ratio
    row["prior/particle_rms"] = particle_rms
    row["diversity/ratio"] = diversity_ratio
    per_term_values = values[17 + n_objectives:]
    for idx, (tid, _, _, _, _, _) in enumerate(per_term_records):
        offset = idx * 5
        d_adv_w_v, d_pen_v, d_tot_v, r_mean_v, f_mean_v = per_term_values[offset:offset + 5]
        r_m = float(r_mean_v) if math.isfinite(r_mean_v) else 0.0
        f_m = float(f_mean_v) if math.isfinite(f_mean_v) else 0.0
        margin_v = r_m - f_m
        if not math.isfinite(margin_v):
            margin_v = 0.0
        eff_th, raw_th = (
            thermal_coordinator.term_scales(terms[idx]) if thermal_coordinator is not None else (1.0, 1.0)
        )
        row[f"loss/critics/{tid}/d_adversarial"] = float(d_adv_w_v)
        row[f"loss/critics/{tid}/gradient_penalty"] = float(d_pen_v)
        row[f"loss/critics/{tid}/d_total"] = float(d_tot_v)
        row[f"critic/{tid}/real_mean"] = r_m
        row[f"critic/{tid}/fake_mean"] = f_m
        row[f"critic/{tid}/margin"] = margin_v
        row[f"critic/{tid}/thermal_scale"] = eff_th
        row[f"critic/{tid}/raw_thermal_scale"] = raw_th
    return row, detach(batch)
