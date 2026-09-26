"""Lower a graph of references and losses to the existing configuration.

The lowered tables are exactly what ``hypergan train config.toml`` reads, so
fingerprints, resume, the viewer and replicated execution treat a model built
in Python like any other config. The saved file is the durable artifact.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re

from ..config import resolve_config_at, fingerprint as _fingerprint
from ..metric_plugins import SCALAR_INPUTS
from . import toml_writer
from .nodes import (Adversarial, DataSource, Field, Latent, Net, Objective, Ref, Select, _Candidate,
                    adversarial, as_source, candidate, first_parameter, hndl, net, plain)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

GENERATOR, DISCRIMINATOR = "generator", "discriminator"


@dataclass(frozen=True)
class Model:
    """A lowered model: the raw config tables plus derived network roles.

    ``config`` is plain data (what the file stores). ``base`` is the directory
    that relative ``.hndl`` file paths resolve against. ``roles`` maps each
    network name to generator, critic, generator-side or frozen.
    """
    config: dict
    roles: dict = field(default_factory=dict)
    base: Path = None
    path: Path = None
    notes: tuple = ()


# ---------------------------------------------------------------------------
# Lowering


class _Lowering:
    def __init__(self, source, losses, metrics, evaluations, samplers):
        self.source = as_source(source)
        self.losses = list(losses)
        self.metrics, self.evaluations, self.samplers = list(metrics), list(evaluations), list(samplers)
        self.notes = []
        adversarials = [term for term in self.losses if isinstance(term, Adversarial)]
        if not adversarials:
            raise ValueError("A model needs at least one hg.adversarial(critic, real=..., fake=...) loss")
        unknown = [term for term in self.losses if not isinstance(term, (Adversarial, Objective))]
        if unknown:
            raise TypeError(f"losses must be hg.adversarial/hg.l1/hg.mse/hg.loss terms, got {unknown!r}")
        self.primary, self.extra = adversarials[0], adversarials[1:]
        self.objectives = [term for term in self.losses if isinstance(term, Objective)]
        if not isinstance(self.primary.fake, Net) or self.primary.fake.reuse_of is not None:
            raise ValueError("The first adversarial loss's fake must be a network output (it becomes the "
                             "generator); select nested outputs inside the critic instead")
        if not isinstance(self.primary.real, Field):
            raise ValueError("The first adversarial loss's real must be a data field, e.g. real=hg.data('x')")
        self.generator, self.critic = self.primary.fake, self.primary.critic
        self.real_field = self.primary.real.field
        if self.source.kind != "items" and self.real_field != "real":
            raise ValueError(f"{self.source.factory} produces the real sample as field 'real'; "
                             f"use hg.data('real') (got {self.real_field!r}), or an hg.dataset whose items name it")
        if self.source.kind == "items" and self.source.args.get("real", self.real_field) != self.real_field:
            raise ValueError("hg.dataset(real=...) disagrees with the first adversarial loss's real field")
        self.critics = []
        for term in adversarials:
            if term.critic not in self.critics:
                self.critics.append(term.critic)
        self.nets = []
        self.latents = []
        for ref in self._roots():
            self._collect(ref)
        self._name()

    # Traversal ---------------------------------------------------------------
    def _roots(self):
        for term in self.losses:
            if isinstance(term, Adversarial):
                yield term.critic
                yield term.real
                yield term.fake
            else:
                yield from term.inputs.values()
        for item in self.samplers:
            yield from item.inputs.values()
        for item in self.evaluations:
            if item.output is not None:
                yield item.output

    def _collect(self, ref):
        if isinstance(ref, Select):
            return self._collect(ref.parent)
        if isinstance(ref, Latent):
            if ref not in self.latents:
                self.latents.append(ref)
            return
        if isinstance(ref, Net):
            if ref in self.nets:
                return
            if ref.reuse_of is not None:
                self._collect(ref.reuse_of)
            for value in ref.inputs.values():
                self._collect(value)
            self.nets.append(ref)

    def _upstream(self, refs):
        seen = []
        def visit(ref):
            if isinstance(ref, Select):
                return visit(ref.parent)
            if isinstance(ref, Net) and ref not in seen:
                seen.append(ref)
                if ref.reuse_of is not None:
                    visit(ref.reuse_of)
                for value in ref.inputs.values():
                    visit(value)
        for ref in refs:
            visit(ref)
        return seen

    # Names -------------------------------------------------------------------
    def _name(self):
        self.names = {}
        for node, fixed in ((self.generator, GENERATOR), (self.critic, DISCRIMINATOR)):
            if node.name not in (None, fixed):
                raise ValueError(f"{node.name!r} is the model's {fixed} (it is the "
                                 f"{'fake' if fixed == GENERATOR else 'critic'} of the first adversarial loss); "
                                 f"its checkpoint key is {fixed!r}, so drop name= or pass name={fixed!r}")
            self.names[node] = fixed
        claimed = {}
        for node in self.nets:
            name = self.names.get(node) or node.name or (None if node.hndl else node.default_name)
            if name is None:
                raise ValueError(f"An hg.hndl network ({_where(node, self)}) needs name=...: its name is its "
                                 "checkpoint key and metric prefix, so it is never generated from call order")
            if node not in self.names and name in (GENERATOR, DISCRIMINATOR):
                raise ValueError(f"The name {name!r} is reserved for the first adversarial pair; give this "
                                 f"network another name=... ({_where(node, self)})")
            if not name.isidentifier():
                raise ValueError(f"Network name {name!r} must be a Python identifier")
            claimed.setdefault(name, []).append(node)
            self.names[node] = name
        clashes = {name: nodes for name, nodes in claimed.items() if len(nodes) > 1}
        if clashes:
            name, nodes = next(iter(clashes.items()))
            raise ValueError(f"{len(nodes)} networks would be named {name!r} ({', '.join(_where(n, self) for n in nodes)}). "
                             "Names are checkpoint keys, so pass name=... to each; they are not numbered by call order")

    # Paths -------------------------------------------------------------------
    def path(self, ref, term=False):
        if isinstance(ref, Select):
            return f"{self.path(ref.parent, term)}.{ref.key}"
        if isinstance(ref, Field):
            return "batch.real" if ref.field == self.real_field else f"batch.{ref.field}"
        if isinstance(ref, Latent):
            return "latent"
        if isinstance(ref, _Candidate):
            return "candidate"
        if isinstance(ref, Net):
            if term and ref is self.generator:
                return "generated"
            return f"components.{self.names[ref]}"
        raise TypeError(f"Not a graph reference: {ref!r}")

    def shape(self, ref):
        if isinstance(ref, Field):
            return ref.shape or self.source.shapes.get(ref.field)
        if isinstance(ref, _Candidate):
            return self.shape(self.primary.real)
        return ref.shape

    # Tables ------------------------------------------------------------------
    def components(self):
        critic_inputs = [v for c in self.critics for v in c.inputs.values() if not isinstance(v, _Candidate)]
        generator_side = self._upstream([self.generator]
                                        + [v for t in self.objectives for v in t.inputs.values()]
                                        + [r for t in self.extra for r in (t.real, t.fake)])
        conditioning = [n for n in self._upstream(critic_inputs) if n not in generator_side and n not in self.critics]
        observed = [n for n in self._upstream([v for s in self.samplers for v in s.inputs.values()]
                                              + [e.output for e in self.evaluations if e.output is not None])
                    if n not in generator_side and n not in self.critics and n not in conditioning]
        specs = {}
        for node in self.nets:
            name = self.names[node]
            for value in node.inputs.values():
                if isinstance(value, _Candidate) and node not in self.critics:
                    raise ValueError(f"{name} reads hg.candidate but no hg.adversarial loss scores it")
            if node in self.critics and sum(isinstance(v, _Candidate) for v in node.inputs.values()) != 1:
                raise ValueError(f"Critic {name} must take hg.candidate as exactly one input")
            spec = {}
            if node.reuse_of is not None:
                spec["reuse"] = self.names[node.reuse_of]
                if node.freeze:
                    spec["freeze_parameters"] = True
            else:
                spec["factory"] = node.factory
                args = deepcopy(node.args)
                if node.hndl and "input_shape" not in args:
                    args["input_shape"] = self._input_shape(node, name)
                if args:
                    spec["args"] = args
            spec["inputs"] = {arg: self.path(ref) for arg, ref in node.inputs.items()}
            trainable = node.trainable
            if trainable is None and node.reuse_of is None and (node in conditioning or node in observed):
                trainable = False
                why = "critic conditioning" if node in conditioning else "samplers/evaluations"
                self.notes.append(f"{name} is frozen: it is only used by {why}, so no loss trains it")
            if trainable is not None and node.reuse_of is None:
                spec["trainable"] = trainable
            specs[name] = spec
        first = [GENERATOR, DISCRIMINATOR]
        return {name: specs[name] for name in first + [n for n in specs if n not in first]}

    def _input_shape(self, node, name):
        shapes = {}
        for arg, ref in node.inputs.items():
            shape = self.shape(ref)
            if shape is None:
                raise ValueError(f"Cannot infer the shape of input {arg!r} of HNDL network {name}; "
                                 f"pass input_shape=... or give the data field a shape: hg.data({arg!r}, shape=[...])")
            shapes[arg] = shape
        return shapes["x"] if list(shapes) == ["x"] else shapes

    def adversarial_terms(self):
        terms, ids = [], {}
        for term in self.extra:
            component = self.names[term.critic]
            ident = term.name or component
            if ident in ids:
                raise ValueError(f"Two adversarial losses would publish as {ident!r}; pass name=... to each")
            ids[ident] = term
            spec = {"id": ident, "component": component, "weight": term.weight,
                    "real": self.path(term.real, True), "fake": self.path(term.fake, True)}
            if term.penalty not in (None, False, 0, 0.0):
                spec["penalty"] = True
                if term.penalty is not True:
                    spec["penalty_coeff"] = plain(term.penalty, "adversarial penalty")
            terms.append(spec)
        return terms

    def objective_tables(self):
        tables, ids = [], {}
        for term in self.objectives:
            ident = term.name or term.default_name
            if ident in ids:
                raise ValueError(f"Two losses would publish as loss/objectives/{ident}; pass name=... to each")
            ids[ident] = term
            bad = [k for k in term.detach if k not in term.inputs]
            if bad:
                raise ValueError(f"loss {ident}: detach names {bad} are not its inputs")
            spec = {"id": ident, "factory": term.factory, "inputs": {k: self.path(v, True) for k, v in term.inputs.items()},
                    "weight": term.weight, "detach": list(term.detach)}
            if term.args:
                spec["args"] = deepcopy(term.args)
            tables.append(spec)
        return tables

    def data_table(self, source, real=None):
        args = deepcopy(source.args)
        if source.kind == "items":
            args.setdefault("real", real or self.real_field)
        return {"factory": source.factory, "args": args}

    def metric_tables(self):
        custom = {}
        for item in self.metrics:
            if set(item.inputs.values()) - SCALAR_INPUTS:
                raise ValueError(f"metric {item.name} binds unsupported update scalars")
            custom[_unique(custom, item.name, "metric")] = {
                "factory": item.factory, "args": item.args, "inputs": item.inputs,
                "mode": "scalar", "every_steps": item.every}
        for item in self.evaluations:
            spec = {"factory": item.factory, "args": item.args, "inputs": item.inputs, "mode": "snapshot",
                    "on_error": item.on_error,
                    "evaluation": {"data": self.data_table(as_source(item.data)), "sample_count": item.samples,
                                   "batch_size": item.batch_size, "seed": item.seed, "device": item.device}}
            if item.every is None:
                spec["trigger"] = "manual"
            else:
                spec.update(trigger="interval", every_steps=item.every)
            if item.output is not None:
                spec["evaluation"]["generated"] = self.path(item.output, True)
            custom[_unique(custom, item.name, "evaluation")] = spec
        return custom

    def sampler_tables(self):
        tables = {}
        for item in self.samplers:
            spec = {"factory": item.factory, "args": item.args,
                    "inputs": {k: self.path(v, True) for k, v in item.inputs.items()}, "count": item.count}
            if item.seed is not None:
                spec["seed"] = item.seed
            tables[_unique(tables, item.name, "sampler")] = spec
        return tables

    def prior_table(self):
        if len(self.latents) > 1:
            raise ValueError(f"The engine has one prior per model; this graph uses {len(self.latents)} "
                             "hg.latent(...) objects. Reuse one latent and slice it inside the network")
        latent = self.latents[0] if self.latents else Latent(64)
        return {"kind": latent.kind, "args": deepcopy(latent.args)}


def _unique(table, name, what):
    if name in table:
        raise ValueError(f"Two {what}s are named {name!r}; pass name=... to each")
    return name


def _where(node, lowering):
    inputs = ", ".join(f"{k}={lowering.path(v) if not isinstance(v, Net) or v in lowering.names else v.describe()}"
                       for k, v in node.inputs.items())
    return f"{node.factory}({inputs})"


def _merge(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def model(data, losses=None, *, generator=None, discriminator=None, latent=None, metrics=(), evaluations=(),
          samplers=(), name="custom/graph", settings=None, steps=None, batch_size=None, device=None, seed=None, lr=None):
    """Build a model from a data source and a list of losses.

    ``hg.model(data, [hg.adversarial(d, real=x, fake=g), hg.l1(g, x)])``, or the
    shorthand ``hg.model(data, generator=G, discriminator=D)`` for one pair.
    Roles come from the losses; ``settings`` overrides any config section,
    e.g. ``{"optimizer": {"lr": 1e-4}}``.
    """
    if (losses is None) == (generator is None and discriminator is None):
        raise ValueError("Pass either losses=[...] or generator=... and discriminator=...")
    if losses is None:
        losses = [_shorthand(data, generator, discriminator, latent)]
    lowering = _Lowering(data, losses, metrics, evaluations, samplers)
    config = {"schema_version": 1, "name": name, "data": lowering.data_table(lowering.source)}
    config["components"] = lowering.components()
    config["prior"] = lowering.prior_table()
    primary = lowering.primary
    if primary.name is not None:
        raise ValueError("The first adversarial loss publishes as loss/*_adversarial; name= applies to further critics")
    if primary.weight != 1.0:
        config["adversarial"] = {"weight": primary.weight}
    if primary.penalty is not None and primary.penalty is not True:
        config["gradient_penalty"] = {"coeff": 0.0 if primary.penalty is False else plain(primary.penalty, "penalty")}
    if config["prior"]["kind"] == "gaussian":
        config["prior_regularizer"] = {"weight": 0.0}
        lowering.notes.append("A Gaussian latent has no learned table, so prior_regularizer.weight is 0")
    objectives = lowering.objective_tables()
    if objectives:
        config["objectives"] = objectives
    terms = lowering.adversarial_terms()
    if terms:
        config["adversarial_terms"] = terms
    custom = lowering.metric_tables()
    if custom:
        config["metrics"] = {"custom": custom}
    samplers_table = lowering.sampler_tables()
    if samplers_table:
        config["samplers"] = samplers_table
    training = {k: v for k, v in (("steps", steps), ("batch_size", batch_size), ("device", device), ("seed", seed)) if v is not None}
    if training:
        config["training"] = training
    if lr is not None:
        config["optimizer"] = {"lr": lr}
    if settings:
        _merge(config, plain(settings, "settings"))
    result = Model(config=config, notes=tuple(lowering.notes))
    resolved = resolve(result)
    return Model(config=config, roles=roles(resolved), notes=tuple(lowering.notes))


def _shorthand(data, generator, discriminator, latent):
    source = as_source(data)
    if generator is None or discriminator is None:
        raise ValueError("The shorthand needs both generator= and discriminator=")
    z = latent if isinstance(latent, Latent) else Latent(latent or 64)
    real = Field("real")
    fake = _as_net(generator, z, source.shapes.get("real"), "generator")
    critic = _as_net(discriminator, candidate, ["B", 1], "discriminator", source.shapes.get("real"))
    return adversarial(critic, real=real, fake=fake)


def _as_net(value, input_ref, shape, role, input_shape=None):
    if isinstance(value, Net):
        return value
    if isinstance(value, str) and ":" not in value:
        if shape is None:
            raise ValueError(f"The {role} HNDL source needs an output shape; use hg.hndl(source, shape=..., x=...)")
        kwargs = {"input_shape": input_shape} if input_shape is not None else {}
        return hndl(value, shape=shape, x=input_ref, **kwargs)
    target = value
    if isinstance(value, str):
        import importlib
        module, qualname = value.split(":")
        target = importlib.import_module(module)
        for part in qualname.split("."):
            target = getattr(target, part)
    return net(value, **{first_parameter(target): input_ref})


# ---------------------------------------------------------------------------
# Roles, resolution and files


def roles(config):
    """Each network's role, derived from the resolved losses (never from class names)."""
    critics = ["discriminator"] + [t["component"] for t in config.get("adversarial_terms") or ()]
    result = {}
    for name, spec in config["components"].items():
        if name == "generator":
            result[name] = "generator"
        elif name in critics:
            result[name] = "critic"
        elif "reuse" in spec:
            result[name] = f"reuses {spec['reuse']}"
        elif not spec.get("trainable", True):
            result[name] = "frozen"
        else:
            result[name] = "generator-side"
    return result


def resolve(model):
    """The resolved configuration exactly as training will see it (Torch-free)."""
    return resolve_config_at(model.config, model.base or Path.cwd())


def fingerprint(model):
    """The numerical identity of the model (the run's config_sha256)."""
    return _fingerprint(resolve(model))


_HASH = re.compile(r"^# body-sha256: ([0-9a-f]{64})$", re.M)


def _body(config, directory):
    config = deepcopy(config)
    for spec in config.get("components", {}).values():
        file = spec.get("args", {}).get("file")
        if file and Path(file).is_absolute():
            relative = os.path.relpath(file, directory)
            # Relative to the config when they share a project; otherwise keep it absolute.
            if Path(relative).parts[:3] != ("..", "..", ".."):
                spec["args"]["file"] = relative
    return toml_writer.dumps(config)


def save(model, path, *, overwrite=False):
    """Write the model's config file (TOML); ``hypergan train PATH`` runs it as is.

    Networks, data, losses and observations are referenced by import path,
    never copied. The file records a hash of what this function wrote; saving
    again replaces it only if nobody edited it since, so the Python and the
    file cannot silently diverge. ``overwrite=True`` replaces an edited file.
    """
    path = Path(path)
    if path.suffix != ".toml":
        raise ValueError("Config files use the .toml suffix")
    body = _body(model.config, path.parent.resolve())
    digest = hashlib.sha256(body.encode()).hexdigest()
    header = ("HyperGAN config written by hypergan.graph (hg.save). Train it with `hypergan train` or hg.train.\n"
              "Networks and data are referenced by import path; edit those modules, not copies.\n"
              "hg.save replaces this file only while it is unedited; after editing it here, load it with hg.load.\n"
              f"body-sha256: {digest}\n")
    text = toml_writer.dumps({}, header=header).rstrip("\n") + "\n\n" + body
    if path.exists() and not overwrite:
        existing = path.read_text()
        match = _HASH.search(existing)
        if existing != text:
            if match is None:
                raise FileExistsError(f"{path} exists and was not written by hg.save; pass overwrite=True or choose another path")
            current = existing.split("\n\n", 1)[1] if "\n\n" in existing else ""
            if hashlib.sha256(current.encode()).hexdigest() != match.group(1):
                raise FileExistsError(f"{path} was edited after hg.save wrote it; load it with hg.load(path) "
                                      "to train the edited version, or pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path


def load(path):
    """Read a config file (written by hg.save or by hand) as a Model."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    base = path.parent.resolve()
    result = Model(config=config, base=base, path=path.resolve())
    return Model(config=config, roles=roles(resolve(result)), base=base, path=path.resolve())


def describe(model):
    """A short text summary: networks with roles and inputs, losses and observations."""
    resolved = resolve(model)
    lines = [f"model {resolved['name']}  fingerprint {_fingerprint(resolved)[:12]}"]
    lines.append(f"  data     {resolved['data']['factory']}")
    lines.append(f"  latent   {resolved['prior']['kind']} z_dim={resolved['prior']['args']['z_dim']}")
    for name, role in model.roles.items():
        spec = resolved["components"][name]
        inputs = ", ".join(f"{k}={v}" for k, v in spec["inputs"].items())
        lines.append(f"  net      {name:<14} {role:<15} {spec.get('factory', 'reuse')}({inputs})")
    lines.append(f"  loss     adversarial     discriminator  weight={resolved['adversarial']['weight']}")
    for term in resolved.get("adversarial_terms", ()):
        lines.append(f"  loss     adversarial     {term['component']:<14} id={term['id']} weight={term['weight']}")
    for term in resolved["objectives"]:
        lines.append(f"  loss     {term['factory'].split(':')[-1]:<15} id={term.get('id')} weight={term['weight']} "
                     f"inputs={term['inputs']}")
    for metric_id, spec in resolved["metrics"]["custom"].items():
        lines.append(f"  observe  {spec['mode']:<15} {metric_id}")
    for sampler_id in resolved.get("samplers", {}):
        lines.append(f"  observe  sampler         {sampler_id}")
    lines.extend(f"  note     {note}" for note in model.notes)
    return "\n".join(lines)
