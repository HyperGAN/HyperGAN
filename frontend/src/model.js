// The Model tab: what this run trains and how. Everything comes from one
// bounded document, GET /runs/{id}/model, built from the recorded
// configuration; per-layer tables appear when training (or `hypergan model
// RUN --write`) recorded model.json. Run-owned strings only ever reach
// textContent, so the page stays inside the viewer's strict CSP.
const node = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined && text !== null) result.textContent = text;
  if (className) result.className = className;
  return result;
};
const brief = (value, limit = 240) => {
  const text = value === undefined || value === null ? "" : String(value).replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
};
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 });
const exact = new Intl.NumberFormat();
const number = (value) => finite(value) ? (Number.isInteger(value) ? exact.format(value) : String(+value.toPrecision(6))) : "—";
const params = (value) => finite(value) ? (value >= 10000 ? compact.format(value) : exact.format(value)) : "—";
// Scalars read as themselves; small containers as compact JSON on one line.
function show(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return number(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return value;
  if (Array.isArray(value) && value.every((x) => x === null || typeof x !== "object")) return `[${value.map(show).join(", ")}]`;
  return brief(JSON.stringify(value), 400);
}
const shape = (dims) => Array.isArray(dims) ? dims.map((d) => String(d)).join(" × ") : show(dims);
function ports(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return shape(value);
  const entries = Object.entries(value);
  if (entries.length === 1) return shape(entries[0][1]);
  return entries.map(([name, dims]) => `${name}: ${shape(dims)}`).join("; ");
}
function definitions(entries, className = "model-facts") {
  const list = node("dl", undefined, className);
  for (const [label, value] of entries) {
    if (value === undefined) continue;
    const row = node("div");
    row.append(node("dt", label), node("dd", show(value)));
    list.append(row);
  }
  return list;
}
function unavailable(reason, prefix = "Unavailable") {
  return node("p", `${prefix}: ${brief(reason || "no reason recorded", 600)}`, "model-unavailable");
}
function section(title, description) {
  const result = node("section", undefined, "model-section");
  const heading = node("h2", title);
  heading.id = `model-${title.toLowerCase().replace(/[^a-z]+/g, "-")}`;
  result.setAttribute("aria-labelledby", heading.id);
  result.append(heading);
  if (description) result.append(node("p", description, "model-lede"));
  return result;
}
function table(columns, caption) {
  const scroll = node("div", undefined, "table-scroll");
  scroll.tabIndex = 0;
  const element = node("table", undefined, "model-table");
  if (caption) element.append(node("caption", caption, "sr-only"));
  const head = node("tr");
  for (const column of columns) {
    const th = node("th", column);
    th.scope = "col";
    head.append(th);
  }
  const thead = node("thead");
  thead.append(head);
  const body = node("tbody");
  element.append(thead, body);
  scroll.append(element);
  return { scroll, body };
}
function row(body, cells, className) {
  const tr = node("tr", undefined, className);
  for (const cell of cells) {
    const td = node("td");
    if (cell instanceof Node) td.append(cell);
    else td.textContent = cell ?? "—";
    tr.append(td);
  }
  body.append(tr);
  return tr;
}
function sourceBlock(source, label) {
  const details = node("details", undefined, "model-source");
  const summary = node("summary", label);
  const facts = [source.file ? `matches ${source.file}` : source.kind === "template" ? "packaged template" : "inline",
    `${source.lines} lines`];
  if (source.truncated) facts.push("truncated for display");
  summary.append(node("span", facts.join(" · ")));
  const pre = node("pre");
  const code = node("code");
  // One element per line so CSS counters number them; text stays textContent.
  for (const line of String(source.text || "").split("\n")) code.append(node("span", line, "model-line"));
  pre.append(code);
  details.append(summary, pre);
  if (source.parameters && Object.keys(source.parameters).length)
    details.append(definitions(Object.entries(source.parameters), "model-facts model-parameters"));
  return details;
}

export function modelPanel(api, base, { catalog, onPlot }) {
  let loaded = null;
  let rendered = null;
  let pending = null;
  const root = () => document.getElementById("model-content");
  const status = (text) => {
    const element = document.getElementById("model-status");
    element.textContent = text || "";
    element.hidden = !text;
  };
  function seriesCell(metric, note) {
    const cell = node("span", undefined, "model-series");
    if (!metric) {
      cell.append(node("small", note || "no series"));
      return cell;
    }
    const known = metric in (catalog()?.metrics || {});
    if (known) {
      const button = node("button", metric, "text-button");
      button.type = "button";
      button.title = "Show this series under Metrics";
      button.addEventListener("click", () => onPlot(metric));
      cell.append(button);
    } else cell.append(node("code", metric));
    if (note) cell.append(node("small", note));
    return cell;
  }
  function renderFormulation(model) {
    const f = model.formulation || {};
    const result = section("GAN formulation", "The adversarial game and critic penalty this run optimizes.");
    const card = node("article", undefined, "model-card");
    const title = node("h3", f.name || "Unknown formulation");
    title.append(node("span", f.family === "legacy" ? "recorded fields" : f.family || "", "model-badge"));
    card.append(title, node("p", `${f.loss || ""} · ParticleGAN ${f.particlegan || "unknown"}`, "model-quiet"));
    if (f.note) card.append(node("p", f.note, "model-note"));
    card.append(definitions(Object.entries(f.parameters || {})));
    const equations = Object.entries(f.equations || {});
    if (equations.length) {
      const details = node("details", undefined, "model-equations");
      details.open = true;
      details.append(node("summary", "Equations"), definitions(equations, "model-facts model-math"));
      card.append(details);
    }
    const prior = model.prior || {};
    const priorCard = node("article", undefined, "model-card");
    priorCard.append(node("h3", `Prior · ${prior.kind || "unknown"}`));
    const sigma = prior.sigma;
    priorCard.append(definitions([
      ["Latent dimension", prior.z_dim], ["Particles", prior.num_particles], ["Learnable", prior.learnable],
      ["Initialization", prior.initialization ? `${show(prior.initialization.device)} · seed ${show(prior.initialization.seed)}` : undefined],
      ["σ", sigma ? (sigma.mode === "calibrated" ? `calibrated from sigma_rel ${show(sigma.sigma_rel)}` : `${show(sigma.value)} (${sigma.mode})`) : undefined],
    ]));
    const grid = node("div", undefined, "model-grid");
    grid.append(card, priorCard);
    result.append(grid);
    return result;
  }
  function renderLosses(model) {
    const losses = model.losses || {};
    const result = section("Losses", "Each player's terms, their weights, and the metric series that records them.");
    for (const [side, title] of [["discriminator", "Discriminator (critic)"], ["generator", "Generator and prior"]]) {
      const terms = losses[side] || [];
      result.append(node("h3", title, "model-subtitle"));
      const { scroll, body } = table(["Term", "Kind", "Weight", "Details", "Series"], `${title} loss terms`);
      for (const term of terms) {
        const weight = term.weight ?? term.coeff;
        const details = [];
        if (term.critic) details.push(`critic ${term.critic}`);
        if (term.real || term.fake) details.push(`real ${show(term.real)} · fake ${show(term.fake)}`);
        if (term.inputs && Object.keys(term.inputs).length) details.push(Object.entries(term.inputs).map(([k, v]) => `${k} ← ${v}`).join(", "));
        for (const key of ["kappa", "lazy_k", "anchor_weight", "anchor_decay", "arm", "norm", "target_std", "rows"])
          if (term[key] !== undefined && term[key] !== null) details.push(`${key} ${show(term[key])}`);
        if (term.detach?.length) details.push(`detached: ${term.detach.join(", ")}`);
        if (term.active === false) details.push("inactive (weight 0)");
        row(body, [term.id, term.kind || "—", `${term.coeff !== undefined ? "c = " : ""}${show(weight)}`,
          details.join(" · ") || "—", seriesCell(term.metric, term.metric_note)],
          term.active === false ? "model-inactive" : undefined);
      }
      if (!terms.length) row(body, ["No terms recorded", "", "", "", ""]);
      result.append(scroll);
    }
    const totals = node("p", undefined, "model-totals");
    totals.append(node("span", "Totals"));
    for (const [label, metric] of Object.entries(losses.totals || {})) {
      const item = node("span");
      item.append(node("small", label), seriesCell(metric));
      totals.append(item);
    }
    result.append(totals);
    return result;
  }
  function renderOptimizers(model) {
    const o = model.optimizers || {};
    const result = section("Optimizers & hyperparameters", "Learning rates, schedule, noise and averaging.");
    const { scroll, body } = table(["Group", "Optimizer", "Learning rate", "Betas", "Updates"], "Optimizer groups");
    const g = o.generator || {}, p = o.prior || {}, c = o.critic || {};
    row(body, ["Generator", `${show(g.type)} · ${show(g.implementation)}`, show(g.lr), show(g.betas), (g.components || []).join(", ") || "—"]);
    row(body, ["Prior", "shares the generator step", `${show(p.lr)} (×${show(p.lr_mult)})`, show(p.betas), "particle table"]);
    row(body, ["Critic", `${show(c.type)} · ${show(c.implementation)}`, `${show(c.lr)} (×${show(c.lr_mult)})`, show(c.betas), (c.components || []).join(", ") || "—"]);
    result.append(scroll);
    const grid = node("div", undefined, "model-grid");
    const schedule = node("article", undefined, "model-card");
    schedule.append(node("h3", "Schedule"), definitions([
      ["LR anneal start", o.schedule?.lr_anneal_start], ["LR floor", o.schedule?.lr_floor],
      ["Network LR floor", o.schedule?.network_lr_floor], ["Network LR horizon cap", o.schedule?.network_lr_horizon_cap],
      ["EMA (G and prior)", o.ema], ["Critic EMA anchor", c.ema_critic],
      ["Critic guard", c.guard ? `ratio ${show(c.guard.ratio)} after ${show(c.guard.min_steps)} steps` : undefined],
      ["Latent damping max rate", g.latent_damping_max_rate],
    ]));
    if (o.schedule?.metric) schedule.append(seriesCell(o.schedule.metric));
    const noise = node("article", undefined, "model-card");
    noise.append(node("h3", "Noise"), definitions([
      ["Critic input σ", o.noise?.critic_input_std], ["Critic input anneal end", o.noise?.critic_input_anneal_end],
      ["Generator output σ", o.noise?.generator_output_std], ["Generator output warmup", o.noise?.generator_output_warmup],
    ]));
    grid.append(schedule, noise);
    result.append(grid);
    return result;
  }
  function layerTable(entry, subgraph) {
    const nodes = subgraph.nodes || [];
    const details = node("details", undefined, "model-layers");
    const label = `Layers · ${exact.format(subgraph.node_count ?? nodes.length)} nodes`;
    const summary = node("summary", subgraph.module_path && !["network", "(config)", "(root)"].includes(subgraph.module_path) ? `${label} · ${subgraph.module_path}` : label);
    if (subgraph.parameters) summary.append(node("span", `${params(subgraph.parameters.total)} parameters`));
    details.append(summary);
    const shapes = nodes.some((n) => n.out);
    const { scroll, body } = table(["Line", "Node", "Op", "Output shape", "Parameters", "Trainable"], `${entry.name} layers`);
    // Comment paragraphs in the source become group rows, matched by line.
    const sections = entry.source?.sections && (entry.graph?.subgraphs || []).length === 1 ? [...entry.source.sections] : [];
    for (const item of nodes) {
      while (sections.length && finite(item.line) && sections[0].line <= item.line) {
        const heading = row(body, [sections.shift().title], "model-group");
        heading.firstChild.colSpan = 6;
      }
      const name = node("span", item.id);
      if (item.explicit?.length) name.title = `explicit: ${item.explicit.join(", ")}`;
      const op = node("span", item.op, `model-op model-op-${String(item.category || "other").replace(/[^a-z]/g, "")}`);
      if (item.category) op.title = item.category;
      row(body, [finite(item.line) ? String(item.line) : "—", name, op, shapes ? ports(item.out) : "—",
        finite(item.params) ? params(item.params) : "—",
        item.trainable === false || (finite(item.params) && item.params > 0 && item.trainable_params === 0) ? "frozen" : finite(item.params) && item.params > 0 ? "yes" : "—"]);
    }
    if (subgraph.nodes_truncated) row(body, [`Showing the first ${nodes.length} nodes`]);
    details.append(scroll);
    return details;
  }
  function renderNetwork(entry) {
    const card = node("article", undefined, "model-card model-network");
    const title = node("h3", entry.name);
    title.append(node("span", entry.role, `model-badge model-role-${entry.role}`));
    card.append(title);
    const graph = entry.graph || {};
    const facts = [["Factory", entry.factory], ["Optimizer group", entry.optimizer_group]];
    if (entry.reuse_of) facts.push(["Shares weights with", entry.reuse_of]);
    if (entry.source?.file) facts.push(["Source file", entry.source.file]);
    else if (entry.source?.text) facts.push(["Source", "inline HNDL (file name not recorded)"]);
    if (entry.input_shape !== undefined) facts.push(["Input → output", `${ports(entry.input_shape)} → ${ports(entry.output_shape)}`]);
    const first = (graph.subgraphs || [])[0];
    if (entry.input_shape === undefined && first?.input_shape) facts.push(["Input → output", `${ports(first.input_shape)} → ${ports(first.output_shape)}`]);
    if (graph.parameters) facts.push(["Parameters", `${params(graph.parameters.total)} total · ${params(graph.parameters.trainable)} trainable · ${params(graph.parameters.frozen)} frozen`]);
    if (entry.trainable === false) facts.push(["Trainable", false]);
    const bindings = Object.entries(entry.inputs || {});
    if (bindings.length) facts.push(["Inputs", bindings.map(([k, v]) => `${k} ← ${v}`).join(", ")]);
    card.append(definitions(facts));
    if (graph.status === "captured") card.append(unavailable(graph.reason, "Shapes unavailable"));
    else if (graph.status === "unavailable") card.append(unavailable(graph.reason, entry.reuse_of ? "Alias" : "Layer detail unavailable"));
    for (const subgraph of graph.subgraphs || []) card.append(layerTable(entry, subgraph));
    if (graph.status === "built-no-hndl") card.append(node("p", "Python module without an HNDL graph; only parameter totals are known.", "model-quiet"));
    if (entry.source?.text !== undefined) card.append(sourceBlock(entry.source, "HNDL source"));
    else if (entry.source?.status === "unavailable" && !entry.templates?.length) card.append(unavailable(entry.source.reason, "Source unavailable"));
    for (const template of entry.templates || []) card.append(sourceBlock(template, `Template ${template.name}`));
    if (entry.args && Object.keys(entry.args).length) {
      const details = node("details", undefined, "model-source");
      details.append(node("summary", "Constructor arguments"), definitions(Object.entries(entry.args)));
      card.append(details);
    }
    return card;
  }
  function renderNetworks(model) {
    const result = section("Networks", "One card per component: role, contract, parameters, layers and source.");
    const recorded = model.networks_recorded;
    if (recorded) result.append(node("p", `Layer detail ${recorded.origin === "backfill" ? "backfilled with hypergan model" : "recorded at training start"}${recorded.hndl ? ` · hndl ${recorded.hndl}` : ""}.`, "model-quiet"));
    const list = node("div", undefined, "model-networks");
    for (const entry of model.networks || []) list.append(renderNetwork(entry));
    result.append(list);
    return result;
  }
  function renderSettings(model) {
    const result = section("Data & training", "Everything else the configuration fixes for this run.");
    const grid = node("div", undefined, "model-grid");
    const data = node("article", undefined, "model-card");
    data.append(node("h3", "Data"), definitions([["Factory", model.data?.factory], ...Object.entries(model.data?.args || {})]));
    const training = node("article", undefined, "model-card");
    const run = model.run || {};
    training.append(node("h3", "Training"), definitions([
      ...Object.entries(model.training || {}).filter(([key]) => !["lr_anneal_start", "lr_floor", "network_lr_floor", "network_lr_horizon_cap", "ema",
        "input_noise_std", "input_noise_anneal_end", "output_noise_std", "output_noise_warmup"].includes(key)),
      ["Global batch size", run.global_batch_size ?? undefined]]));
    const sampling = node("article", undefined, "model-card");
    sampling.append(node("h3", "Sampling & metrics"), definitions([...Object.entries(model.sampling || {}),
      ["Metric preset", model.metrics?.preset], ["Custom metrics", (model.metrics?.custom || []).join(", ") || "none"]]));
    grid.append(data, training, sampling);
    result.append(grid);
    if (model.warnings?.length) {
      result.append(node("h3", "Configuration warnings", "model-subtitle"));
      const list = node("ul", undefined, "model-warnings");
      for (const warning of model.warnings) list.append(node("li", warning));
      result.append(list);
    }
    return result;
  }
  function render(model) {
    const header = node("header", undefined, "model-header");
    const p = model.provenance || {};
    header.append(node("p", [
      model.config_sha256 ? `Configuration ${model.config_sha256.slice(0, 12)}` : "Configuration",
      p.hypergan_commit ? `HyperGAN ${String(p.hypergan_commit).slice(0, 8)}${p.dirty ? " (dirty)" : ""}` : null,
      p.hndl ? `hndl ${p.hndl}` : null, p.torch ? `torch ${p.torch}` : null,
      p.defaults_source ? `defaults: ${p.defaults_source}` : null,
    ].filter(Boolean).join(" · "), "model-quiet"));
    const nav = node("nav", undefined, "model-jump");
    nav.setAttribute("aria-label", "Model sections");
    const sections = [renderFormulation(model), renderLosses(model), renderOptimizers(model), renderNetworks(model), renderSettings(model)];
    for (const item of sections) {
      const heading = item.querySelector("h2");
      const link = node("a", heading.textContent);
      link.href = `#${heading.id}`;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        heading.scrollIntoView({ block: "start" });
      });
      nav.append(link);
    }
    header.append(nav);
    root().replaceChildren(header, ...sections);
  }
  async function load({ force = false } = {}) {
    if (pending) return pending;
    if (loaded && !force) return loaded;
    status("Loading the model description…");
    pending = (async () => {
      try {
        const { data } = await api(`${base()}/model`);
        // A metadata refresh that changed nothing keeps open sections as they are.
        const identity = JSON.stringify({ ...data, seconds: undefined });
        if (identity !== rendered) render(data);
        rendered = identity;
        status("");
        loaded = data;
        return data;
      } catch (error) {
        loaded = null;
        rendered = null;
        root().replaceChildren();
        status(/not found|configuration/i.test(error.message)
          ? "No model configuration is recorded for this run."
          : `Model description unavailable: ${error.message}`);
        return null;
      } finally {
        pending = null;
      }
    })();
    return pending;
  }
  return { load, invalidate() { loaded = null; } };
}
