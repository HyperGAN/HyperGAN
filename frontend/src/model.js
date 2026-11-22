// The Model tab: what this run trains and how. Everything comes from one
// bounded document, GET /runs/{id}/model, built from the recorded
// configuration; per-layer tables appear when training (or `hypergan model
// RUN --write`) recorded model.json. Run-owned strings only ever reach
// textContent, so the page stays inside the viewer's strict CSP.
const SVG_NS = "http://www.w3.org/2000/svg";
const node = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined && text !== null) result.textContent = text;
  if (className) result.className = className;
  return result;
};
const svgNode = (tag, attrs = {}) => {
  const result = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) result.setAttribute(k, String(v));
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
const compactCount = (value) => finite(value) ? (value >= 1000 ? compact.format(value) : exact.format(value)) : "—";
const rate = (value) => finite(value) ? (Number.isInteger(value) ? String(value) : String(+value.toPrecision(4))) : "—";

// Scalars read as themselves; small containers as compact JSON on one line.
function show(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return number(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return value;
  if (Array.isArray(value) && value.every((x) => x === null || typeof x !== "object")) return `[${value.map(show).join(", ")}]`;
  return brief(JSON.stringify(value), 400);
}

const ACRONYMS = {
  lr: "LR", ema: "EMA", rng: "RNG", cudnn: "cuDNN", tf32: "TF32", fid: "FID",
  gpu: "GPU", cpu: "CPU", hndl: "HNDL", gan: "GAN", mse: "MSE", rgb: "RGB",
  id: "ID", std: "σ", mog: "MoG",
};
const SPECIAL_LABELS = {
  z_dim: "Latent dim",
  num_particles: "Particles",
  adversarial_weight: "Adversarial weight",
  coeff: "Gradient penalty coeff",
  matmul_allow_tf32: "Matmul TF32",
  cudnn_allow_tf32: "cuDNN TF32",
  cudnn_benchmark: "cuDNN benchmark",
  cudnn_deterministic: "cuDNN deterministic",
};
function humanizeKey(key) {
  const raw = String(key || "");
  if (SPECIAL_LABELS[raw]) return SPECIAL_LABELS[raw];
  if (!raw.includes("_") && /[A-Z]/.test(raw)) return raw;
  const words = raw.split("_").filter(Boolean);
  if (!words.length) return raw;
  return words.map((word, index) => {
    const lower = word.toLowerCase();
    if (ACRONYMS[lower]) return ACRONYMS[lower];
    return index === 0 ? lower[0].toUpperCase() + lower.slice(1) : lower;
  }).join(" ");
}

function formatPriorKind(kind) {
  if (!kind) return "Unknown";
  const lower = String(kind).toLowerCase();
  if (lower === "mog") return "MoG";
  if (lower === "gaussian") return "Gaussian";
  if (lower === "particles") return "Particles";
  return humanizeKey(kind);
}

function shapeDims(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object" && Array.isArray(value.shape)) return value.shape;
  return null;
}

function shape(dims) {
  const list = shapeDims(dims);
  return list ? list.map((d) => String(d)).join(" × ") : show(dims);
}

function compactShape(dims) {
  const list = shapeDims(dims);
  return list ? `[${list.map((d) => String(d)).join(",")}]` : show(dims);
}

function ports(value, formatter = shape) {
  const direct = shapeDims(value);
  if (direct) return formatter(direct);
  if (!value || typeof value !== "object" || Array.isArray(value)) return formatter(value);
  const entries = Object.entries(value);
  if (!entries.length) return "—";
  if (entries.length === 1) return formatter(entries[0][1]);
  return entries.map(([name, dims]) => `${name}: ${formatter(dims)}`).join("; ");
}

function networkShapes(entry) {
  const first = (entry.graph?.subgraphs || [])[0];
  const input = entry.input_shape !== undefined && entry.input_shape !== null ? entry.input_shape : first?.input_shape;
  const output = entry.output_shape !== undefined && entry.output_shape !== null ? entry.output_shape : first?.output_shape;
  return { input, output };
}

function shortFactory(factory) {
  if (!factory) return "—";
  const text = String(factory);
  const colon = text.lastIndexOf(":");
  if (colon !== -1 && colon < text.length - 1) return text.slice(colon + 1);
  return text;
}

function shortFile(file) {
  if (!file) return "";
  const text = String(file);
  const slash = text.lastIndexOf("/");
  return slash !== -1 ? text.slice(slash + 1) : text;
}

function isNonEmpty(value) {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function definitions(entries, className = "model-facts") {
  const list = node("dl", undefined, className);
  for (const [label, value] of entries) {
    if (value === undefined) continue;
    const rowEl = node("div");
    rowEl.append(node("dt", label), node("dd", show(value)));
    list.append(rowEl);
  }
  return list;
}

function unavailable(reason, prefix = "Unavailable") {
  return node("p", `${prefix}: ${brief(reason || "no reason recorded", 600)}`, "model-unavailable");
}

function section(title, description) {
  const result = node("section", undefined, "model-section");
  const heading = node("h2", title);
  heading.id = `model-${title.toLowerCase().replace(/[^a-z]+/g, "-").replace(/^-|-$/g, "")}`;
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
    details.append(definitions(Object.entries(source.parameters).map(([k, v]) => [humanizeKey(k), v]), "model-facts model-parameters"));
  return details;
}

function isFrozenNode(item) {
  return item.trainable === false || (finite(item.params) && item.params > 0 && item.trainable_params === 0);
}

const KEY_ARCHITECTURAL_OPS = new Set([
  "attention", "cross_attention", "self_attention", "multi_head_attention",
  "transformer_block", "resblock", "residual_block",
  "conv", "conv2d", "conv1d", "conv_transpose", "conv_transpose2d",
  "linear", "dense", "embedding", "pos_embed", "feed_forward", "mlp",
  "pixel_shuffle", "upsample", "downsample", "avg_pool", "max_pool", "adaptive_avg_pool",
]);
const GLUE_CATEGORIES = new Set(["arithmetic", "shape", "activation", "spatial"]);

function isKeyLayer(item) {
  if (finite(item.params) && item.params > 0) return true;
  if (item.category === "pretrained") return true;
  const op = String(item.op || "").toLowerCase();
  return KEY_ARCHITECTURAL_OPS.has(op);
}

function summarizeOps(subgraphs) {
  const counts = new Map();
  for (const sg of subgraphs || []) {
    for (const item of sg.nodes || []) {
      const hasParams = finite(item.params) && item.params > 0;
      const isPretrained = item.category === "pretrained";
      const op = String(item.op || "");
      const opLower = op.toLowerCase();
      if (!hasParams && !isPretrained) {
        if (GLUE_CATEGORIES.has(item.category) || !KEY_ARCHITECTURAL_OPS.has(opLower)) continue;
      }
      const existing = counts.get(op) || {
        op,
        category: item.category || "other",
        count: 0,
        params: 0,
        frozenCount: 0,
        pretrained: false,
      };
      existing.count += 1;
      existing.params += finite(item.params) ? item.params : 0;
      if (isFrozenNode(item)) existing.frozenCount += 1;
      if (isPretrained) existing.pretrained = true;
      counts.set(op, existing);
    }
  }
  return [...counts.values()].sort((a, b) => {
    if (a.pretrained !== b.pretrained) return a.pretrained ? -1 : 1;
    if (b.params !== a.params) return b.params - a.params;
    return b.count - a.count || a.op.localeCompare(b.op);
  });
}

const ARG_SHORT_NAMES = {
  out_channels: "ch",
  out_features: "out",
  in_channels: "in",
  in_features: "in",
  kernel_size: "k",
  stride: "s",
  padding: "pad",
  dilation: "dil",
  num_groups: "groups",
  negative_slope: "slope",
  relative_position_bias: "rel_pos",
  spatial_shape: "spatial",
  scale_factor: "scale",
  out_bias: "out_bias",
  qkv_bias: "qkv_bias",
};
const FALLBACK_ARG_KEYS = [
  "model", "out_channels", "out_features", "kernel_size", "stride", "heads",
  "hidden", "num_groups", "shape", "scale_factor", "negative_slope", "max_len",
];

function formatArgValue(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return number(value);
  if (typeof value === "string") return brief(value, 28);
  if (Array.isArray(value)) {
    if (value.length === 2 && value[0] === value[1] && typeof value[0] === "number") {
      return String(value[0]);
    }
    if (value.every((v) => v === null || typeof v !== "object")) {
      return `[${value.map((v) => typeof v === "number" ? number(v) : String(v)).join(",")}]`;
    }
  }
  if (typeof value === "object" && typeof value.truncated === "string") return brief(value.truncated, 28);
  return brief(JSON.stringify(value), 28);
}

function formatKeyArgs(item) {
  const args = item.args && typeof item.args === "object" ? item.args : {};
  const explicit = Array.isArray(item.explicit) && item.explicit.length
    ? item.explicit.filter((k) => k in args)
    : FALLBACK_ARG_KEYS.filter((k) => k in args);
  if (!explicit.length) return { text: "—", full: "" };
  const allParts = explicit.map((k) => `${ARG_SHORT_NAMES[k] || k}=${formatArgValue(args[k])}`);
  const full = explicit.map((k) => `${k}=${show(args[k])}`).join(", ");
  const shown = allParts.slice(0, 4);
  let text = shown.join(", ");
  if (allParts.length > 4) text += ` +${allParts.length - 4}`;
  if (text.length > 76) text = `${text.slice(0, 75)}…`;
  return { text, full };
}

function roleShortTag(role) {
  if (role === "generator") return "G";
  if (role === "critic") return "D";
  if (role === "encoder") return "Enc";
  if (role === "auxiliary") return "Aux";
  if (role === "alias") return "Alias";
  return role || "Net";
}

function makeFlowArrow() {
  const wrap = node("div", undefined, "model-flow-arrow");
  wrap.setAttribute("aria-hidden", "true");
  const svg = svgNode("svg", { viewBox: "0 0 24 24", width: 20, height: 20, fill: "none" });
  svg.append(
    svgNode("path", { d: "M4 12h15", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round" }),
    svgNode("path", { d: "M14 7l5 5-5 5", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" }),
  );
  wrap.append(svg);
  return wrap;
}

function formatParamInline(p) {
  if (!p || !finite(p.total)) return "";
  if (finite(p.frozen) && p.frozen > 0) {
    return `${params(p.total)} (${params(p.trainable)} train / ${params(p.frozen)} frozen)`;
  }
  return params(p.total);
}

function buildParamBar(p) {
  const wrap = node("div", undefined, "model-param-meter");
  const total = finite(p.total) ? p.total : 0;
  const trainable = finite(p.trainable) ? p.trainable : 0;
  const frozen = finite(p.frozen) ? p.frozen : Math.max(0, total - trainable);
  const headline = node("div", undefined, "model-param-head");
  headline.append(node("strong", `${params(total)} params`));
  if (frozen > 0 && trainable > 0) {
    headline.append(node("span", `${params(trainable)} trainable · ${params(frozen)} frozen`, "model-param-split"));
  } else if (frozen > 0 && trainable === 0) {
    headline.append(node("span", "100% frozen", "model-param-split model-param-frozen-label"));
  } else if (total > 0) {
    headline.append(node("span", "100% trainable", "model-param-split"));
  }
  const track = node("div", undefined, "model-param-bar");
  const trainPct = total > 0 ? Math.max(0, Math.min(100, (trainable / total) * 100)) : 0;
  const frozenPct = total > 0 ? Math.max(0, Math.min(100 - trainPct, (frozen / total) * 100)) : 0;
  if (trainPct > 0) {
    const seg = node("span", undefined, "model-param-seg-train");
    seg.style.width = `${trainPct.toFixed(1)}%`;
    track.append(seg);
  }
  if (frozenPct > 0) {
    const seg = node("span", undefined, "model-param-seg-frozen");
    seg.style.width = `${frozenPct.toFixed(1)}%`;
    track.append(seg);
  }
  wrap.append(headline, track);
  return wrap;
}

function leadingSummaryComments(source) {
  const sections = source?.sections || [];
  if (!sections.length) return [];
  const chosen = [];
  for (const item of sections) {
    if (chosen.length >= 2) break;
    if (chosen.length === 0 && (!finite(item.line) || item.line <= 20)) {
      chosen.push(item.title);
    } else if (chosen.length === 1 && finite(item.line) && item.line <= 6) {
      chosen.push(item.title);
    }
  }
  return chosen;
}

export function modelPanel(api, base, { catalog, onPlot }) {
  let loaded = null;
  let rendered = null;
  let pending = null;
  const root = () => document.getElementById("model-content");
  const overview = () => document.getElementById("model-overview");
  const status = (text) => {
    const element = document.getElementById("model-status");
    element.textContent = text || "";
    element.hidden = !text;
  };
  function clearOverview() {
    const el = overview();
    if (!el) return;
    el.replaceChildren();
    el.hidden = true;
  }
  function renderOverview(model) {
    const el = overview();
    if (!el) return;
    const strip = node("div", undefined, "model-overview-strip");
    // 1. Prior chip
    const prior = model.prior || {};
    const priorParts = [formatPriorKind(prior.kind)];
    if (finite(prior.z_dim)) priorParts.push(`z=${prior.z_dim}`);
    if (finite(prior.num_particles)) priorParts.push(`${compactCount(prior.num_particles)} particles`);
    if (prior.learnable === false && String(prior.kind || "").toLowerCase() !== "gaussian") priorParts.push("frozen");
    const priorBlock = node("div", undefined, "model-overview-item");
    priorBlock.append(node("span", "Prior", "model-overview-label"), node("strong", priorParts.join(" · "), "model-overview-value"));
    strip.append(priorBlock);

    // 2. Primary networks (skip pure aliases; cap visible chips for compactness)
    const allNetworks = model.networks || [];
    const primary = allNetworks.filter((entry) => entry.role !== "alias");
    const aliases = allNetworks.filter((entry) => entry.role === "alias");
    const shownNets = (primary.length ? primary : allNetworks).slice(0, 4);
    for (const entry of shownNets) {
      const item = node("div", undefined, "model-overview-item model-overview-net");
      const labelRow = node("span", undefined, "model-overview-label");
      labelRow.append(
        node("span", roleShortTag(entry.role), `model-badge model-role-${entry.role} model-overview-badge`),
        document.createTextNode(` ${entry.display_name || entry.name}`),
      );
      const { input, output } = networkShapes(entry);
      const shapeText = input !== undefined && output !== undefined
        ? `${ports(input, compactShape)} → ${ports(output, compactShape)}`
        : "";
      const paramText = formatParamInline(entry.graph?.parameters);
      const detailParts = [];
      if (paramText) detailParts.push(paramText);
      if (shapeText) detailParts.push(shapeText);
      if (!detailParts.length && entry.source?.file) detailParts.push(shortFile(entry.source.file));
      if (!detailParts.length) detailParts.push(shortFactory(entry.factory));
      item.append(labelRow, node("strong", detailParts.join(" · "), "model-overview-value"));
      strip.append(item);
    }
    const extraCount = Math.max(0, primary.length - shownNets.length) + aliases.length;
    if (extraCount > 0) {
      const more = node("div", undefined, "model-overview-item model-overview-extra");
      const parts = [];
      if (primary.length > shownNets.length) parts.push(`+${primary.length - shownNets.length} nets`);
      if (aliases.length) parts.push(`+${aliases.length} ${aliases.length === 1 ? "alias" : "aliases"}`);
      more.append(node("span", "More", "model-overview-label"), node("strong", parts.join(" · "), "model-overview-value"));
      strip.append(more);
    }

    // 3. Optimizers / LR summary
    const o = model.optimizers || {};
    const optParts = [];
    if (finite(o.generator?.lr)) optParts.push(`LR ${rate(o.generator.lr)}`);
    if (finite(o.critic?.lr_mult)) optParts.push(`D ×${rate(o.critic.lr_mult)}`);
    if (o.prior && o.prior.status !== "unavailable" && finite(o.prior.lr_mult)) optParts.push(`Prior ×${rate(o.prior.lr_mult)}`);
    if (finite(o.ema) && o.ema > 0) optParts.push(`EMA ${rate(o.ema)}`);
    if (optParts.length) {
      const optBlock = node("div", undefined, "model-overview-item");
      optBlock.append(node("span", "Optimizers", "model-overview-label"), node("strong", optParts.join(" · "), "model-overview-value"));
      strip.append(optBlock);
    }

    // 4. Jump link to #model-section
    const jump = node("a", "Full model ↓", "model-overview-jump");
    jump.href = "#model-section";
    jump.addEventListener("click", (event) => {
      event.preventDefault();
      document.getElementById("model-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    el.replaceChildren(strip, jump);
    el.hidden = false;
  }

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
      button.title = "Plot this series in Learning curves";
      button.addEventListener("click", () => onPlot(metric));
      cell.append(button);
    } else cell.append(node("code", metric));
    if (note) cell.append(node("small", note));
    return cell;
  }

  function renderArchitecture(model) {
    const result = section(
      "Architecture & Dataflow",
      "How prior latents and data flow through generators, auxiliary modules, critics, and loss objectives.",
    );
    const networks = model.networks || [];
    const byName = new Map(networks.map((n) => [n.name, n]));
    const edges = model.edges || [];
    const losses = model.losses || {};

    const labelOf = (ref) => {
      if (!ref) return "—";
      if (ref === "prior") return "prior";
      if (ref === "data") return "data";
      if (ref === "candidate") return "real / fake";
      if (ref.startsWith("components.")) {
        const key = ref.slice("components.".length);
        const target = byName.get(key);
        return target ? (target.display_name || target.name) : key;
      }
      return ref;
    };

    const incomingFor = (targetKey) => edges.filter((e) => e.to === targetKey);

    const flow = node("div", undefined, "model-flow");

    // Stage 1: Sources (Prior + Data)
    const stage1 = node("div", undefined, "model-flow-stage");
    stage1.append(node("h3", "1 · Inputs", "model-flow-stage-title"));
    const stage1List = node("div", undefined, "model-flow-nodes");

    const prior = model.prior || {};
    const priorCard = node("article", undefined, "model-flow-node model-flow-source");
    const priorHead = node("div", undefined, "model-flow-node-head");
    priorHead.append(node("strong", "Prior"), node("span", formatPriorKind(prior.kind), "model-badge model-role-encoder"));
    const priorMeta = [];
    if (finite(prior.z_dim)) priorMeta.push(`z=${prior.z_dim}`);
    if (finite(prior.num_particles)) priorMeta.push(`${compactCount(prior.num_particles)} particles`);
    priorMeta.push(prior.learnable === false ? "frozen" : "learnable");
    priorCard.append(priorHead, node("p", priorMeta.join(" · "), "model-flow-meta"));
    stage1List.append(priorCard);

    const dataCard = node("article", undefined, "model-flow-node model-flow-source");
    const dataHead = node("div", undefined, "model-flow-node-head");
    dataHead.append(node("strong", "Data"), node("span", "batch", "model-badge model-role-auxiliary"));
    const dataArgs = Object.entries(model.data?.args || {})
      .filter(([, v]) => isNonEmpty(v) && typeof v !== "object")
      .slice(0, 3)
      .map(([k, v]) => `${k}=${show(v)}`);
    dataCard.append(dataHead, node("p", shortFactory(model.data?.factory), "model-flow-shape"));
    if (dataArgs.length) dataCard.append(node("p", dataArgs.join(" · "), "model-flow-meta"));
    stage1List.append(dataCard);
    stage1.append(stage1List);
    flow.append(stage1);

    // Stage 2: Synthesis & Encoding (non-critic networks)
    const synthesisOrder = { auxiliary: 0, encoder: 1, generator: 2, alias: 3 };
    const synthesisNets = networks
      .filter((n) => n.role !== "critic")
      .sort((a, b) => (synthesisOrder[a.role] ?? 4) - (synthesisOrder[b.role] ?? 4));

    const makeNetworkFlowCard = (entry) => {
      const card = node("article", undefined, `model-flow-node model-flow-role-${entry.role}`);
      const head = node("div", undefined, "model-flow-node-head");
      const nameLink = node("a", entry.display_name || entry.name, "model-flow-link");
      nameLink.href = `#model-net-${entry.name}`;
      nameLink.addEventListener("click", (event) => {
        event.preventDefault();
        document.getElementById(`model-net-${entry.name}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      head.append(nameLink, node("span", entry.role, `model-badge model-role-${entry.role}`));
      card.append(head);

      const { input, output } = networkShapes(entry);
      if (input !== undefined && output !== undefined) {
        card.append(node("p", `${ports(input, compactShape)} → ${ports(output, compactShape)}`, "model-flow-shape"));
      } else if (entry.reuse_of) {
        card.append(node("p", `shares weights of ${byName.get(entry.reuse_of)?.display_name || entry.reuse_of}`, "model-flow-meta"));
      }

      const badges = node("div", undefined, "model-flow-tags");
      const gp = entry.graph?.parameters;
      if (gp && finite(gp.total)) {
        badges.append(node("span", `${params(gp.total)} params`, "model-flow-tag"));
        if (finite(gp.frozen) && gp.frozen > 0) {
          badges.append(node("span", gp.trainable === 0 ? "frozen" : `${params(gp.frozen)} frozen`, "model-flow-tag model-flow-tag-warn"));
        }
      } else if (entry.trainable === false || entry.freeze_parameters) {
        badges.append(node("span", "frozen", "model-flow-tag model-flow-tag-warn"));
      }
      if (entry.reuse_of) {
        badges.append(node("span", "reuse", "model-flow-tag model-flow-tag-reuse"));
      }
      if (badges.childElementCount) card.append(badges);

      // Incoming ports
      const portList = node("div", undefined, "model-flow-ports");
      const inEdges = incomingFor(`components.${entry.name}`);
      const advTerms = (losses.discriminator || []).filter((t) => t.kind === "adversarial_d" && t.critic === entry.name);
      for (const edge of inEdges) {
        if (edge.path === "candidate" && advTerms.length) {
          for (const term of advTerms) {
            const label = `${edge.port} ← ${term.real || "batch.real"} vs ${term.fake || "generated"}`;
            portList.append(node("span", label, "model-flow-port"));
          }
          continue;
        }
        const isReuse = edge.path === "reuse";
        const text = isReuse
          ? `weights ← ${labelOf(edge.from)} (reuse)`
          : `${edge.port} ← ${edge.path === edge.port ? labelOf(edge.from) : edge.path}`;
        const pill = node("span", text, `model-flow-port${isReuse ? " model-flow-port-reuse" : ""}`);
        portList.append(pill);
      }
      if (portList.childElementCount) card.append(portList);
      return card;
    };

    if (synthesisNets.length) {
      flow.append(makeFlowArrow());
      const stage2 = node("div", undefined, "model-flow-stage");
      stage2.append(node("h3", "2 · Synthesis & Encoding", "model-flow-stage-title"));
      const stage2List = node("div", undefined, "model-flow-nodes");
      for (const entry of synthesisNets) stage2List.append(makeNetworkFlowCard(entry));
      stage2.append(stage2List);
      flow.append(stage2);
    }

    // Stage 3: Critics & Objectives
    const criticNets = networks.filter((n) => n.role === "critic");
    const extraObjectives = (losses.generator || []).filter(
      (t) => t.kind !== "adversarial_g" && t.kind !== "prior_regularizer",
    );
    const priorReg = (losses.generator || []).find((t) => t.kind === "prior_regularizer");

    if (criticNets.length || extraObjectives.length || priorReg) {
      flow.append(makeFlowArrow());
      const stage3 = node("div", undefined, "model-flow-stage");
      stage3.append(node("h3", "3 · Critics & Objectives", "model-flow-stage-title"));
      const stage3List = node("div", undefined, "model-flow-nodes");
      for (const entry of criticNets) stage3List.append(makeNetworkFlowCard(entry));

      for (const obj of extraObjectives) {
        const card = node("article", undefined, "model-flow-node model-flow-objective");
        const head = node("div", undefined, "model-flow-node-head");
        head.append(
          node("strong", obj.id),
          node("span", `${shortFactory(obj.kind) || "objective"}${finite(obj.weight) ? ` ×${rate(obj.weight)}` : ""}`, "model-badge model-role-auxiliary"),
        );
        card.append(head);
        const objEdges = incomingFor(`objective.${obj.id}`);
        const portList = node("div", undefined, "model-flow-ports");
        for (const edge of objEdges) {
          const text = `${edge.port} ← ${edge.path}${edge.detached ? " · detached" : ""}`;
          portList.append(node("span", text, `model-flow-port${edge.detached ? " model-flow-port-detached" : ""}`));
        }
        if (portList.childElementCount) card.append(portList);
        stage3List.append(card);
      }

      if (priorReg) {
        const card = node("article", undefined, `model-flow-node model-flow-objective${priorReg.active === false ? " model-flow-inactive" : ""}`);
        const head = node("div", undefined, "model-flow-node-head");
        head.append(
          node("strong", "prior_regularizer"),
          node("span", priorReg.active === false ? "inactive" : `w=${rate(priorReg.weight)}`, "model-badge model-role-auxiliary"),
        );
        card.append(head, node("p", "Prior particle spread regularizer", "model-flow-meta"));
        stage3List.append(card);
      }

      stage3.append(stage3List);
      flow.append(stage3);
    }

    result.append(flow);
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
    const keyNodes = nodes.filter(isKeyLayer);
    const canFilter = nodes.length > 16 && keyNodes.length > 0 && keyNodes.length < nodes.length;
    let showAll = !canFilter;

    const { scroll, body } = table(["Line", "Node", "Op", "Key args", "Output shape", "Parameters", "Trainable"], `${entry.name} layers`);

    const renderRows = () => {
      body.replaceChildren();
      const visible = showAll ? nodes : keyNodes;
      const allSections = entry.source?.sections && (entry.graph?.subgraphs || []).length === 1 ? [...entry.source.sections] : [];
      let secIdx = 0;
      for (const item of visible) {
        if (allSections.length && finite(item.line)) {
          let pendingHeading = null;
          while (secIdx < allSections.length && allSections[secIdx].line <= item.line) {
            pendingHeading = allSections[secIdx];
            secIdx += 1;
          }
          if (pendingHeading) {
            const heading = row(body, [pendingHeading.title], "model-group");
            heading.firstChild.colSpan = 7;
          }
        }
        const name = node("span", item.id);
        if (item.explicit?.length) name.title = `explicit: ${item.explicit.join(", ")}`;
        const op = node("span", item.op, `model-op model-op-${String(item.category || "other").replace(/[^a-z]/g, "")}`);
        if (item.category) op.title = item.category;
        const { text: argText, full: argFull } = formatKeyArgs(item);
        const argEl = node("span", argText, "model-args");
        if (argFull) argEl.title = argFull;
        row(body, [
          finite(item.line) ? String(item.line) : "—",
          name,
          op,
          argEl,
          shapes ? ports(item.out) : "—",
          finite(item.params) ? params(item.params) : "—",
          isFrozenNode(item) ? "frozen" : finite(item.params) && item.params > 0 ? "yes" : "—",
        ]);
      }
      if (subgraph.nodes_truncated) {
        const trunc = row(body, [`Showing the first ${nodes.length} nodes`]);
        trunc.firstChild.colSpan = 7;
      }
    };

    if (canFilter) {
      const bar = node("div", undefined, "model-layers-bar");
      const info = node("span", undefined, "model-layers-count");
      const toggle = node("button", undefined, "secondary small-button model-layers-toggle");
      toggle.type = "button";
      const syncBar = () => {
        info.textContent = showAll
          ? `Showing all ${exact.format(nodes.length)} nodes`
          : `Showing ${exact.format(keyNodes.length)} key layers of ${exact.format(nodes.length)} nodes`;
        toggle.textContent = showAll
          ? `Key layers only (${exact.format(keyNodes.length)})`
          : `Show all ${exact.format(nodes.length)} nodes`;
        toggle.setAttribute("aria-pressed", String(showAll));
      };
      toggle.addEventListener("click", () => {
        showAll = !showAll;
        syncBar();
        renderRows();
      });
      syncBar();
      bar.append(info, toggle);
      details.append(bar);
    }

    renderRows();
    details.append(scroll);
    return details;
  }

  function renderNetwork(entry) {
    const card = node("article", undefined, "model-card model-network");
    card.id = `model-net-${entry.name}`;
    const displayName = entry.display_name || entry.name;
    const title = node("h3");
    title.append(document.createTextNode(displayName));
    if (entry.display_name && entry.display_name !== entry.name) {
      title.append(node("span", `(${entry.name})`, "model-engine-name"));
    }
    title.append(node("span", entry.role, `model-badge model-role-${entry.role}`));
    if (entry.reuse_of) {
      title.append(node("span", `reuses ${entry.reuse_of}`, "model-badge model-role-alias"));
    }
    if (entry.trainable === false || entry.freeze_parameters) {
      title.append(node("span", "frozen", "model-badge model-badge-warn"));
    }
    card.append(title);

    const graph = entry.graph || {};
    const { input, output } = networkShapes(entry);

    // Prominent header metrics bar: I/O shape, Parameter bar, Source / optimizer
    const metricsBar = node("div", undefined, "model-net-metrics");
    if (input !== undefined && output !== undefined) {
      const ioBlock = node("div", undefined, "model-net-metric");
      ioBlock.append(
        node("span", "Input → Output", "model-net-metric-label"),
        node("code", `${ports(input)} → ${ports(output)}`, "model-net-io"),
      );
      metricsBar.append(ioBlock);
    }
    if (graph.parameters) {
      const paramBlock = node("div", undefined, "model-net-metric model-net-metric-params");
      paramBlock.append(buildParamBar(graph.parameters));
      metricsBar.append(paramBlock);
    }
    const sourceBlockMetric = node("div", undefined, "model-net-metric");
    sourceBlockMetric.append(
      node("span", "Source · Optimizer", "model-net-metric-label"),
      node(
        "span",
        [
          entry.source?.file || (entry.reuse_of ? `reuse(${entry.reuse_of})` : shortFactory(entry.factory)),
          entry.optimizer_group ? `${entry.optimizer_group} group` : null,
        ].filter(Boolean).join(" · "),
        "model-net-source-label",
      ),
    );
    metricsBar.append(sourceBlockMetric);
    card.append(metricsBar);

    // Architecture summary right on the card (before expanding Layers / HNDL source)
    const comments = leadingSummaryComments(entry.source);
    const opSummary = summarizeOps(graph.subgraphs);
    const hasProviders = entry.pretrained_providers && Object.keys(entry.pretrained_providers).length > 0;
    const hasSourceParams = entry.source?.parameters && Object.keys(entry.source.parameters).length > 0;
    if (comments.length || opSummary.length || hasProviders || hasSourceParams) {
      const summaryBox = node("div", undefined, "model-net-summary");
      if (comments.length) {
        summaryBox.append(node("p", comments.join(" "), "model-net-comment"));
      }
      if (opSummary.length) {
        const chips = node("div", undefined, "model-op-chips");
        const shown = opSummary.slice(0, 8);
        for (const item of shown) {
          const frozenSuffix = item.frozenCount === item.count && item.count > 0 ? " (frozen)" : "";
          const chip = node(
            "span",
            `${item.count}× ${item.op}${frozenSuffix}`,
            `model-op-chip model-op-${String(item.category || "other").replace(/[^a-z]/g, "")}`,
          );
          if (item.params > 0) chip.title = `${params(item.params)} parameters (${item.category})`;
          chips.append(chip);
        }
        if (opSummary.length > shown.length) {
          chips.append(node("span", `+${opSummary.length - shown.length} more ops`, "model-op-chip model-op-chip-more"));
        }
        summaryBox.append(chips);
      }
      if (hasProviders) {
        const provText = Array.isArray(entry.pretrained_providers)
          ? entry.pretrained_providers.map(show).join(", ")
          : Object.entries(entry.pretrained_providers).map(([k, v]) => `${k}: ${show(v)}`).join(" · ");
        summaryBox.append(node("p", `Pretrained providers: ${provText}`, "model-quiet"));
      }
      if (hasSourceParams) {
        const spText = Object.entries(entry.source.parameters)
          .slice(0, 4)
          .map(([k, v]) => `${humanizeKey(k)}: ${show(v)}`)
          .join(" · ");
        summaryBox.append(node("p", `Parameters: ${spText}`, "model-quiet"));
      }
      card.append(summaryBox);
    }

    const facts = [["Factory", entry.factory], ["Optimizer group", entry.optimizer_group]];
    if (entry.reuse_of) facts.push(["Shares weights with", entry.reuse_of]);
    if (entry.source?.file) facts.push(["Source file", entry.source.file]);
    else if (entry.source?.text) facts.push(["Source", "inline HNDL (file name not recorded)"]);
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
      details.append(
        node("summary", "Constructor arguments"),
        definitions(Object.entries(entry.args).map(([k, v]) => [humanizeKey(k), v])),
      );
      card.append(details);
    }
    return card;
  }

  function renderNetworks(model) {
    const result = section("Networks", "One card per component: role, I/O contract, parameters, op breakdown, layers and source.");
    const recorded = model.networks_recorded;
    if (recorded) {
      result.append(node("p", `Layer detail ${recorded.origin === "backfill" ? "backfilled with hypergan model" : "recorded at training start"}${recorded.hndl ? ` · hndl ${recorded.hndl}` : ""}.`, "model-quiet"));
    }
    const list = node("div", undefined, "model-networks");
    for (const entry of model.networks || []) list.append(renderNetwork(entry));
    result.append(list);
    return result;
  }

  function renderFormulation(model) {
    const f = model.formulation || {};
    const result = section("GAN formulation & Prior", "The ParticleGAN recipe and latent prior consumed by this run.");
    const card = node("article", undefined, "model-card");
    const title = node("h3", f.name || "Unknown formulation");
    title.append(node("span", f.snapshot_recorded ? "recorded recipe" : "dependency defaults", "model-badge"));
    card.append(title, node("p", `ParticleGAN ${f.particlegan || "version not recorded"}`, "model-quiet"));
    if (f.note) card.append(node("p", f.note, "model-note"));
    card.append(definitions(
      Object.entries(f.parameters || {})
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => [humanizeKey(k), v]),
    ));
    const prior = model.prior || {};
    const priorCard = node("article", undefined, "model-card");
    priorCard.append(node("h3", `Prior · ${formatPriorKind(prior.kind)}`));
    const sigma = prior.sigma;
    const initParts = [];
    if (prior.initialization?.device) initParts.push(show(prior.initialization.device));
    if (prior.initialization?.seed !== null && prior.initialization?.seed !== undefined) initParts.push(`seed ${show(prior.initialization.seed)}`);
    priorCard.append(definitions([
      ["Latent dimension", prior.z_dim],
      ["Particles", prior.num_particles ?? undefined],
      ["Learnable", prior.learnable],
      ["Initialization", initParts.length ? initParts.join(" · ") : undefined],
      ["σ", sigma ? (sigma.mode === "calibrated" ? `calibrated from σ_rel ${show(sigma.sigma_rel)}` : `${show(sigma.value)} (${sigma.mode})`) : undefined],
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
        for (const key of ["rows"])
          if (term[key] !== undefined && term[key] !== null) details.push(`${key} ${show(term[key])}`);
        if (term.detach?.length) details.push(`detached: ${term.detach.join(", ")}`);
        if (term.active === false) details.push(term.inactive_reason ? `inactive: ${term.inactive_reason}` : "inactive (weight 0)");
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
      item.append(node("small", humanizeKey(label)), seriesCell(metric));
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
    row(body, ["Generator", `${show(g.type)} · ${show(g.implementation)}`, rate(g.lr), show(g.betas), (g.components || []).join(", ") || "—"]);
    if (p.status === "unavailable") row(body, ["Prior", "no prior group", "—", "—", brief(p.reason || "no trainable table")], "model-inactive");
    else row(body, ["Prior", "shares generator step", `${rate(p.lr)} (×${rate(p.lr_mult)})`, show(p.betas), "particle table"]);
    row(body, ["Critic", `${show(c.type)} · ${show(c.implementation)}`, `${rate(c.lr)} (×${rate(c.lr_mult)})`, show(c.betas), (c.components || []).join(", ") || "—"]);
    result.append(scroll);
    const grid = node("div", undefined, "model-grid");
    const schedule = node("article", undefined, "model-card");
    const scheduleEntries = [
      ["LR anneal start", o.schedule?.lr_anneal_start],
      ["LR floor", o.schedule?.lr_floor],
      ["Network LR floor", o.schedule?.network_lr_floor],
      ["Network LR horizon cap", o.schedule?.network_lr_horizon_cap],
      ["EMA (G and prior)", o.ema],
    ].filter(([, v]) => isNonEmpty(v));
    schedule.append(node("h3", "Schedule"), definitions(scheduleEntries.length ? scheduleEntries : [["Schedule", "default"]]));
    if (o.schedule?.metric) schedule.append(seriesCell(o.schedule.metric));
    const noise = node("article", undefined, "model-card");
    const noiseEntries = [
      ["Critic input σ", o.noise?.critic_input_std],
      ["Critic input anneal end", o.noise?.critic_input_anneal_end],
      ["Generator output σ", o.noise?.generator_output_std],
      ["Generator output warmup", o.noise?.generator_output_warmup],
    ].filter(([, v]) => isNonEmpty(v));
    noise.append(node("h3", "Noise"), definitions(noiseEntries.length ? noiseEntries : [["Noise", "none"]]));
    grid.append(schedule, noise);
    result.append(grid);
    return result;
  }

  function renderSettings(model) {
    const result = section("Data & training", "Dataset configuration, training loop parameters, and sampling defaults.");
    const grid = node("div", undefined, "model-grid");

    const data = node("article", undefined, "model-card");
    const dataEntries = [
      ["Factory", model.data?.factory],
      ...Object.entries(model.data?.args || {})
        .filter(([, v]) => isNonEmpty(v))
        .map(([k, v]) => [humanizeKey(k), v]),
    ];
    data.append(node("h3", "Data"), definitions(dataEntries));

    const training = node("article", undefined, "model-card");
    const run = model.run || {};
    const skipTraining = new Set([
      "lr_anneal_start", "lr_floor", "network_lr_floor", "network_lr_horizon_cap", "ema",
      "input_noise_std", "input_noise_anneal_end", "output_noise_std", "output_noise_warmup",
    ]);
    const advancedKeys = new Set([
      "phase_draws", "data_rng_device", "data_seed_offset", "prior_seed_offset", "backend",
    ]);
    const primaryTraining = [];
    const advancedTraining = [];
    for (const [key, val] of Object.entries(model.training || {})) {
      if (skipTraining.has(key) || !isNonEmpty(val)) continue;
      if (advancedKeys.has(key)) {
        if (key === "backend" && typeof val === "object" && !Array.isArray(val)) {
          for (const [bk, bv] of Object.entries(val)) {
            if (isNonEmpty(bv)) advancedTraining.push([humanizeKey(bk), bv]);
          }
        } else {
          advancedTraining.push([humanizeKey(key), val]);
        }
      } else {
        primaryTraining.push([humanizeKey(key), val]);
      }
    }
    if (isNonEmpty(run.global_batch_size)) primaryTraining.push(["Global batch size", run.global_batch_size]);
    training.append(node("h3", "Training"), definitions(primaryTraining));
    if (advancedTraining.length) {
      const adv = node("details", undefined, "model-source");
      adv.append(node("summary", "Advanced runtime & RNG"), definitions(advancedTraining));
      training.append(adv);
    }

    const sampling = node("article", undefined, "model-card");
    const samplingEntries = [
      ...Object.entries(model.sampling || {})
        .filter(([, v]) => isNonEmpty(v))
        .map(([k, v]) => [humanizeKey(k), v]),
      ["Metric preset", model.metrics?.preset ?? undefined],
      ["Custom metrics", (model.metrics?.custom || []).length ? model.metrics.custom.join(", ") : "none"],
    ];
    sampling.append(node("h3", "Sampling & metrics"), definitions(samplingEntries));

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
    renderOverview(model);
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
    const sections = [
      renderArchitecture(model),
      renderNetworks(model),
      renderFormulation(model),
      renderLosses(model),
      renderOptimizers(model),
      renderSettings(model),
    ];
    for (const item of sections) {
      const heading = item.querySelector("h2");
      const link = node("a", heading.textContent);
      link.href = `#${heading.id}`;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        heading.scrollIntoView({ behavior: "smooth", block: "start" });
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
        clearOverview();
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
