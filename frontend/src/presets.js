// Metric presets, FoldliteDB pinned-loss view selection, URL view-state
// persistence, and run header/health formatting helpers for the Show page.

export const MAX_CHART_SERIES = 8;
export const PINNED_HEADLINE_METRICS = ["loss/g_total", "loss/d_total"];
export const VALID_SCALES = ["linear", "symlog"];
export const VALID_SMOOTHING = ["0", "0.2", "0.5"];
export const PRESET_NAMES = ["defaults", "losses", "health", "diversity"];

export function isScalarTrainingMetric(definition) {
  if (!definition || typeof definition !== "object") return false;
  if (definition.scope === "snapshot") return false;
  return definition.kind === undefined || definition.kind === "scalar";
}

export function defaultMetrics(catalogMetrics = {}) {
  const metrics = catalogMetrics || {};
  const has = (id) => isScalarTrainingMetric(metrics[id]);
  const criticSlot = has("critic/accuracy") ? "critic/accuracy" : "critic/margin";
  const preferred = [
    "loss/g_total",
    "loss/d_total",
    "loss/gradient_penalty",
    "loss/prior_regularizer",
    criticSlot,
    "diversity/generated_rms",
    "diversity/ratio",
    "diversity/pooled4_ratio",
  ];
  return preferred.filter(has).slice(0, MAX_CHART_SERIES);
}

export function presetMetrics(presetName, catalogMetrics = {}) {
  const metrics = catalogMetrics || {};
  const allIds = Object.keys(metrics).filter((id) =>
    isScalarTrainingMetric(metrics[id]),
  );
  const has = (id) => isScalarTrainingMetric(metrics[id]);
  const pick = (candidates) =>
    [...new Set(candidates.filter(has))].slice(0, MAX_CHART_SERIES);

  switch (presetName) {
    case "defaults":
      return defaultMetrics(metrics);
    case "losses": {
      const priority = [
        "loss/g_total",
        "loss/d_total",
        "loss/gradient_penalty",
        "loss/prior_regularizer",
        "loss/g_adversarial",
        "loss/d_adversarial",
        "critic/accuracy",
        "critic/margin",
      ];
      const matching = allIds.filter(
        (id) => id.startsWith("loss/") || id.startsWith("critic/"),
      );
      return pick([...priority, ...matching]);
    }
    case "health": {
      const gradAndHealth = allIds.filter(
        (id) =>
          id.startsWith("gradients/") ||
          id.startsWith("grad_norm/") ||
          id.startsWith("health/"),
      );
      const critic = [
        "critic/accuracy",
        "critic/margin",
        ...allIds.filter((id) => id.startsWith("critic/")),
      ];
      return pick([
        "grad_norm/generator",
        "grad_norm/critic",
        "grad_norm/prior",
        ...gradAndHealth,
        "loss/gradient_penalty",
        ...critic,
      ]);
    }
    case "diversity": {
      const priority = [
        "diversity/generated_rms",
        "diversity/reference_rms",
        "diversity/ratio",
        "diversity/pooled4_generated_rms",
        "diversity/pooled4_reference_rms",
        "diversity/pooled4_ratio",
      ];
      const matching = allIds.filter((id) => id.startsWith("diversity/"));
      return pick([...priority, ...matching]);
    }
    case "clear":
    case "none":
      return [];
    default:
      return defaultMetrics(metrics);
  }
}

export function sameMetricSet(a, b) {
  const setA = a instanceof Set ? a : new Set(a);
  const setB = b instanceof Set ? b : new Set(b);
  if (setA.size !== setB.size) return false;
  for (const item of setA) {
    if (!setB.has(item)) return false;
  }
  return true;
}

export function reducedSelection(selected, catalogMetrics = {}) {
  const list = [...(selected || [])];
  if (list.length === 0) return [];
  const metrics = catalogMetrics || {};
  const pinned = PINNED_HEADLINE_METRICS.filter((id) =>
    isScalarTrainingMetric(metrics[id]),
  );
  return [...new Set([...list, ...pinned])];
}

function parseStepBound(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const trimmed = String(raw).trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

export function parseViewQuery(searchString, catalogMetrics = {}) {
  const params = new URLSearchParams(searchString || "");
  const metrics = catalogMetrics || {};
  const scaleRaw = params.get("scale");
  const scale = VALID_SCALES.includes(scaleRaw) ? scaleRaw : "linear";
  const smoothRaw = params.get("smooth");
  const smooth = VALID_SMOOTHING.includes(smoothRaw) ? smoothRaw : "0";

  let stepFrom = parseStepBound(params.get("from"));
  let stepTo = parseStepBound(params.get("to"));
  if (stepFrom !== null && stepTo !== null && stepFrom > stepTo) {
    stepFrom = null;
    stepTo = null;
  }

  let selected;
  if (!params.has("metrics")) {
    selected = defaultMetrics(metrics);
  } else {
    const raw = (params.get("metrics") || "").trim();
    if (!raw || raw === "none") {
      selected = [];
    } else {
      const valid = [
        ...new Set(
          raw
            .split(",")
            .map((id) => id.trim())
            .filter((id) => id && isScalarTrainingMetric(metrics[id])),
        ),
      ].slice(0, MAX_CHART_SERIES);
      selected = valid.length > 0 ? valid : defaultMetrics(metrics);
    }
  }

  return { selected, scale, smooth, stepFrom, stepTo };
}

export function buildViewSearch(
  { selected = [], scale = "linear", smooth = "0", stepFrom = null, stepTo = null },
  catalogMetrics = {},
) {
  const params = new URLSearchParams();
  const list = [...selected];
  const defaults = defaultMetrics(catalogMetrics);
  if (!sameMetricSet(list, defaults)) {
    params.set("metrics", list.join(","));
  }
  if (VALID_SCALES.includes(scale) && scale !== "linear") {
    params.set("scale", scale);
  }
  if (VALID_SMOOTHING.includes(smooth) && smooth !== "0") {
    params.set("smooth", smooth);
  }
  if (Number.isSafeInteger(stepFrom) && stepFrom >= 0) {
    params.set("from", String(stepFrom));
  }
  if (Number.isSafeInteger(stepTo) && stepTo >= 0) {
    params.set("to", String(stepTo));
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

export function deduplicatedTopologyParts(topology) {
  if (!topology || typeof topology !== "object") return [];
  const label = typeof topology.label === "string" ? topology.label.trim() : "";
  const detail = typeof topology.detail === "string" ? topology.detail.trim() : "";
  const parts = [];
  if (label) parts.push(label);
  if (detail && detail.toLowerCase() !== label.toLowerCase()) parts.push(detail);
  return parts;
}

export function formatTopologyPill(topology) {
  return deduplicatedTopologyParts(topology).join(" · ");
}

export function formatBatchPill(run) {
  if (!run || typeof run !== "object") return "";
  const label = typeof run.batch_label === "string" ? run.batch_label.trim() : "";
  const detail = typeof run.batch_detail === "string" ? run.batch_detail.trim() : "";
  return [label, detail].filter(Boolean).join(" · ");
}

export function formatHostPill(run) {
  if (!run || typeof run !== "object") return "";
  const host = typeof run.hostname === "string" ? run.hostname.trim() : "";
  if (!host) return "";
  return `${run.remote ? "Remote · " : ""}${host}`;
}

export function stepMultiplier(run) {
  if (!run || typeof run !== "object") return 1;
  const candidates = [
    run.step_multiplier,
    run.topology?.world_size,
    run.world_size,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      const n = Math.trunc(candidate);
      if (n >= 1) return n;
    }
  }
  return 1;
}

const ACTIONABLE_WARNING_RE =
  /contention|shar(?:e|es|ing)\s+.*(?:device|gpu|cuda)|same\s+(?:device|gpu|cuda)|evaluat.*(?:device|gpu|cuda)|collapse|\bnan\b|\binf\b|diverg|out of memory|\boom\b|memory/i;
const NOISY_TRACEBACK_RE =
  /torch\.compile|torch\._dynamo|torch\._inductor|inductor|dynamo|traceback \(most recent call last\)|File ".*", line \d+/i;

export function collectHealthAlerts(run) {
  if (!run || typeof run !== "object") return [];
  const seen = new Set();
  const result = [];
  const push = (raw) => {
    const clean = String(raw ?? "").replace(/\s+/g, " ").trim();
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    result.push(clean);
  };
  if (Array.isArray(run.health_alerts)) {
    for (const item of run.health_alerts) {
      if (typeof item === "string") push(item);
    }
  }
  if (Array.isArray(run.warnings)) {
    for (const warning of run.warnings) {
      if (typeof warning !== "string") continue;
      if (warning.length > 240 && NOISY_TRACEBACK_RE.test(warning)) continue;
      const clean = warning.replace(/\s+/g, " ").trim();
      if (clean.length > 240 && !ACTIONABLE_WARNING_RE.test(clean.slice(0, 240))) continue;
      if (ACTIONABLE_WARNING_RE.test(clean)) {
        push(clean.length > 240 ? `${clean.slice(0, 240)}…` : clean);
      }
    }
  }
  return result;
}
