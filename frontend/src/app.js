import { evaluationShelf } from "./evaluations.js";
import { init, use } from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);

const $ = (id) => document.getElementById(id);
const colors = [
  "#d7bb81",
  "#8fbdaa",
  "#b3a5d2",
  "#d69478",
  "#85b0cb",
  "#c5b38f",
];
const state = {
  run: null,
  catalog: null,
  evaluationMetrics: {},
  evaluationResults: [],
  consoleSupported: false,
  map: null,
  selected: new Set(),
  groups: new Map(),
  charts: new Map(),
  worker: null,
  stream: null,
  cursor: null,
  sequence: 0,
  epoch: 0,
  ready: false,
  pending: false,
  bucket: 1,
  queue: Promise.resolve(),
  queueBytes: 0,
  queueCount: 0,
  retry: null,
  retries: 0,
  renderTimer: null,
  sampleVersions: new Map(),
  stepFrom: null,
  stepTo: null,
  refreshing: false,
  bootstrapSignal: false,
  coarsenAttempts: 0,
};
let requestID = 0;
/* host.js falls back to a portable SHA-256, so a missing Web Crypto API no longer
   stops the reducer; name the cause anyway instead of relaying a raw TypeError. */
const WEB_CRYPTO_FAILURE = /crypto|subtle|reading 'digest'|bundled reducer digest/i;
function viewWorkerError(message) {
  const text = message || "View worker failed";
  return WEB_CRYPTO_FAILURE.test(text)
    ? `${text} — this browser exposed no Web Crypto API (crypto.subtle), which browsers withhold from plain-HTTP origins other than localhost. Open the viewer over HTTPS (for example \`tailscale serve\`) or from the server itself.`
    : text;
}
class ViewWorker {
  constructor() {
    this.worker = new Worker("/assets/view-worker.js", { type: "module" });
    this.pending = null;
    this.closed = false;
    this.worker.onmessage = ({ data }) => {
      if (!this.pending || data.id !== this.pending.id) return;
      const p = this.pending;
      this.pending = null;
      clearTimeout(p.timer);
      data.error
        ? p.reject(new Error(viewWorkerError(data.error)))
        : p.resolve(data.ok);
    };
    this.worker.onerror = (e) => this.close(new Error(viewWorkerError(e.message)));
  }
  call(message) {
    if (this.closed) return Promise.reject(new Error("View worker closed"));
    if (this.pending)
      return Promise.reject(new Error("View worker request overlap"));
    return new Promise((resolve, reject) => {
      const id = ++requestID;
      const timer = setTimeout(
        () => this.close(new Error("View worker timed out")),
        10000,
      );
      this.pending = { id, resolve, reject, timer };
      this.worker.postMessage({ ...message, id });
    });
  }
  close(error = new Error("View worker replaced")) {
    this.closed = true;
    this.worker.terminate();
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(error);
      this.pending = null;
    }
  }
}
function connection(label, kind = "") {
  const element = $("connection");
  element.className = `connection ${kind}`;
  element.replaceChildren(
    Object.assign(document.createElement("i"), { ariaHidden: "true" }),
    document.createTextNode(label),
  );
}
function notice(message) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
}
function stopStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
  clearTimeout(state.retry);
  state.retry = null;
}
function showLogin() {
  stopStream();
  state.epoch++;
  state.ready = false;
  state.worker?.close();
  state.worker = null;
  $("workspace").hidden = true;
  $("login").hidden = false;
  connection("Session required");
}
async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, {
    credentials: "same-origin",
    ...options,
  });
  if (response.status === 401) {
    showLogin();
    throw new Error("Session expired. Enter your session token.");
  }
  if (!response.ok && response.status !== 202) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      message = data.error || data.detail || message;
    } catch {}
    throw new Error(message);
  }
  return { status: response.status, data: await response.json() };
}
const base = () => `/runs/${encodeURIComponent(state.run.run_id)}`;
const evaluations = evaluationShelf(api, base, (results) => {
  const definitions = {};
  state.evaluationResults = [];
  for (const {event, catalog} of results) {
    if (event.status !== 'complete' || !event.source_position_known) continue;
    for (const [id, value] of Object.entries(event.metrics || {})) {
      const definition = catalog.metrics[id];
      if (definition?.kind !== 'scalar') continue;
      // Protocol and definition are separate selectable series, never averaged.
      const key = `evaluation:${id}:${definition.definition_hash}:${event.protocol_sha256}`;
      definitions[key] = {...definition, label: `${definition.label || id} · Evaluation`,
        metricID: id, evaluation: true, protocol: event.protocol_sha256};
      state.evaluationResults.push({key, value, event});
      if (!state.evaluationMetrics[key] && state.selected.size < 8) state.selected.add(key);
    }
  }
  state.evaluationMetrics = definitions;
  if (state.catalog) { renderCatalog(); render(); }
});
function metricDefinitions() { return {...(state.catalog?.metrics || {}), ...state.evaluationMetrics}; }
function trainingSelection() { return [...state.selected].filter(id => !state.evaluationMetrics[id]); }
const fmt = (value) =>
  value === null || value === undefined
    ? "—"
    : Number(value).toLocaleString(undefined, { maximumSignificantDigits: 6 });
function updateRun(run) {
  state.run = run;
  evaluations.update(state.catalog, run);
  $("run-name").textContent =
    run.config?.name || run.name || "Training experiment";
  $("run-id").textContent = run.run_id;
  $("run-status").textContent = run.status || "Unknown";
  $("step").textContent = fmt(run.steps);
  $("durable").textContent = fmt(run.last_durable_step);
  const consistency = run.metric_consistency;
  $("metric-consistency").textContent = !consistency ? "" :
    consistency.status === "caught_up" ? `Metrics committed through step ${fmt(consistency.committed_step)} · view caught up` :
    consistency.status === "pending" ? `Metrics committed through step ${fmt(consistency.committed_step)} · view catching up` :
    `Metric projection status unavailable${consistency.committed_step === undefined ? "" : ` · committed step ${fmt(consistency.committed_step)}`}`;
  $("total-steps").textContent = run.total_steps
    ? `of ${fmt(run.total_steps)} configured steps`
    : "Completed optimizer updates";
  $("raw-events").href = `/api/v1${base()}/events`;
}
async function refreshArtifacts() {
  if (!state.run) return;
  const runID = state.run.run_id;
  const request = state.artifactRequest = (state.artifactRequest || 0) + 1;
  const result = await api(`${base()}/artifacts`);
  if (state.run?.run_id !== runID || request !== state.artifactRequest) return;
  const artifacts = result.data.artifacts || {};
  const signature = JSON.stringify([runID, artifacts]);
  if (signature === state.artifactSignature) return;
  state.artifactSignature = signature;
  renderArtifacts(artifacts);
}
const MAX_SAMPLE_GROUPS = 20;
const MAX_SAMPLE_VERSIONS = 100;
const provenanceStep = (artifact) => {
  const step = artifact.provenance?.step;
  return Number.isFinite(step) ? step : -1;
};
const provenanceSequence = (artifact) => {
  const sequence = artifact.provenance?.sample_sequence;
  return Number.isFinite(sequence) ? sequence : -1;
};
function sampleGroups(artifacts) {
  // Samples are indexed by a short stable name ('g' generated, 'x' real) and
  // grouped per modality so an image name shows one picture at a time.
  const groups = new Map();
  for (const [id, artifact] of Object.entries(artifacts)) {
    const name =
      typeof artifact.name === "string" && artifact.name ? artifact.name : id;
    const modality = artifact.modality || "unspecified";
    const key = `${name}\u0000${modality}`;
    if (!groups.has(key)) groups.set(key, { key, name, modality, versions: [] });
    groups.get(key).versions.push({ id, artifact });
  }
  for (const group of groups.values()) {
    group.versions.sort(
      (a, b) =>
        provenanceStep(a.artifact) - provenanceStep(b.artifact) ||
        provenanceSequence(a.artifact) - provenanceSequence(b.artifact) ||
        (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );
    if (group.versions.length > MAX_SAMPLE_VERSIONS)
      group.versions = group.versions.slice(-MAX_SAMPLE_VERSIONS);
    group.latest = group.versions[group.versions.length - 1];
  }
  return [...groups.values()].sort(
    (a, b) =>
      provenanceStep(b.latest.artifact) - provenanceStep(a.latest.artifact) ||
      Number(b.modality === "image") - Number(a.modality === "image") ||
      (a.name < b.name ? -1 : a.name > b.name ? 1 : 0),
  );
}
function renderArtifacts(artifacts) {
  $("artifact-items").replaceChildren();
  const groups = sampleGroups(artifacts);
  $("artifacts").hidden = !groups.length;
  for (const group of groups.slice(0, MAX_SAMPLE_GROUPS))
    $("artifact-items").append(renderSampleGroup(group));
}
function renderSampleGroup(group) {
  const li = document.createElement("li");
  li.dataset.sample = group.name;
  li.dataset.modality = group.modality;
  const info = document.createElement("div");
  const heading = document.createElement("strong");
  heading.className = "sample-name";
  heading.textContent = group.name;
  const details = document.createElement("span");
  info.append(heading, details);
  li.append(info);
  const body = document.createElement("div");
  body.className = "sample-body";
  li.append(body);
  const pinned = state.sampleVersions.get(group.key);
  const pinnedIndex = group.versions.findIndex((v) => v.id === pinned);
  let index = pinnedIndex < 0 ? group.versions.length - 1 : pinnedIndex;
  let history = null;
  let position = null;
  let latestButton = null;
  if (group.versions.length > 1) {
    history = document.createElement("div");
    history.className = "sample-history";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(group.versions.length - 1);
    slider.step = "1";
    slider.value = String(index);
    slider.className = "sample-slider";
    slider.setAttribute("aria-label", `${group.name} sample history`);
    position = document.createElement("span");
    position.className = "sample-position";
    latestButton = document.createElement("button");
    latestButton.className = "secondary sample-latest";
    latestButton.textContent = "Latest";
    latestButton.onclick = () => {
      slider.value = String(group.versions.length - 1);
      state.sampleVersions.delete(group.key);
      show(group.versions.length - 1);
    };
    slider.oninput = () => {
      const chosen = Number(slider.value);
      if (chosen === group.versions.length - 1)
        state.sampleVersions.delete(group.key);
      else state.sampleVersions.set(group.key, group.versions[chosen].id);
      show(chosen);
    };
    history.append(slider, position, latestButton);
    li.append(history);
  }
  function show(chosen) {
    index = chosen;
    const { id, artifact } = group.versions[index];
    const step = artifact.provenance?.step;
    const shape = artifact.shape || artifact.metadata?.shape;
    details.textContent = [
      artifact.role || "Artifact",
      group.modality,
      artifact.media_type || "unknown type",
      `Step ${fmt(step)}`,
      Array.isArray(shape) ? `Shape ${shape.join(" × ")}` : null,
      artifact.bytes !== undefined ? `${fmt(artifact.bytes)} bytes` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    if (position) {
      const latest = index === group.versions.length - 1;
      position.textContent =
        `Version ${index + 1} of ${group.versions.length} · step ${fmt(step)}` +
        (latest ? " · latest" : "");
      latestButton.hidden = latest;
    }
    body.replaceChildren(sampleVersion(group, id, artifact));
  }
  show(index);
  return li;
}
function sampleVersion(group, id, artifact) {
  const fragment = document.createDocumentFragment();
  if (artifact.status === "unavailable") {
    const reason = document.createElement("p");
    reason.textContent = artifact.reason || "Artifact unavailable";
    fragment.append(reason);
    return fragment;
  }
  const path = `/api/v1${base()}/artifacts/${encodeURIComponent(id)}`;
  const shape = artifact.shape || artifact.metadata?.shape;
  const controls = document.createElement("div");
  controls.className = "artifact-controls";
  const download = document.createElement("a");
  download.href = path;
  download.download =
    artifact.media_type === "application/json" ? "samples.json" : "artifact";
  download.className = "text-link";
  download.textContent = "Download";
  controls.append(download);
  if (
    artifact.modality === "image" && artifact.media_type === "image/png" &&
    Number.isSafeInteger(artifact.bytes) && artifact.bytes > 0 && artifact.bytes <= 8388608 &&
    [artifact.width, artifact.height].every((n) => Number.isSafeInteger(n) && n > 0 && n <= 4096) &&
    artifact.width * artifact.height <= 4194304
  ) {
    const image = document.createElement("img");
    image.className = "image-grid";
    image.alt = `Sample ${group.name} image grid at step ${fmt(artifact.provenance?.step)}`;
    image.width = artifact.width;
    image.height = artifact.height;
    image.loading = "lazy";
    image.decoding = "async";
    image.src = path;
    image.onerror = () => {
      const error = document.createElement("p");
      error.textContent = "Image unavailable or removed by preview retention.";
      image.replaceWith(error);
    };
    fragment.append(image);
    download.download = "grid.png";
  }
  if (
    artifact.modality === "tensor" &&
    artifact.media_type === "application/json" &&
    Number.isSafeInteger(artifact.bytes) &&
    artifact.bytes <= 8388608 &&
    Array.isArray(shape) &&
    shape.length > 0 &&
    shape.length <= 8 &&
    shape.every((n) => Number.isSafeInteger(n) && n > 0) &&
    shape.reduce((a, b) => a * b, 1) <= 262144
  ) {
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = "Preview numbers";
    const preview = document.createElement("pre");
    preview.className = "numeric-preview";
    preview.hidden = true;
    button.onclick = async () => {
      if (!preview.hidden) {
        preview.hidden = true;
        button.textContent = "Preview numbers";
        return;
      }
      button.disabled = true;
      try {
        preview.textContent = await numericalPreview(path, shape);
        preview.hidden = false;
        button.textContent = "Hide numbers";
      } catch (error) {
        preview.textContent = error.message;
        preview.hidden = false;
      } finally {
        button.disabled = false;
      }
    };
    controls.append(button);
    fragment.append(preview);
  }
  if (artifact.modality === "tensor" && controls.children.length === 1) {
    const hint = document.createElement("p");
    hint.textContent = "Numeric preview supports JSON tensors up to 8 MiB and 262,144 values. Download this tensor to inspect it.";
    fragment.append(hint);
  }
  fragment.append(controls);
  return fragment;
}
async function numericalPreview(path, shape) {
  const response = await fetch(path, { credentials: "same-origin" });
  if (!response.ok)
    throw new Error(`Artifact unavailable (${response.status})`);
  const reader = response.body.getReader();
  const chunks = [];
  let bytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes += value.length;
    if (bytes > 8388608) {
      await reader.cancel();
      throw new Error("Preview exceeds 8 MiB; use Download.");
    }
    chunks.push(value);
  }
  const buffer = new Uint8Array(bytes);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }
  const payload = JSON.parse(
    new TextDecoder("utf-8", { fatal: true }).decode(buffer),
  );
  if (
    !Array.isArray(payload.shape) ||
    JSON.stringify(payload.shape) !== JSON.stringify(shape)
  )
    throw new Error("Artifact shape differs from its descriptor.");
  const flat = [];
  function visit(value, depth) {
    if (depth > 8 || flat.length > 262144)
      throw new Error("Numerical preview exceeds its shape limit.");
    if (depth < shape.length) {
      if (!Array.isArray(value) || value.length !== shape[depth])
        throw new Error("Artifact values differ from its shape.");
      for (const item of value) visit(item, depth + 1);
    } else if (typeof value === "number" && Number.isFinite(value))
      flat.push(value);
    else
      throw new Error(
        "Artifact contains unsupported numerical values; use Download.",
      );
  }
  visit(payload.samples, 0);
  if (flat.length !== shape.reduce((a, b) => a * b, 1))
    throw new Error("Artifact values differ from its shape.");
  return `${flat.slice(0, 128).map(String).join(", ")}${flat.length > 128 ? `\nShowing 128 of ${flat.length} values.` : ""}`;
}
function defaults() {
  const preferred = [
    "loss/g_total",
    "loss/d_total",
    "loss/gradient_penalty",
    "loss/prior_regularizer",
  ];
  return [...preferred.filter((id) => id in state.catalog.metrics), ...Object.keys(state.evaluationMetrics)].slice(0, 8);
}
function renderCatalog() {
  const search = $("search").value.toLowerCase();
  $("metric-count").textContent = Object.values(metricDefinitions()).filter(d => (d.evaluation || d.scope !== "snapshot") && d.kind === "scalar").length;
  $("metric-list").replaceChildren();
  for (const [id, definition] of Object.entries(metricDefinitions())) {
    if ((!definition.evaluation && definition.scope === "snapshot") || definition.kind !== "scalar") continue;
    if (!`${id} ${definition.label}`.toLowerCase().includes(search)) continue;
    const label = document.createElement("label");
    label.className = "metric-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selected.has(id);
    input.value = id;
    input.setAttribute("aria-label", definition.label || id);
    input.addEventListener("change", () => {
      if (input.checked && state.selected.size >= 8) {
        input.checked = false;
        notice("Choose at most 8 metrics for this view.");
        return;
      }
      input.checked ? state.selected.add(id) : state.selected.delete(id);
      reconfigure();
    });
    const text = document.createElement("span");
    text.textContent = definition.label || id;
    const small = document.createElement("small");
    small.textContent = definition.evaluation ? `${definition.metricID} · protocol ${definition.protocol.slice(0, 8)}` : id;
    text.append(small);
    label.append(input, text);
    $("metric-list").append(label);
  }
}
async function metadata(expectedEpoch = state.epoch) {
  const [run, catalog, views] = await Promise.all([
    api(base()),
    api(`${base()}/metrics/catalog`),
    api(`${base()}/views`),
  ]);
  if (expectedEpoch !== state.epoch) return;
  updateRun(run.data);
  state.catalog = catalog.data;
  evaluations.update(state.catalog, state.run);
  state.map = views.data.map_revision;
  evaluations.refresh(views.data.streams || []).catch(error => notice(error.message));
  if (views.data.discovery_error) notice(views.data.discovery_error);
  state.selected = new Set(
    [...state.selected].filter((id) => metricDefinitions()[id]?.kind === "scalar" && (metricDefinitions()[id]?.evaluation || metricDefinitions()[id]?.scope !== "snapshot")),
  );
  renderCatalog();
  await refreshArtifacts();
  $("console-settings").hidden = !state.consoleSupported;
  if (state.consoleSupported) {
    const settings = await api(`${base()}/console`);
    $("progress-every").value = settings.data.progress_every;
  }
}
async function connect() {
  connection("Connecting…");
  const capability = await api("/capabilities");
  state.consoleSupported = capability.data.controls?.includes("console") === true;
  if (capability.data.run_id === null) {
    waitForRun();
    return;
  }
  const run = await api(`/runs/${encodeURIComponent(capability.data.run_id)}`);
  updateRun(run.data);
  await metadata();
  state.selected = new Set(defaults());
  renderCatalog();
  $("login").hidden = true;
  $("workspace").hidden = false;
  state.bucket = Math.max(1, Math.ceil((state.run.steps || 1) / 180));
  await reconfigure();
}
function waitForRun() {
  stopStream();
  $("login").hidden = true;
  $("workspace").hidden = false;
  $("run-name").textContent = "Waiting for training";
  $("run-status").textContent = "Waiting";
  $("coverage").textContent =
    "This server is ready. The experiment will appear when training starts.";
  connection("Waiting for run");
  const stream = new EventSource("/api/v1/stream");
  state.stream = stream;
  stream.addEventListener("metadata", () => {
    stream.close();
    connect().catch((error) => notice(error.message));
  });
  stream.onerror = () => {
    stream.close();
    connection("Disconnected · reconnecting", "error");
    state.retry = setTimeout(
      () => connect().catch((error) => notice(error.message)),
      2000,
    );
  };
}
function requestPath() {
  const params = new URLSearchParams({
    series: trainingSelection().sort().join(","),
    bucket_steps: String(state.bucket),
  });
  if (state.stepFrom !== null) params.set("step_from", state.stepFrom);
  if (state.stepTo !== null) params.set("step_to", state.stepTo);
  return `${base()}/views/${state.map}/bootstrap?${params}`;
}
async function reconfigure({ coarser = false } = {}) {
  const epoch = ++state.epoch;
  stopStream();
  state.worker?.close();
  state.worker = new ViewWorker();
  state.ready = false;
  state.pending = false;
  state.bootstrapSignal = false;
  state.cursor = null;
  state.sequence = 0;
  state.queue = Promise.resolve();
  state.queueCount = 0;
  state.queueBytes = 0;
  if (coarser) {
    state.bucket *= 2;
    state.coarsenAttempts++;
  } else state.coarsenAttempts = 0;
  state.groups.clear();
  $("stream-position").textContent = "Loading history";
  delete $("stream-position").dataset.projection;
  render();
  $("coverage").textContent = "Preparing bounded history…";
  connection("Loading history…");
  if (!trainingSelection().length) {
    $("coverage").textContent = state.selected.size ? "Evaluation snapshots · no training series selected" : "No metrics selected";
    openStream(epoch);
    return;
  }
  // Wait for ready before loading history so completion notifications cannot
  // race subscription registration, and startup performs a single request.
  openStream(epoch);
}
async function loadBootstrap(epoch) {
  if (epoch !== state.epoch || state.pending || state.ready) return;
  state.pending = true;
  try {
    const response = await api(requestPath());
    if (epoch !== state.epoch) return;
    if (response.status === 202) {
      $("coverage").textContent =
        "Indexing history · waiting for stream notification";
      return;
    }
    const bootstrap = response.data;
    const result = await state.worker.call({
      op: "bootstrap",
      bootstrap,
      selected: trainingSelection(),
      stepFrom: state.stepFrom,
      stepTo: state.stepTo,
    });
    if (epoch !== state.epoch) return;
    state.groups.clear();
    for (const group of result.groups)
      state.groups.set(JSON.stringify(group.key), group);
    state.cursor = result.cursor;
    state.sequence = result.projection_sequence;
    state.ready = true;
    state.retries = 0;
    state.bucket = bootstrap.bucket_steps;
    $("bucket-label").textContent =
      `${fmt(state.bucket)} step${state.bucket === 1 ? "" : "s"} / bucket`;
    $("coverage").textContent = "History loaded";
    markViewUpdated();
    notice("");
    render();
    stopStream();
    openStream(epoch);
  } catch (error) {
    if (epoch === state.epoch) {
      connection("View unavailable", "error");
      notice(error.message);
      if (
        /2048 groups|1 MiB|capacity/.test(error.message) &&
        state.coarsenAttempts < 4
      ) {
        queueMicrotask(() => {
          if (epoch === state.epoch) reconfigure({ coarser: true });
        });
      }
    }
  } finally {
    if (epoch === state.epoch) {
      state.pending = false;
      if (state.bootstrapSignal) {
        state.bootstrapSignal = false;
        loadBootstrap(epoch);
      }
    }
  }
}
function scheduleReconnect(epoch, catchingUp = false) {
  if (epoch !== state.epoch || state.retry) return;
  const delay = catchingUp ? 100 : Math.min(1000 * 2 ** state.retries++, 15000);
  connection("Disconnected · reconnecting", "error");
  state.retry = setTimeout(async () => {
    state.retry = null;
    await state.queue.catch(() => {});
    if (epoch === state.epoch) openStream(epoch);
  }, delay);
}
function parse(event) {
  if (event.data.length > 131072)
    throw new Error("Stream frame exceeds browser byte budget");
  return JSON.parse(event.data);
}
function openStream(epoch) {
  if (epoch !== state.epoch) return;
  if (state.stream) state.stream.close();
  const query = new URLSearchParams({ stream_id: `projection:${state.map}` });
  if (state.cursor) query.set("cursor", state.cursor);
  const stream = new EventSource(`/api/v1${base()}/stream?${query}`);
  const controlsOnly = trainingSelection().length === 0;
  state.stream = stream;
  stream.onopen = () => {
    if (epoch === state.epoch && (state.ready || controlsOnly))
      connection("Live stream", "live");
  };
  stream.onerror = () => {
    stream.close();
    if (state.stream === stream) state.stream = null;
    scheduleReconnect(epoch);
  };
  for (const name of ["ready", "heartbeat"])
    stream.addEventListener(name, (event) => {
      if (epoch !== state.epoch) return;
      try {
        const data = parse(event);
        if (data.run) updateRun(data.run);
        if (name === "ready") api(`${base()}/views`).then(result => evaluations.refresh(result.data.streams || [])).catch(error => notice(error.message));
        if (name === "ready") {
          state.retries = 0;
          refreshArtifacts().catch(error => notice(error.message));
        }
        if (state.ready || controlsOnly) connection("Live stream", "live");
        else if (name === "ready") loadBootstrap(epoch);
      } catch (error) {
        notice(error.message);
      }
    });
  stream.addEventListener("bootstrap_ready", () => {
    if (state.ready || controlsOnly) return;
    if (state.pending) state.bootstrapSignal = true;
    else loadBootstrap(epoch);
  });
  stream.addEventListener("metadata", () => refreshMetadata(epoch));
  stream.addEventListener("stream_added", () => {
    if (epoch === state.epoch) api(`${base()}/views`).then(result => evaluations.refresh(result.data.streams || [])).catch(error => notice(error.message));
  });
  stream.addEventListener("discovery_error", event => {
    if (epoch === state.epoch) { try { notice(parse(event).reason); } catch (error) { notice(error.message); } }
  });
  stream.addEventListener("artifacts", () => {
    if (epoch === state.epoch)
      refreshArtifacts().catch((error) => notice(error.message));
  });
  stream.addEventListener("gap", () => {
    if (epoch !== state.epoch) return;
    stream.close();
    notice("Catching up from the last applied frame…");
    // A gap concerns delivery, not mathematical state. Preserve the worker and
    // drain its queue before resuming the acknowledged cursor. Each bounded
    // replay makes progress even when a cold bootstrap has a large live suffix.
    scheduleReconnect(epoch, true);
  });
  stream.addEventListener("reset_required", () => {
    if (epoch !== state.epoch) return;
    stream.close();
    notice("Run history changed. Reloading the current view.");
    refreshMetadata(epoch);
  });
  stream.addEventListener("frame", (event) => {
    if (epoch !== state.epoch || !state.ready) return;
    let envelope;
    try {
      envelope = parse(event);
    } catch (error) {
      notice(error.message);
      stream.close();
      refreshMetadata(epoch);
      return;
    }
    if (
      state.queueCount >= 128 ||
      state.queueBytes + event.data.length > 1048576
    ) {
      stream.close();
      connection("Catching up…", "error");
      notice(
        "The browser queue reached its limit. Reconnecting from the last applied frame.",
      );
      scheduleReconnect(epoch, true);
      return;
    }
    state.queueCount++;
    state.queueBytes += event.data.length;
    state.queue = state.queue
      .then(async () => {
        if (epoch !== state.epoch) return;
        const result = await state.worker.call({ op: "frame", envelope });
        if (epoch !== state.epoch) return;
        for (const group of result.groups)
          state.groups.set(JSON.stringify(group.key), group);
        state.cursor = result.cursor;
        state.sequence = result.projection_sequence;
        markViewUpdated();
        if (result.latestStep !== null && result.latestStep !== undefined) {
          state.run.steps = Math.max(state.run.steps || 0, result.latestStep);
          evaluations.update(state.catalog, state.run);
          $("step").textContent = fmt(state.run.steps);
        }
        scheduleRender();
      })
      .catch((error) => {
        if (epoch !== state.epoch) return;
        stream.close();
        notice(error.message);
        if (error.message.includes("capacity") && state.coarsenAttempts < 4)
          reconfigure({ coarser: true });
        else if (error.message.includes("capacity")) {
          state.ready = false;
          notice(
            "This view contains too many attempt partitions. Select fewer metrics or narrow the step range.",
          );
        } else refreshMetadata(epoch);
      })
      .finally(() => {
        if (epoch === state.epoch) {
          state.queueCount--;
          state.queueBytes -= event.data.length;
        }
      });
  });
}
async function refreshMetadata(epoch) {
  if (epoch !== state.epoch || state.refreshing) return;
  state.refreshing = true;
  try {
    stopStream();
    await state.queue.catch(() => {});
    await metadata();
    if (epoch === state.epoch) await reconfigure();
  } catch (error) {
    notice(error.message);
    connection("Reconnect required", "error");
  } finally {
    state.refreshing = false;
  }
}
function scheduleRender() {
  if (state.renderTimer) return;
  state.renderTimer = setTimeout(() => {
    state.renderTimer = null;
    render();
  }, 120);
}
function markViewUpdated() {
  $("stream-position").dataset.projection = String(state.sequence);
  $("stream-position").textContent =
    `View updated ${new Date().toLocaleTimeString()}`;
}
function partitions() {
  const result = new Map();
  for (const group of state.groups.values()) {
    const [metric, definition, attempt] = group.key;
    if (!state.selected.has(metric)) continue;
    const id = JSON.stringify([metric, definition, attempt]);
    if (!result.has(id))
      result.set(id, { metric, definition, attempt, points: new Map() });
    const target = result.get(id);
    for (const point of [
      group.value.first,
      group.value.min,
      group.value.max,
      group.value.last,
    ])
      if (point) target.points.set(JSON.stringify(point.position), point);
  }
  for (const part of result.values())
    part.points = [...part.points.values()].sort(
      (a, b) =>
        a.position[0] - b.position[0] ||
        (a.position[1] < b.position[1]
          ? -1
          : a.position[1] > b.position[1]
            ? 1
            : 0),
    );
  for (const {key, value, event} of state.evaluationResults) {
    if (!state.selected.has(key) || (state.stepFrom !== null && event.step < state.stepFrom) ||
        (state.stepTo !== null && event.step > state.stepTo)) continue;
    const id = JSON.stringify([key, event.attempt_id]);
    if (!result.has(id)) result.set(id, {metric: key, definition: metricDefinitions()[key].definition_hash,
      attempt: event.attempt_id, evaluation: true, points: []});
    result.get(id).points.push({value, position: [event.step, event.evaluation_id], event});
  }
  for (const part of result.values()) if (part.evaluation)
    part.points.sort((a, b) => a.position[0] - b.position[0] || a.position[1].localeCompare(b.position[1]));
  return [...result.values()];
}
function render() {
  if (!state.catalog) return;
  const parts = partitions();
  const active = new Set();
  $("values-table").replaceChildren();
  for (const metric of state.selected) {
    const items = parts.filter((part) => part.metric === metric);
    active.add(metric);
    let card = state.charts.get(metric);
    if (!card) {
      const element = document.createElement("article");
      element.className = "chart-card";
      const heading = document.createElement("div");
      heading.className = "chart-heading";
      const title = document.createElement("h3");
      title.textContent = metricDefinitions()[metric]?.label || metric;
      const value = document.createElement("span");
      value.className = "value";
      heading.append(title, value);
      const description = document.createElement("p");
      description.className = "chart-description";
      description.textContent = metricDefinitions()[metric]?.evaluation
        ? `${metricDefinitions()[metric].metricID} · protocol ${metricDefinitions()[metric].protocol.slice(0, 8)}` : metric;
      const canvas = document.createElement("div");
      canvas.className = "chart-canvas";
      canvas.setAttribute("role", "img");
      canvas.setAttribute(
        "aria-label",
        `${title.textContent} over training steps; exact latest values are in the data table`,
      );
      const note = document.createElement("p");
      note.className = "chart-note";
      element.append(heading, description, canvas, note);
      $("charts").append(element);
      const chart = init(canvas, null, { renderer: "canvas" });
      card = { element, chart, value, note };
      state.charts.set(metric, card);
    }
    const evaluation = !!metricDefinitions()[metric]?.evaluation;
    const alpha = evaluation ? 0 : Number($("smoothing").value),
      log = $("scale").value === "log";
    let omitted = 0;
    const series = [];
    let latest = null;
    items.forEach((part, index) => {
      const points = part.points;
      if (
        points.length &&
        (!latest || points.at(-1).position[0] > latest.position[0])
      )
        latest = points.at(-1);
      let ema = null;
      const smooth = [];
      const raw = points.map((point) => {
        if (log && point.value <= 0) {
          omitted++;
          ema = null;
          smooth.push([point.position[0], null]);
          return [point.position[0], null];
        }
        ema =
          ema === null ? point.value : alpha * point.value + (1 - alpha) * ema;
        smooth.push([point.position[0], ema]);
        return [point.position[0], point.value];
      });
      const name = `${part.attempt.slice(0, 8)} · ${part.definition.slice(0, 6)}`;
      series.push({
        name,
        type: "line",
        showSymbol: evaluation || raw.filter(point => point[1] !== null).length === 1,
        symbolSize: 6,
        connectNulls: false,
        data: raw,
        lineStyle: { width: alpha ? 1 : 1.7, opacity: evaluation ? 0 : alpha ? 0.35 : 1 },
        itemStyle: { color: colors[index % colors.length] },
        animation: false,
      });
      if (alpha)
        series.push({
          name: `${name} · EMA`,
          type: "line",
          showSymbol: smooth.filter(point => point[1] !== null).length === 1,
          symbolSize: 6,
          data: smooth,
          lineStyle: { width: 2 },
          itemStyle: { color: colors[index % colors.length] },
          animation: false,
        });
      for (const last of evaluation ? points : points.slice(-1)) {
        const row = document.createElement("tr");
        row.dataset.metric = metric;
        row.dataset.step = String(last.position[0]);
        if (evaluation) row.dataset.evaluation = last.event.evaluation_id;
        for (const text of [
          metricDefinitions()[metric]?.metricID || metric,
          String(last.position[0]),
          String(last.value),
          evaluation ? `${part.attempt} · evaluation ${last.event.evaluation_id} · ${last.event.seconds ?? "unknown"} seconds` : part.attempt,
        ]) {
          const cell = document.createElement("td");
          cell.textContent = text;
          row.append(cell);
        }
        $("values-table").append(row);
      }
    });
    card.value.textContent = latest ? fmt(latest.value) : "—";
    card.note.textContent = [
      evaluation ? "Evaluation metric · discrete snapshot measurements; every evaluation is retained. Duration and provenance appear below." : "",
      omitted ? `${omitted} nonpositive points excluded from log scale.` : "",
      alpha
        ? "EMA uses visible envelope points; raw values remain visible."
        : "",
    ]
      .filter(Boolean)
      .join(" ");
    card.chart.setOption(
      {
        animation: false,
        color: colors,
        grid: { left: 54, right: 20, top: 25, bottom: 36 },
        textStyle: { fontFamily: "system-ui" },
        tooltip: {
          trigger: "axis",
          renderMode: "richText",
          backgroundColor: "#28352e",
          borderColor: "#526157",
          textStyle: { color: "#e8e8df", fontSize: 10 },
          formatter: (entries) =>
            entries.length
              ? `Step ${entries[0].value[0]}\n` +
                entries
                  .map((p) => `${p.seriesName}: ${String(p.value[1])}`)
                  .join("\n")
              : "",
        },
        xAxis: {
          type: "value",
          axisLabel: { color: "#85978b", fontSize: 9 },
          axisLine: { lineStyle: { color: "#38483d" } },
          splitLine: { show: false },
          axisTick: { show: false },
        },
        yAxis: {
          type: log ? "log" : "value",
          scale: true,
          axisLabel: { color: "#85978b", fontSize: 9 },
          splitLine: { lineStyle: { color: "#2b3930" } },
          axisLine: { show: false },
        },
        series,
      },
      true,
    );
  }
  const metricOrder = [...state.selected];
  const rows = [...$("values-table").children].sort((a, b) =>
    metricOrder.indexOf(a.dataset.metric) - metricOrder.indexOf(b.dataset.metric) ||
    Number(a.dataset.step) - Number(b.dataset.step) ||
    (a.dataset.evaluation || '').localeCompare(b.dataset.evaluation || ''));
  $("values-table").replaceChildren(...rows);
  for (const [metric, card] of state.charts)
    if (!active.has(metric)) {
      card.chart.dispose();
      card.element.remove();
      state.charts.delete(metric);
    }
  $("empty").hidden = parts.some((part) => part.points.length > 0);
  for (const [metric, prefix] of [
    ["loss/g_total", "g"],
    ["loss/d_total", "d"],
  ]) {
    let latest = null;
    for (const part of parts.filter((p) => p.metric === metric)) {
      const point = part.points.at(-1);
      if (point && (!latest || point.position[0] > latest.position[0]))
        latest = point;
    }
    $(`${prefix}-loss`).textContent = latest ? fmt(latest.value) : "—";
    $(`${prefix}-step`).textContent = latest
      ? `Published at step ${fmt(latest.position[0])}`
      : "No published value";
  }
}
new ResizeObserver(() => {
  for (const card of state.charts.values()) card.chart.resize();
}).observe($("charts"));
$("console-settings").addEventListener("submit", async (event) => {
  event.preventDefault();
  const progress_every = Number($("progress-every").value);
  if (!Number.isSafeInteger(progress_every) || progress_every < 1 || progress_every > 1000000000) {
    $("console-status").textContent = "Choose an integer from 1 to 1,000,000,000.";
    return;
  }
  const button = $("console-settings").querySelector("button");
  button.disabled = true;
  try {
    await api(`${base()}/console`, {method: "PUT", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({progress_every})});
    $("console-status").textContent = `Saved: every ${progress_every} steps. Active CLI checks at update boundaries, at most four times per second. Long updates delay changes; the setting persists on resume.`;
  } catch (error) { $("console-status").textContent = error.message; }
  finally { button.disabled = false; }
});
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = $("token").value;
  $("token").value = "";
  $("login-error").textContent = "";
  try {
    await api("/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    await connect();
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});
$("search").addEventListener("input", () => {
  if (state.catalog) renderCatalog();
});
$("select-defaults").onclick = () => {
  state.selected = new Set(defaults());
  renderCatalog();
  reconfigure();
};
$("select-none").onclick = () => {
  state.selected.clear();
  renderCatalog();
  reconfigure();
};
$("scale").onchange = render;
$("smoothing").onchange = render;
$("reconnect").onclick = () =>
  state.run
    ? refreshMetadata(state.epoch)
    : connect().catch((error) => notice(error.message));
$("apply-range").onclick = () => {
  const from = $("step-from").value,
    to = $("step-to").value;
  const parsed = [
    from === "" ? null : Number(from),
    to === "" ? null : Number(to),
  ];
  if (
    parsed.some((n) => n !== null && (!Number.isSafeInteger(n) || n < 0)) ||
    (parsed.every((n) => n !== null) && parsed[0] > parsed[1])
  ) {
    notice("Enter a valid nonnegative step range.");
    return;
  }
  [state.stepFrom, state.stepTo] = parsed;
  reconfigure();
};
window.addEventListener("pagehide", () => {
  stopStream();
  state.worker?.close();
});
connect().catch((error) => {
  if ($("login").hidden) {
    notice(error.message);
    connection("Unavailable", "error");
    $("workspace").hidden = false;
  }
});
