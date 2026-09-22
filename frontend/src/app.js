import { evaluationShelf } from "./evaluations.js";
import { chartColors as colors, chartStyle, init } from "./chart.js";

const $ = (id) => document.getElementById(id);
const state = {
  run: null,
  catalog: null,
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
// Snapshot evaluations own their results end to end; Learning curves charts the
// training stream only, so an evaluation id never reaches the catalog or a chart.
const evaluations = evaluationShelf(api, base);
function metricDefinitions() { return state.catalog?.metrics || {}; }
function trainingSelection() { return [...state.selected]; }
// The manifest status is an internal word; the badge reads as the person's
// view of the run. Unmapped values fall back to the raw word, capitalised.
const RUN_STATUS_LABELS = {
  initializing: "Starting",
  pending: "Starting",
  starting: "Starting",
  tuning: "Tuning startup",
  running: "Training",
  training: "Training",
  complete: "Complete",
  stopped: "Stopped",
  cancelled: "Stopped",
  interrupted: "Interrupted",
  failed: "Failed",
};
const statusLabel = (status) =>
  RUN_STATUS_LABELS[status] ||
  (status ? status.charAt(0).toUpperCase() + status.slice(1) : "Unknown");
const fmt = (value) =>
  value === null || value === undefined
    ? "—"
    : Number(value).toLocaleString(undefined, { maximumSignificantDigits: 6 });
// Headline counters stay a fixed width: exact below 100k, compact above it.
const count = (value) =>
  !Number.isFinite(value)
    ? "—"
    : Number(value).toLocaleString(
        undefined,
        value >= 1e5
          ? { notation: "compact", maximumFractionDigits: 2 }
          : { maximumFractionDigits: 0 },
      );
const rate = (value) =>
  !Number.isFinite(value) || value < 0
    ? "—"
    : value >= 1
      ? value.toFixed(1)
      : Number(value.toPrecision(3)).toString();
// Matches the CLI progress line: 45s, 12m 05s, 1h 23m.
function duration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const pad = (value) => String(value).padStart(2, "0");
  if (total < 60) return `${total}s`;
  if (total < 3600) return `${Math.floor(total / 60)}m ${pad(total % 60)}s`;
  return `${Math.floor(total / 3600)}h ${pad(Math.floor((total % 3600) / 60))}m`;
}
function updateInitializationTuning(tuning, warmup) {
  const panel = $("initialization-tuning");
  const labels = {pending: "Preparing startup tuning", running: "Tuning startup", complete: "Startup tuning complete",
    failed: "Startup tuning failed"};
  panel.hidden = !labels[tuning?.status];
  if (panel.hidden) { panel.textContent = ""; return; }
  panel.dataset.status = tuning.status;
  panel.dataset.outcome = tuning.dynamics_outcome || "";
  const parts = [tuning.status === "complete" && tuning.dynamics_outcome === "unresolved"
    ? "Startup tuning unresolved" : labels[tuning.status]];
  const factor = value => Number.isFinite(value) && value > 0 ? Number(value.toPrecision(4)).toString() : null;
  if (tuning.status === "running") {
    if (tuning.phase === "initialization") parts.push("Initialization");
    else if (tuning.phase === "dynamics") parts.push("Trial updates");
  }
  if (tuning.status === "running" && Number.isSafeInteger(tuning.candidate) && tuning.candidate > 0
      && Number.isSafeInteger(tuning.total_candidates) && tuning.total_candidates >= tuning.candidate)
    parts.push(`Candidate ${tuning.candidate} of ${tuning.total_candidates}`);
  if (tuning.status === "running" && tuning.phase === "dynamics") {
    if (Number.isSafeInteger(tuning.trial_step) && tuning.trial_step >= 0
        && Number.isSafeInteger(tuning.trial_steps) && tuning.trial_steps > 0 && tuning.trial_step <= tuning.trial_steps)
      parts.push(`Trial step ${tuning.trial_step} of ${tuning.trial_steps}`);
    if (factor(tuning.lr_factor)) parts.push(`G learning rate × ${factor(tuning.lr_factor)}`);
  }
  if (tuning.status === "complete") {
    if (tuning.outcome === "kept_baseline") parts.push("Kept baseline initialization");
    else if (tuning.outcome === "selected") parts.push("Applied tuned initialization");
    if (tuning.dynamics_outcome === "selected" && factor(tuning.selected_g_lr_factor))
      parts.push(`Selected G learning rate × ${factor(tuning.selected_g_lr_factor)}`);
    else if (tuning.dynamics_outcome === "kept_baseline") parts.push("Configured G learning rate passed");
    else if (tuning.dynamics_outcome === "unresolved") parts.push("No rate candidate passed; kept configured G learning rate");
    else if (tuning.dynamics_outcome === "skipped") parts.push("Dynamics check skipped");
    if (typeof tuning.dynamics_reason === "string" && tuning.dynamics_reason.trim())
      parts.push(tuning.dynamics_reason.replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, 240));
  }
  if (typeof tuning.message === "string" && tuning.message.trim())
    parts.push(tuning.message.replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, 240));
  if (tuning.status === "complete" && Number.isSafeInteger(warmup?.steps) && warmup.steps >= 2
      && Number.isSafeInteger(warmup.completed_steps) && warmup.completed_steps >= 0
      && warmup.completed_steps <= warmup.steps) {
    parts.push(warmup.status === "complete" ? "G learning-rate warmup complete"
      : `G learning-rate warmup · Update ${warmup.completed_steps} of ${warmup.steps}`);
    if (factor(warmup.g_lr)) parts.push(`Current G learning rate ${factor(warmup.g_lr)}`);
    if (factor(warmup.target_g_lr)) parts.push(`Configured target ${factor(warmup.target_g_lr)} before annealing`);
  }
  panel.textContent = parts.join(" · ");
}
function updateRun(run) {
  state.run = run;
  evaluations.update(state.catalog, run);
  $("run-name").textContent =
    run.config?.name || run.name || "Training experiment";
  if ($("run-id").textContent !== run.run_id) {
    $("run-id").textContent = run.run_id;
    $("run-id-status").textContent = "";
  }
  $("run-status").textContent = statusLabel(run.status);
  $("run-status").dataset.status = run.status || "unknown";
  updateInitializationTuning(run.initialization_tuning, run.g_lr_warmup);
  $("step").textContent = fmt(run.steps);
  $("steps-per-second").textContent = rate(run.steps_per_second);
  $("training-time").textContent = duration(run.training_seconds);
  $("samples-seen").textContent = count(run.samples_seen);
  $("samples-seen").title = Number.isFinite(run.samples_seen)
    ? `${Number(run.samples_seen).toLocaleString()} samples`
    : "";
  $("samples-batch").textContent = Number.isFinite(run.global_batch_size)
    ? `${fmt(run.global_batch_size)} per completed update`
    : "Updates × global batch";
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
// Preview retention spans the whole run: it keeps the first sample and thins the
// older ones rather than dropping them, and --preview-keep all keeps every
// published sample. This matches the service's bound on the index.
const MAX_SAMPLE_VERSIONS = 4096;
const provenanceStep = (artifact) => {
  const step = artifact.provenance?.step;
  return Number.isFinite(step) ? step : -1;
};
const provenanceSequence = (artifact) => {
  const sequence = artifact.provenance?.sample_sequence;
  return Number.isFinite(sequence) ? sequence : -1;
};
const isFinalSample = (artifact) => artifact.provenance?.final === true;
function foldRawTensors(groups) {
  // The JSON tensor published with an image grid is the same sample as numbers.
  // It belongs on that image's card as a download, not in a second group; a
  // tensor with no picture (numerical recipes) and the final sample keep theirs.
  for (const [key, group] of [...groups]) {
    if (group.modality !== "tensor") continue;
    const images = groups.get(`${group.name}\u0000image`);
    if (!images) continue;
    const byStep = new Map(
      images.versions.map((version) => [provenanceStep(version.artifact), version]),
    );
    const remaining = [];
    for (const version of group.versions) {
      const image = isFinalSample(version.artifact)
        ? undefined
        : byStep.get(provenanceStep(version.artifact));
      if (image && !image.tensor) image.tensor = version;
      else remaining.push(version);
    }
    if (remaining.length) group.versions = remaining;
    else groups.delete(key);
  }
}
// A slider position is not an identity: samples arrive and old ones can be
// pruned, so every position shifts. Pin the sample itself by the step it was
// taken at (with its sequence to break ties within one step).
const versionKey = ({ artifact }) =>
  `${provenanceStep(artifact)}\u0000${provenanceSequence(artifact)}`;
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
  for (const group of groups.values())
    group.versions.sort(
      (a, b) =>
        provenanceStep(a.artifact) - provenanceStep(b.artifact) ||
        // The run's finished sample is the newest version of its step.
        Number(isFinalSample(a.artifact)) - Number(isFinalSample(b.artifact)) ||
        provenanceSequence(a.artifact) - provenanceSequence(b.artifact) ||
        (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );
  foldRawTensors(groups);
  for (const group of groups.values()) {
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
  // No pin means "follow the latest", so a newly published sample is shown.
  // A pin holds that exact sample; if it is gone, fall back to the latest.
  const pinnedIndex =
    pinned === undefined
      ? -1
      : group.versions.findIndex((version) => versionKey(version) === pinned);
  if (pinned !== undefined && pinnedIndex < 0) state.sampleVersions.delete(group.key);
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
      else state.sampleVersions.set(group.key, versionKey(group.versions[chosen]));
      show(chosen);
    };
    history.append(slider, position, latestButton);
    li.append(history);
  }
  function show(chosen) {
    index = chosen;
    const version = group.versions[index];
    const { artifact } = version;
    const step = artifact.provenance?.step;
    const shape = artifact.shape || artifact.metadata?.shape;
    details.textContent = [
      sampleKind(group, artifact),
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
    body.replaceChildren(sampleVersion(group, version));
  }
  show(index);
  return li;
}
function sampleKind(group, artifact) {
  // Say what the file is, not which internal role/modality pair produced it.
  if (group.modality === "image") return "Image grid";
  if (group.modality === "tensor" && artifact.role === "sample")
    return isFinalSample(artifact)
      ? "Final sample · raw generator output (JSON numbers)"
      : "Raw generator output (JSON numbers)";
  return [artifact.role || "Artifact", group.modality, artifact.media_type || "unknown type"]
    .join(" · ");
}
function sampleVersion(group, { id, artifact, tensor }) {
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
    if (tensor) {
      controls.append(rawTensorLink(tensor));
      fragment.append(
        note(
          "The raw tensor is the same sample as numbers: the values this grid was drawn from, for analysis outside the viewer.",
        ),
      );
    }
  }
  const previewable =
    artifact.modality === "tensor" &&
    artifact.media_type === "application/json" &&
    Number.isSafeInteger(artifact.bytes) &&
    artifact.bytes <= 8388608 &&
    Array.isArray(shape) &&
    shape.length > 0 &&
    shape.length <= 8 &&
    shape.every((n) => Number.isSafeInteger(n) && n > 0) &&
    shape.reduce((a, b) => a * b, 1) <= 262144;
  if (artifact.modality === "tensor")
    fragment.append(
      note(
        (artifact.role !== "sample"
          ? "Raw numbers as JSON. "
          : isFinalSample(artifact)
            ? "The sample saved when the run finished, as raw numbers straight from the generator. "
            : "The generator's output for this step as raw numbers, with no picture to draw from it. ") +
          (previewable
            ? '"Preview numbers" shows the first values; Download saves the whole tensor as JSON.'
            : "Numeric preview supports JSON tensors up to 8 MiB and 262,144 values. Download this tensor to inspect it."),
      ),
    );
  if (previewable) {
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
  fragment.append(controls);
  return fragment;
}
function note(text) {
  const paragraph = document.createElement("p");
  paragraph.className = "sample-note";
  paragraph.textContent = text;
  return paragraph;
}
function rawTensorLink({ id, artifact }) {
  const shape = artifact.shape || artifact.metadata?.shape;
  const link = document.createElement("a");
  link.href = `/api/v1${base()}/artifacts/${encodeURIComponent(id)}`;
  link.download = "samples.json";
  link.className = "text-link raw-tensor";
  link.textContent = Array.isArray(shape)
    ? `Download raw tensor (JSON, shape ${shape.join(" × ")})`
    : "Download raw tensor (JSON)";
  return link;
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
    "diversity/generated_rms",
    "diversity/ratio",
    "diversity/pooled4_ratio",
    "throughput/steps_per_second",
  ];
  return preferred.filter((id) => id in state.catalog.metrics).slice(0, 8);
}
function renderCatalog() {
  const search = $("search").value.toLowerCase();
  $("metric-count").textContent = Object.values(metricDefinitions()).filter(d => d.scope !== "snapshot" && d.kind === "scalar").length;
  $("metric-list").replaceChildren();
  for (const [id, definition] of Object.entries(metricDefinitions())) {
    if (definition.scope === "snapshot" || definition.kind !== "scalar") continue;
    if (!`${id} ${definition.label}`.toLowerCase().includes(search)) continue;
    const label = document.createElement("label");
    label.className = "metric-option";
    label.title = definition.description || "";
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
    small.textContent = id;
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
    [...state.selected].filter((id) => metricDefinitions()[id]?.kind === "scalar" && metricDefinitions()[id]?.scope !== "snapshot"),
  );
  renderCatalog();
  await refreshArtifacts();
}
async function connect() {
  connection("Connecting…");
  const capability = await api("/capabilities");
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
    $("coverage").textContent = "No metrics selected";
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
      description.textContent = metric;
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
    const alpha = Number($("smoothing").value),
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
        showSymbol: raw.filter(point => point[1] !== null).length === 1,
        symbolSize: 6,
        connectNulls: false,
        data: raw,
        lineStyle: { width: alpha ? 1 : 1.7, opacity: alpha ? 0.35 : 1 },
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
      for (const last of points.slice(-1)) {
        const row = document.createElement("tr");
        row.dataset.metric = metric;
        row.dataset.step = String(last.position[0]);
        for (const text of [
          metric,
          String(last.position[0]),
          String(last.value),
          part.attempt,
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
      omitted ? `${omitted} nonpositive points excluded from log scale.` : "",
      alpha
        ? "EMA uses visible envelope points; raw values remain visible."
        : "",
    ]
      .filter(Boolean)
      .join(" ");
    card.chart.setOption({ ...chartStyle(log), series }, true);
  }
  const metricOrder = [...state.selected];
  const rows = [...$("values-table").children].sort((a, b) =>
    metricOrder.indexOf(a.dataset.metric) - metricOrder.indexOf(b.dataset.metric) ||
    Number(a.dataset.step) - Number(b.dataset.step));
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
$("copy-run-id").addEventListener("click", async () => {
  const code = $("run-id");
  const value = code.textContent;
  if (!value) return;
  try {
    // Clipboard writes need a secure context, which plain-HTTP loopback is
    // not in every browser. Select the id instead so a keystroke copies it.
    if (!navigator.clipboard) throw new Error("no clipboard");
    await navigator.clipboard.writeText(value);
    $("run-id-status").textContent = "Copied";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    $("run-id-status").textContent = "Selected — press Ctrl+C to copy";
  }
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
