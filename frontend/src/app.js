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
  stepFrom: null,
  stepTo: null,
  refreshing: false,
  bootstrapSignal: false,
  coarsenAttempts: 0,
};
let requestID = 0;
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
      data.error ? p.reject(new Error(data.error)) : p.resolve(data.ok);
    };
    this.worker.onerror = (e) =>
      this.close(new Error(e.message || "View worker failed"));
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
const fmt = (value) =>
  value === null || value === undefined
    ? "—"
    : Number(value).toLocaleString(undefined, { maximumSignificantDigits: 6 });
function updateRun(run) {
  state.run = run;
  $("run-name").textContent =
    run.config?.name || run.name || "Training experiment";
  $("run-id").textContent = run.run_id;
  $("run-status").textContent = run.status || "Unknown";
  $("step").textContent = fmt(run.steps);
  $("durable").textContent = fmt(run.last_durable_step);
  $("total-steps").textContent = run.total_steps
    ? `of ${fmt(run.total_steps)} configured steps`
    : "Completed optimizer updates";
  $("raw-events").href = `/api/v1${base()}/events`;
  renderArtifacts(run.artifacts || []);
}
function renderArtifacts(artifacts) {
  $("artifact-items").replaceChildren();
  $("artifacts").hidden = !artifacts.length;
  for (const artifact of artifacts.slice(0, 20)) {
    const li = document.createElement("li");
    li.textContent = `${artifact.role || "artifact"} · ${artifact.modality || "unspecified"} · ${artifact.media_type || "unknown type"} · ${fmt(artifact.size_bytes)} bytes`; // Never interpret unknown tensors as images or HTML.
    $("artifact-items").append(li);
  }
}
function defaults() {
  const preferred = [
    "loss/g_total",
    "loss/d_total",
    "loss/gradient_penalty",
    "loss/prior_regularizer",
  ];
  return preferred.filter((id) => id in state.catalog.metrics).slice(0, 8);
}
function renderCatalog() {
  const search = $("search").value.toLowerCase();
  $("metric-count").textContent = Object.keys(state.catalog.metrics).length;
  $("metric-list").replaceChildren();
  for (const [id, definition] of Object.entries(state.catalog.metrics)) {
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
  state.map = views.data.map_revision;
  state.selected = new Set(
    [...state.selected].filter((id) => id in state.catalog.metrics),
  );
  renderCatalog();
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
    series: [...state.selected].sort().join(","),
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
  if (!state.selected.size) {
    $("coverage").textContent = "No metrics selected";
    connection("No selection");
    return;
  }
  openStream(epoch);
  await loadBootstrap(epoch);
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
      selected: [...state.selected],
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
function scheduleReconnect(epoch) {
  if (epoch !== state.epoch || state.retry) return;
  const delay = Math.min(1000 * 2 ** state.retries++, 15000);
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
  state.stream = stream;
  stream.onopen = () => {
    if (epoch === state.epoch && state.ready) connection("Live stream", "live");
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
        if (state.ready) connection("Live stream", "live");
        else if (name === "ready") loadBootstrap(epoch);
      } catch (error) {
        notice(error.message);
      }
    });
  stream.addEventListener("bootstrap_ready", () => {
    if (state.ready) return;
    if (state.pending) state.bootstrapSignal = true;
    else loadBootstrap(epoch);
  });
  stream.addEventListener("metadata", () => refreshMetadata(epoch));
  for (const name of ["reset_required", "gap"])
    stream.addEventListener(name, (event) => {
      if (epoch !== state.epoch) return;
      stream.close();
      notice(
        name === "gap"
          ? "The stream paused because this browser fell behind. Reloading coverage."
          : "Run history changed. Reloading the current view.",
      );
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
      scheduleReconnect(epoch);
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
      title.textContent = state.catalog.metrics[metric]?.label || metric;
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
        showSymbol: false,
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
          showSymbol: false,
          data: smooth,
          lineStyle: { width: 2 },
          itemStyle: { color: colors[index % colors.length] },
          animation: false,
        });
      const last = points.at(-1);
      if (last) {
        const row = document.createElement("tr");
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
