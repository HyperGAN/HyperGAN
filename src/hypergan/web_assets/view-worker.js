/* View state and delivery coverage belong here; mathematics stays in core WASM. */
import { Reducer } from "/reducers/host.js";
const ready = Reducer.load(new URL("/reducers/", self.location));
const MAX_GROUPS = 2048;
let view = null;
const integer = (n, name) => {
  if (!Number.isSafeInteger(n) || n < 0) throw new Error(`Invalid ${name}`);
  return n;
};
const text = (s, name) => {
  if (typeof s !== "string" || !s.length || s.length > 256)
    throw new Error(`Invalid ${name}`);
  return s;
};
const hash = (s) => typeof s === "string" && /^[a-f0-9]{64}$/.test(s);
function key(value) {
  if (!Array.isArray(value) || value.length !== 4 || !hash(value[1]))
    throw new Error("Invalid view group key");
  text(value[0], "metric ID");
  text(value[2], "attempt ID");
  integer(value[3], "bucket");
  return JSON.stringify(value);
}
function shown(attempt, step) {
  if (!view.lineage.has(attempt)) return false;
  const upper = view.lineage.get(attempt);
  return (
    (upper === null || step <= upper) &&
    (view.stepFrom === null || step >= view.stepFrom) &&
    (view.stepTo === null || step <= view.stepTo)
  );
}
function output(reducer, groups) {
  return [...groups.values()].map((group) => ({
    key: group.key,
    value: reducer.finalize(group.state),
  }));
}
self.onmessage = async ({ data }) => {
  try {
    const reducer = await ready;
    let result;
    if (data.op === "bootstrap") {
      const b = data.bootstrap;
      if (b.schema_version !== 1 || b.module_sha256 !== reducer.spec.sha256)
        throw new Error("Incompatible bootstrap reducer");
      if (
        !hash(b.map_revision) ||
        !hash(b.view_revision) ||
        !hash(b.lineage_revision)
      )
        throw new Error("Invalid bootstrap revision");
      integer(b.projection_sequence, "projection sequence");
      integer(b.bucket_steps, "bucket size");
      if (
        b.bucket_steps === 0 ||
        !Array.isArray(b.groups) ||
        b.groups.length > MAX_GROUPS
      )
        throw new Error("View capacity exceeded; choose wider buckets");
      text(b.run_id, "run identity");
      if (typeof b.cursor !== "string" || b.cursor.length > 4096)
        throw new Error("Invalid bootstrap cursor");
      if (!Array.isArray(b.lineage) || b.lineage.length > 4096)
        throw new Error("Invalid lineage");
      const lineage = new Map();
      for (const item of b.lineage) {
        text(item.attempt_id, "attempt");
        if (item.through_step !== null)
          integer(item.through_step, "lineage step");
        if (lineage.has(item.attempt_id))
          throw new Error("Duplicate lineage attempt");
        lineage.set(item.attempt_id, item.through_step);
      }
      const groups = new Map();
      const selected = new Set(data.selected);
      if (selected.size > 32) throw new Error("Select at most 32 metrics");
      const stepFrom = data.stepFrom ?? null,
        stepTo = data.stepTo ?? null;
      if (stepFrom !== null) integer(stepFrom, "range start");
      if (stepTo !== null) integer(stepTo, "range end");
      for (const group of b.groups) {
        const id = key(group.key);
        if (
          groups.has(id) ||
          group.state.reducer !== "envelope/v1" ||
          !lineage.has(group.key[2]) ||
          !selected.has(group.key[0]) ||
          group.key[3] % b.bucket_steps !== 0
        )
          throw new Error("Invalid bootstrap group");
        reducer.finalize(group.state);
        const upper = lineage.get(group.key[2]);
        for (const point of [
          group.state.first,
          group.state.min,
          group.state.max,
          group.state.last,
        ]) {
          if (!point) continue;
          const step = point.position[0];
          if (
            !hash(point.position[1]) ||
            Math.floor(step / b.bucket_steps) * b.bucket_steps !==
              group.key[3] ||
            (upper !== null && step > upper) ||
            (stepFrom !== null && step < stepFrom) ||
            (stepTo !== null && step > stepTo)
          )
            throw new Error(
              "Bootstrap point is outside its declared partition",
            );
        }
        groups.set(id, group);
      }
      view = { ...b, groups, lineage, selected, stepFrom, stepTo };
      result = {
        groups: output(reducer, groups),
        cursor: view.cursor,
        projection_sequence: view.projection_sequence,
      };
    } else if (data.op === "frame") {
      if (!view) throw new Error("Bootstrap required");
      const { frame, cursor, stream_id } = data.envelope;
      if (
        stream_id !== `projection:${view.map_revision}` ||
        frame.map_revision !== view.map_revision ||
        frame.source.run_id !== view.run_id
      )
        throw new Error("Stream identity changed; bootstrap required");
      integer(frame.projection_sequence, "projection sequence");
      if (frame.projection_sequence <= view.projection_sequence) {
        result = {
          groups: [],
          cursor: view.cursor,
          projection_sequence: view.projection_sequence,
          replay: true,
        };
      } else {
        if (frame.projection_sequence !== view.projection_sequence + 1)
          throw new Error("Coverage gap; bootstrap required");
        if (
          typeof cursor !== "string" ||
          cursor.length > 4096 ||
          !Array.isArray(frame.emissions) ||
          frame.emissions.length > 128
        )
          throw new Error("Invalid projection frame");
        // Stage all touched states. A failure never advances any state or cursor.
        const staged = new Map();
        let latestStep = null;
        for (const e of frame.emissions) {
          if (
            !Array.isArray(e.key) ||
            e.key.length !== 3 ||
            !hash(e.id) ||
            !hash(e.definition_hash) ||
            typeof e.value !== "number" ||
            !Number.isFinite(e.value)
          )
            throw new Error("Invalid contribution");
          const [metric, attempt, step] = e.key;
          integer(step, "step");
          text(metric, "metric");
          text(attempt, "attempt");
          if (attempt !== frame.source.attempt_id)
            throw new Error("Contribution attempt differs from source");
          if (!view.selected.has(metric) || !shown(attempt, step)) continue;
          latestStep = Math.max(latestStep ?? 0, step);
          const groupKey = [
            metric,
            e.definition_hash,
            attempt,
            Math.floor(step / view.bucket_steps) * view.bucket_steps,
          ];
          const id = key(groupKey),
            old = staged.get(id) || view.groups.get(id);
          const state = reducer.add(
            old ? old.state : reducer.identity("envelope/v1"),
            [{ value: e.value, position: [step, e.id] }],
          );
          staged.set(id, { key: groupKey, state });
        }
        let added = 0;
        for (const id of staged.keys()) if (!view.groups.has(id)) added++;
        if (view.groups.size + added > MAX_GROUPS)
          throw new Error("View capacity exceeded; choose wider buckets");
        const groups = output(reducer, staged);
        for (const [id, group] of staged) view.groups.set(id, group);
        view.cursor = cursor;
        view.projection_sequence = frame.projection_sequence;
        result = {
          groups,
          cursor,
          projection_sequence: view.projection_sequence,
          latestStep,
        };
      }
    } else throw new Error("Unknown view worker operation");
    self.postMessage({ id: data.id, ok: result });
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error.message || error) });
  }
};
