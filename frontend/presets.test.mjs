import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildViewSearch,
  collectHealthAlerts,
  defaultMetrics,
  formatBatchPill,
  formatHostPill,
  formatTopologyPill,
  parseViewQuery,
  presetMetrics,
  reducedSelection,
  stepMultiplier,
} from "./src/presets.js";

const STANDARD_GAN_CATALOG = {
  "loss/g_total": { kind: "scalar", label: "Generator total" },
  "loss/d_total": { kind: "scalar", label: "Discriminator total" },
  "loss/d_adversarial": { kind: "scalar", label: "D adversarial" },
  "loss/g_adversarial": { kind: "scalar", label: "G adversarial" },
  "loss/gradient_penalty": { kind: "scalar", label: "Gradient penalty" },
  "loss/prior_regularizer": { kind: "scalar", label: "Prior regularizer" },
  "critic/accuracy": { kind: "scalar", label: "Critic accuracy" },
  "critic/margin": { kind: "scalar", label: "Critic margin" },
  "critic/real_mean": { kind: "scalar", label: "Critic real mean" },
  "critic/fake_mean": { kind: "scalar", label: "Critic fake mean" },
  "grad_norm/critic": { kind: "scalar", label: "Critic grad norm" },
  "grad_norm/generator": { kind: "scalar", label: "Generator grad norm" },
  "grad_norm/prior": { kind: "scalar", label: "Prior grad norm" },
  "diversity/generated_rms": { kind: "scalar", label: "Generated RMS" },
  "diversity/reference_rms": { kind: "scalar", label: "Reference RMS" },
  "diversity/ratio": { kind: "scalar", label: "Diversity ratio" },
  "diversity/pooled4_generated_rms": { kind: "scalar", label: "Pooled4 generated RMS" },
  "diversity/pooled4_reference_rms": { kind: "scalar", label: "Pooled4 reference RMS" },
  "diversity/pooled4_ratio": { kind: "scalar", label: "Pooled4 ratio" },
  "throughput/steps_per_second": { kind: "scalar", label: "Steps per second" },
  fid50k_train: { kind: "scalar", scope: "snapshot", label: "FID-50k" },
};

test("defaultMetrics returns 8 balanced series, includes critic/accuracy, and excludes throughput", () => {
  const defaults = defaultMetrics(STANDARD_GAN_CATALOG);
  assert.equal(defaults.length, 8);
  assert.ok(!defaults.includes("throughput/steps_per_second"));
  assert.ok(!defaults.includes("fid50k_train"));
  assert.deepEqual(defaults, [
    "loss/g_total",
    "loss/d_total",
    "loss/gradient_penalty",
    "loss/prior_regularizer",
    "critic/accuracy",
    "diversity/generated_rms",
    "diversity/ratio",
    "diversity/pooled4_ratio",
  ]);
});

test("defaultMetrics falls back to critic/margin when critic/accuracy is absent", () => {
  const withoutAccuracy = { ...STANDARD_GAN_CATALOG };
  delete withoutAccuracy["critic/accuracy"];
  const defaults = defaultMetrics(withoutAccuracy);
  assert.equal(defaults.length, 8);
  assert.ok(defaults.includes("critic/margin"));
  assert.ok(!defaults.includes("throughput/steps_per_second"));
  assert.equal(defaults.filter((id) => id.startsWith("diversity/")).length, 3);
});

test("presetMetrics covers losses, health, and diversity bounded to 8 scalar training metrics", () => {
  const losses = presetMetrics("losses", STANDARD_GAN_CATALOG);
  assert.equal(losses.length, 8);
  assert.ok(losses.every((id) => id.startsWith("loss/") || id.startsWith("critic/")));
  assert.ok(losses.includes("loss/g_total") && losses.includes("loss/d_total"));

  const healthReal = presetMetrics("health", STANDARD_GAN_CATALOG);
  assert.equal(healthReal.length, 8);
  assert.ok(healthReal.includes("grad_norm/generator"));
  assert.ok(healthReal.includes("grad_norm/critic"));
  assert.ok(healthReal.includes("loss/gradient_penalty"));
  assert.ok(healthReal.includes("critic/accuracy"));

  const syntheticHealthCatalog = {
    "gradients/g_norm": { kind: "scalar" },
    "gradients/d_norm": { kind: "scalar" },
    "health/nan_count": { kind: "scalar" },
    "critic/margin": { kind: "scalar" },
    "loss/gradient_penalty": { kind: "scalar" },
    "throughput/steps_per_second": { kind: "scalar" },
  };
  const healthSynthetic = presetMetrics("health", syntheticHealthCatalog);
  assert.deepEqual(healthSynthetic, [
    "gradients/g_norm",
    "gradients/d_norm",
    "health/nan_count",
    "loss/gradient_penalty",
    "critic/margin",
  ]);

  const diversity = presetMetrics("diversity", STANDARD_GAN_CATALOG);
  assert.equal(diversity.length, 6);
  assert.ok(diversity.every((id) => id.startsWith("diversity/")));
});

test("reducedSelection pins loss/g_total and loss/d_total when selected is non-empty and returns [] when empty", () => {
  assert.deepEqual(reducedSelection(new Set(), STANDARD_GAN_CATALOG), []);
  assert.deepEqual(reducedSelection([], STANDARD_GAN_CATALOG), []);

  const diversityOnly = new Set(["diversity/ratio", "diversity/pooled4_ratio"]);
  const reduced = reducedSelection(diversityOnly, STANDARD_GAN_CATALOG);
  assert.deepEqual(reduced, [
    "diversity/ratio",
    "diversity/pooled4_ratio",
    "loss/g_total",
    "loss/d_total",
  ]);

  // Does not pin metrics absent from catalog or marked snapshot
  const partialCatalog = {
    "diversity/ratio": { kind: "scalar" },
    "loss/g_total": { kind: "scalar" },
    "loss/d_total": { kind: "scalar", scope: "snapshot" },
  };
  assert.deepEqual(
    reducedSelection(new Set(["diversity/ratio"]), partialCatalog),
    ["diversity/ratio", "loss/g_total"],
  );
});

test("parseViewQuery and buildViewSearch persist non-default state and keep default URLs clean", () => {
  const initial = parseViewQuery("", STANDARD_GAN_CATALOG);
  assert.deepEqual(initial.selected, defaultMetrics(STANDARD_GAN_CATALOG));
  assert.equal(initial.scale, "linear");
  assert.equal(initial.smooth, "0");
  assert.equal(initial.stepFrom, null);
  assert.equal(initial.stepTo, null);
  assert.equal(buildViewSearch(initial, STANDARD_GAN_CATALOG), "");

  const cleared = parseViewQuery("?metrics=", STANDARD_GAN_CATALOG);
  assert.deepEqual(cleared.selected, []);
  assert.equal(buildViewSearch(cleared, STANDARD_GAN_CATALOG), "?metrics=");

  const custom = parseViewQuery(
    "?metrics=diversity/ratio,fid50k_train,unknown/x&scale=symlog&smooth=0.2&from=100&to=500",
    STANDARD_GAN_CATALOG,
  );
  assert.deepEqual(custom, {
    selected: ["diversity/ratio"],
    scale: "symlog",
    smooth: "0.2",
    stepFrom: 100,
    stepTo: 500,
  });
  assert.equal(
    buildViewSearch(custom, STANDARD_GAN_CATALOG),
    "?metrics=diversity%2Fratio&scale=symlog&smooth=0.2&from=100&to=500",
  );
});

test("header pill and health alert helpers deduplicate topology and filter noisy tracebacks", () => {
  assert.equal(
    formatTopologyPill({ label: "2 GPUs", detail: "2× RTX A6000" }),
    "2 GPUs · 2× RTX A6000",
  );
  assert.equal(formatTopologyPill({ label: "CPU", detail: null }), "CPU");
  assert.equal(formatTopologyPill({ label: "CPU", detail: "CPU" }), "CPU");
  assert.equal(
    formatBatchPill({ batch_label: "Batch 256", batch_detail: "128 / GPU" }),
    "Batch 256 · 128 / GPU",
  );
  assert.equal(
    formatHostPill({ hostname: "dgx-01", remote: true }),
    "Remote · dgx-01",
  );

  const noisyCompile =
    "torch.compile dynamo warning with memory reference: Traceback (most recent call last): " +
    'File "/opt/torch/_inductor/compile_fx.py", line 1024, in compile_fx ' +
    "x".repeat(220);
  const alerts = collectHealthAlerts({
    health_alerts: ["Mode collapse risk: diversity/ratio dropped to 0.12"],
    warnings: [
      "Snapshot evaluator fid50k_train is sharing device cuda:0 with training (GPU contention)",
      "Unrelated benign configuration note",
      noisyCompile,
    ],
  });
  assert.deepEqual(alerts, [
    "Mode collapse risk: diversity/ratio dropped to 0.12",
    "Snapshot evaluator fid50k_train is sharing device cuda:0 with training (GPU contention)",
  ]);
});

test("stepMultiplier scales multi-GPU/multi-rank runs by world_size and leaves single-GPU/CPU at 1", () => {
  assert.equal(stepMultiplier(null), 1);
  assert.equal(stepMultiplier({}), 1);
  assert.equal(stepMultiplier({ topology: { device_kind: "cpu", world_size: 1 } }), 1);
  assert.equal(stepMultiplier({ topology: { device_kind: "cuda", world_size: 1, gpu_count: 1 } }), 1);
  assert.equal(stepMultiplier({ step_multiplier: 2, topology: { world_size: 2, gpu_count: 2 } }), 2);
  assert.equal(stepMultiplier({ topology: { world_size: 8, gpu_count: 8 } }), 8);
  assert.equal(stepMultiplier({ world_size: 4 }), 4);
});
