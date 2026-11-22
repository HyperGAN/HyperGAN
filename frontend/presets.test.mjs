import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildViewSearch,
  collectHealthAlerts,
  defaultMetrics,
  formatBatchPill,
  formatHostPill,
  formatTopologyPill,
  groupCatalogMetrics,
  parseViewQuery,
  presetMetrics,
  reducedSelection,
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

test("defaultMetrics returns 4 core series for a clean 2x2 grid and excludes throughput/snapshots", () => {
  const defaults = defaultMetrics(STANDARD_GAN_CATALOG);
  assert.equal(defaults.length, 4);
  assert.ok(!defaults.includes("throughput/steps_per_second"));
  assert.ok(!defaults.includes("fid50k_train"));
  assert.deepEqual(defaults, [
    "loss/g_total",
    "loss/d_total",
    "critic/accuracy",
    "diversity/ratio",
  ]);
});

test("defaultMetrics falls back to critic/margin and loss/gradient_penalty when primary slots are absent", () => {
  const withoutAccuracy = { ...STANDARD_GAN_CATALOG };
  delete withoutAccuracy["critic/accuracy"];
  assert.deepEqual(defaultMetrics(withoutAccuracy), [
    "loss/g_total",
    "loss/d_total",
    "critic/margin",
    "diversity/ratio",
  ]);

  const withoutDiversity = { ...STANDARD_GAN_CATALOG };
  delete withoutDiversity["diversity/ratio"];
  assert.deepEqual(defaultMetrics(withoutDiversity), [
    "loss/g_total",
    "loss/d_total",
    "critic/accuracy",
    "loss/gradient_penalty",
  ]);

  const minimalCatalog = {
    "loss/g_total": { kind: "scalar" },
    "loss/d_total": { kind: "scalar" },
    "loss/gradient_penalty": { kind: "scalar" },
  };
  assert.deepEqual(defaultMetrics(minimalCatalog), [
    "loss/g_total",
    "loss/d_total",
    "loss/gradient_penalty",
  ]);

  assert.deepEqual(defaultMetrics({}), []);
});

test("groupCatalogMetrics categorizes scalar metrics into deterministic ordered groups and omits empties", () => {
  const richCatalog = {
    "custom/extra": { kind: "scalar", label: "Custom extra" },
    "progress/epoch": { kind: "scalar", label: "Epoch" },
    "throughput/steps_per_second": { kind: "scalar", label: "Steps/s" },
    "timing/step_ms": { kind: "scalar", label: "Step ms" },
    "optimizer/lr": { kind: "scalar", label: "LR" },
    "critic/aux/real_mean": { kind: "scalar", label: "Aux real mean" },
    "critic/primary/margin": { kind: "scalar", label: "Primary margin" },
    "loss/critics/primary/d_total": { kind: "scalar", label: "Primary D total" },
    "loss/critics/aux/d_total": { kind: "scalar", label: "Aux D total" },
    "diversity/pooled4_ratio": { kind: "scalar", label: "Pooled4 ratio" },
    "diversity/generated_rms": { kind: "scalar", label: "Generated RMS" },
    "prior/sigma": { kind: "scalar", label: "Prior sigma" },
    "health/nan_count": { kind: "scalar", label: "NaN count" },
    "gradients/g_norm": { kind: "scalar", label: "G norm" },
    "grad_norm/generator": { kind: "scalar", label: "Generator grad norm" },
    "critic/real_mean": { kind: "scalar", label: "Critic real mean" },
    "loss/g_adversarial": { kind: "scalar", label: "G adversarial" },
    "loss/d_adversarial": { kind: "scalar", label: "D adversarial" },
    "loss/objectives/l1": { kind: "scalar", label: "L1 objective" },
    "loss/objectives/aux": { kind: "scalar", label: "Aux objective" },
    "diversity/ratio": { kind: "scalar", label: "Diversity ratio" },
    "critic/margin": { kind: "scalar", label: "Critic margin" },
    "critic/accuracy": { kind: "scalar", label: "Critic accuracy" },
    "loss/prior_regularizer": { kind: "scalar", label: "Prior regularizer" },
    "loss/gradient_penalty": { kind: "scalar", label: "Gradient penalty" },
    "loss/d_total": { kind: "scalar", label: "Discriminator total" },
    "loss/g_total": { kind: "scalar", label: "Generator total" },
    fid50k_train: { kind: "scalar", scope: "snapshot", label: "FID-50k" },
    "hist/weights": { kind: "histogram", label: "Weights" },
  };

  const groups = groupCatalogMetrics(richCatalog);
  assert.deepEqual(
    groups.map((g) => ({
      key: g.key,
      title: g.title,
      defaultOpen: g.defaultOpen,
      ids: g.metrics.map((m) => m.id),
    })),
    [
      {
        key: "core",
        title: "Core",
        defaultOpen: true,
        ids: [
          "loss/g_total",
          "loss/d_total",
          "loss/gradient_penalty",
          "loss/prior_regularizer",
          "critic/accuracy",
          "critic/margin",
          "diversity/ratio",
          "loss/objectives/aux",
          "loss/objectives/l1",
        ],
      },
      {
        key: "losses",
        title: "Losses",
        defaultOpen: false,
        ids: ["loss/d_adversarial", "loss/g_adversarial"],
      },
      {
        key: "diagnostics",
        title: "Critic & Gradients",
        defaultOpen: false,
        ids: [
          "critic/real_mean",
          "grad_norm/generator",
          "gradients/g_norm",
          "health/nan_count",
          "prior/sigma",
        ],
      },
      {
        key: "diversity",
        title: "Diversity",
        defaultOpen: false,
        ids: ["diversity/generated_rms", "diversity/pooled4_ratio"],
      },
      {
        key: "per_critic",
        title: "Per-critic",
        defaultOpen: false,
        ids: [
          "critic/aux/real_mean",
          "critic/primary/margin",
          "loss/critics/aux/d_total",
          "loss/critics/primary/d_total",
        ],
      },
      {
        key: "runtime",
        title: "Runtime & Progress",
        defaultOpen: false,
        ids: [
          "optimizer/lr",
          "timing/step_ms",
          "throughput/steps_per_second",
          "progress/epoch",
          "custom/extra",
        ],
      },
    ],
  );

  // Omits empty groups and handles reversed key order identically.
  const reversed = Object.fromEntries(Object.entries(richCatalog).reverse());
  assert.deepEqual(groupCatalogMetrics(reversed), groups);

  const sparse = groupCatalogMetrics({
    "loss/g_total": { kind: "scalar" },
    "throughput/steps_per_second": { kind: "scalar" },
  });
  assert.deepEqual(
    sparse.map((g) => g.key),
    ["core", "runtime"],
  );
  assert.deepEqual(groupCatalogMetrics(null), []);
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

  const multiCriticCatalog = {
    ...STANDARD_GAN_CATALOG,
    "loss/critics/primary/d_total": { kind: "scalar", label: "Discriminator total (primary)" },
    "loss/critics/aux/d_total": { kind: "scalar", label: "Discriminator total (aux)" },
    "critic/primary/margin": { kind: "scalar", label: "Critic score margin (primary) E[D(x)] - E[D(G(z))]" },
    "critic/aux/margin": { kind: "scalar", label: "Critic score margin (aux) E[D(x)] - E[D(G(z))]" },
  };
  const multiLosses = presetMetrics("losses", multiCriticCatalog);
  assert.ok(multiLosses.includes("loss/g_total"));
  assert.ok(multiLosses.includes("loss/d_total"));
  assert.ok(multiLosses.includes("loss/critics/primary/d_total"));
  assert.ok(multiLosses.includes("loss/critics/aux/d_total"));
  const multiHealth = presetMetrics("health", multiCriticCatalog);
  assert.ok(multiHealth.includes("critic/primary/margin"));
  assert.ok(multiHealth.includes("critic/aux/margin"));
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
