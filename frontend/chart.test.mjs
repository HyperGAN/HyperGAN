import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

// Exercise the actual chart engine without a browser or image comparisons.
const bundle = await build({
  stdin: {
    contents: `export * from './src/chart.js';
      import { use } from 'echarts/core';
      import { SVGRenderer } from 'echarts/renderers';
      use([SVGRenderer]);`,
    resolveDir: fileURLToPath(new URL("./", import.meta.url)),
  },
  bundle: true, platform: "node", format: "esm", write: false,
});
const { chartOptions, init } = await import(
  "data:text/javascript;base64," + Buffer.from(bundle.outputFiles[0].contents).toString("base64")
);

for (const symlog of [false, true]) {
  test(`${symlog ? "symmetric log" : "linear"} includes zero and all observations`, () => {
    const chart = init(null, null, { renderer: "svg", ssr: true, width: 640, height: 300 });
    try {
      for (const values of [[98, 100], [0.001, 0.002], [5], [0, 0], [-5, -2], [-100, 0, 100]]) {
        const options = chartOptions([{ type: "line", data: values.map((y, x) => [x, y]) }], symlog);
        chart.setOption(options, true);
        const [min, max] = chart.getModel().getComponent("yAxis").axis.scale.getExtent();
        const plotted = options.series[0].data.map(p => (p.value || p)[1]);
        assert(min <= Math.min(0, ...plotted));
        assert(max >= Math.max(0, ...plotted));
        if (values.every(y => y >= 0)) assert.equal(min, 0);
        if (values.every(y => y <= 0) && values.some(y => y < 0)) assert.equal(max, 0);
      }
    } finally { chart.dispose(); }
  });
}

test("symmetric log preserves exact tooltip values, gaps, and original tick units", () => {
  const values = [-100, -0.123456789012345, 0, 0.0001, 100, null];
  const series = [{ type: "line", data: values.map((y, x) => [x, y]) }];
  const before = structuredClone(series);
  const options = chartOptions(series, true);
  options.series[0].data.forEach((data, index) => {
    const value = values[index];
    assert.equal(data.rawValue, value);
    assert.equal(options.tooltip.formatter([{ seriesName: "Metric", value: data.value, data }]),
      `Step ${index}\nMetric: ${String(value)}`);
    if (value === null) assert.equal(data.value[1], null);
    else {
      const label = Number(options.yAxis.axisLabel.formatter(data.value[1]));
      assert(Math.abs(label - value) <= Math.abs(value) * 0.001);
    }
  });
  assert.deepEqual(series, before);
  assert.deepEqual(chartOptions(series).series, before);
});
