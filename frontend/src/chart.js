// One echarts registration and one chart style for the whole viewer, so the
// training curves and the snapshot evaluation charts are read the same way.
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
export const chartColors = [
  "#d7ff3f", // Signal
  "#aacbfa", // Sky
  "#d7bdf6", // Lilac
  "#f0b28a", // Semantic warning / additional data series
  "#f1f2ea", // Paper
  "#a0a69a", // Stone
];
const stepTooltip = (entries) =>
  entries.length
    ? `Step ${entries[0].value[0]}\n` +
      entries.map((p) => `${p.seriesName}: ${String(p.data.rawValue ?? p.value[1])}`).join("\n")
    : "";
// Everything except the series. Every y axis includes zero without hiding
// negative observations.
export function chartStyle(formatter) {
  return {
    animation: false,
    color: chartColors,
    grid: { left: 54, right: 20, top: 25, bottom: 36 },
    textStyle: { fontFamily: '"IBM Plex Mono", monospace' },
    tooltip: {
      trigger: "axis",
      renderMode: "richText",
      backgroundColor: "#22251f",
      borderColor: "#7c8471",
      textStyle: { color: "#f1f2ea", fontSize: 10 },
      formatter: formatter || stepTooltip,
    },
    xAxis: {
      type: "value",
      axisLabel: { color: "#a0a69a", fontSize: 9 },
      axisLine: { lineStyle: { color: "#35392f" } },
      splitLine: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      scale: false,
      axisLabel: { color: "#a0a69a", fontSize: 9 },
      splitLine: { lineStyle: { color: "#35392f" } },
      axisLine: { show: false },
    },
  };
}

// Signed log1p is linear near zero and logarithmic farther away. Transform
// presentation coordinates only; tooltips and EMA retain the original values.
export function chartOptions(series, symlog = false) {
  const options = chartStyle();
  if (!symlog) return { ...options, series };
  options.yAxis.axisLabel.formatter = (value) => {
    const original = Math.sign(value) * Math.expm1(Math.abs(value));
    return Number.isFinite(original) ? String(Number(original.toPrecision(4))) : "";
  };
  return {
    ...options,
    series: series.map((item) => ({
      ...item,
      data: item.data.map(([step, value]) => ({
        value: [step, value === null ? null : Math.sign(value) * Math.log1p(Math.abs(value))],
        rawValue: value,
      })),
    })),
  };
}
export { init };
