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
  "#d7bb81",
  "#8fbdaa",
  "#b3a5d2",
  "#d69478",
  "#85b0cb",
  "#c5b38f",
];
const stepTooltip = (entries) =>
  entries.length
    ? `Step ${entries[0].value[0]}\n` +
      entries.map((p) => `${p.seriesName}: ${String(p.value[1])}`).join("\n")
    : "";
// Everything except the series. Every y axis includes zero without hiding
// negative observations.
export function chartStyle(formatter) {
  return {
    animation: false,
    color: chartColors,
    grid: { left: 54, right: 20, top: 25, bottom: 36 },
    textStyle: { fontFamily: "system-ui" },
    tooltip: {
      trigger: "axis",
      renderMode: "richText",
      backgroundColor: "#28352e",
      borderColor: "#526157",
      textStyle: { color: "#e8e8df", fontSize: 10 },
      formatter: formatter || stepTooltip,
    },
    xAxis: {
      type: "value",
      axisLabel: { color: "#85978b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#38483d" } },
      splitLine: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      scale: false,
      axisLabel: { color: "#85978b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#2b3930" } },
      axisLine: { show: false },
    },
  };
}
export { init };
