// Immutable snapshot results use their own source catalog and position. Loading
// is serialized, bounded to the server's 64 streams, and never reduces history.
// Completed scalar results are grouped by metric across streams and drawn as one
// chart per metric, source step on the x axis, one visible point per evaluation.
import { chartStyle, chartColors, init } from './chart.js';
const node = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
};
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
// Run-owned strings reach table cells and status lines; keep them one-line and short.
const brief = (value, limit = 240) => {
  const text = value === undefined || value === null ? '' : String(value).replace(/\s+/g, ' ').trim();
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
};
const svgNode = (tag, attrs, text) => {
  const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) result.setAttribute(key, value);
  if (text !== undefined) result.textContent = text;
  return result;
};
function histogramPlot(value, label, unit) {
  const svg = svgNode('svg', {viewBox: '0 0 600 180', role: 'img', 'aria-label': label, class: 'evaluation-plot'});
  svg.append(svgNode('title', {}, label));
  // Scale before subtracting to avoid overflow even for finite extreme edges.
  const scale = Math.max(1, ...value.edges.map(Math.abs));
  const edges = value.edges.map(x => x / scale);
  const extent = edges.at(-1) - edges[0];
  if (!finite(extent) || extent <= 0) throw new Error('Histogram range cannot be plotted');
  const max = Math.max(...value.counts) || 1;
  value.counts.forEach((count, i) => {
    const x = 35 + 530 * ((edges[i] - edges[0]) / extent);
    const width = 530 * ((edges[i+1] - edges[i]) / extent);
    const height = 125 * (count / max);
    const bar = svgNode('rect', {x, y: 145 - height, width, height});
    bar.append(svgNode('title', {}, `${value.edges[i]} to ${value.edges[i+1]}: ${count}`));
    svg.append(bar);
  });
  svg.append(svgNode('text', {x: 35, y: 170}, String(value.edges[0])));
  svg.append(svgNode('text', {x: 565, y: 170, 'text-anchor': 'end'}, String(value.edges.at(-1))));
  svg.append(svgNode('text', {x: 35, y: 14}, `Maximum: ${Math.max(...value.counts)} ${unit || ''}`));
  return svg;
}
function validateValue(value, kind) {
  if (kind === 'scalar' && finite(value)) return;
  if (kind !== 'histogram' || !value || !Array.isArray(value.edges) || !Array.isArray(value.counts)
      || value.counts.length < 1 || value.counts.length > 512 || value.edges.length !== value.counts.length + 1
      || !value.edges.every(finite) || !value.counts.every(x => finite(x) && x >= 0)
      || !value.edges.every((x, i) => !i || x > value.edges[i-1])) throw new Error('Invalid evaluation value');
}
// One terminal document per evaluation: reject anything else before it is plotted.
function readResult(event, catalog, path) {
  const ids = [...new Set([...Object.keys(event.metrics || {}), ...Object.keys(event.distributions || {}), ...Object.keys(event.measurement_status || {})])];
  if (ids.length !== 1) throw new Error('Expected one metric per evaluation');
  const id = ids[0], definition = catalog.metrics[id];
  if (!definition) throw new Error('Evaluation definition is missing from its source catalog');
  const failed = event.status === 'failed', cancelled = event.status === 'cancelled';
  if (!failed && !cancelled && event.status !== 'complete') throw new Error('Evaluation is not terminal');
  const known = event.source_position_known === true;
  const record = {id, definition, event, catalog, path, known,
    status: cancelled ? 'cancelled' : failed ? 'failed' : 'complete', value: null};
  if (failed || cancelled) return record;
  if (!known || !Number.isSafeInteger(event.step) || event.step < 0) throw new Error('Complete evaluation needs a known source step');
  if (!/^[0-9a-f]{64}$/.test(definition.definition_hash || '') || !/^[0-9a-f]{64}$/.test(event.protocol_sha256 || '')) throw new Error('Invalid evaluation definition or protocol identity');
  record.value = definition.kind === 'histogram' ? event.distributions?.[id] : event.metrics?.[id];
  validateValue(record.value, definition.kind);
  return record;
}
const stepOf = (record) => record.known && Number.isSafeInteger(record.event.step) ? record.event.step : Infinity;
const byStep = (a, b) => stepOf(a) - stepOf(b) || (a.event.evaluation_id || '').localeCompare(b.event.evaluation_id || '');
const protocolOf = (event) => (event.evaluation_protocol && typeof event.evaluation_protocol === 'object') ? event.evaluation_protocol : {};
const evaluationSpec = (event) => { const spec = protocolOf(event).evaluation; return spec && typeof spec === 'object' ? spec : {}; };
const deviceOf = (event) => evaluationSpec(event).device ?? protocolOf(event).runtime?.device;
const samplesOf = (event) => evaluationSpec(event).sample_count ?? protocolOf(event).sample_count;
const statusLabels = {complete: 'Complete', failed: 'Failed', cancelled: 'Cancelled'};
const positionOf = (record) => record.known ? `Source step ${record.event.step}` : 'Source position unknown';
// Discrete measurements: one series per definition and protocol, never averaged,
// with the marker always drawn so a single result is a visible point.
function chartSeries(records) {
  const parts = new Map();
  for (const record of records) {
    if (record.status !== 'complete' || record.definition.kind !== 'scalar') continue;
    const key = `${record.definition.definition_hash}:${record.event.protocol_sha256}`;
    if (!parts.has(key)) parts.set(key, {
      name: `${record.definition.definition_hash.slice(0, 6)} · ${record.event.protocol_sha256.slice(0, 8)}`, data: []});
    parts.get(key).data.push([record.event.step, record.value]);
  }
  return [...parts.values()].map((part, index) => ({
    name: part.name, type: 'line', showSymbol: true, symbolSize: 6, connectNulls: false,
    data: part.data, lineStyle: {width: 1.7}, animation: false,
    itemStyle: {color: chartColors[index % chartColors.length]},
  }));
}
function resultsTable(label, records) {
  const table = node('table');
  table.append(node('caption', `${label} · every evaluation by source step`));
  const head = node('thead'), header = node('tr');
  for (const title of ['Source step', 'Value', 'Status', 'Seconds', 'Device', 'Samples', 'Attempt', 'Evaluation']) {
    const cell = node('th', title); cell.scope = 'col'; header.append(cell);
  }
  head.append(header); table.append(head);
  const body = node('tbody');
  for (const record of records) {
    const {event} = record;
    const row = node('tr');
    row.dataset.step = record.known ? String(event.step) : '';
    row.dataset.evaluation = event.evaluation_id;
    const value = record.status !== 'complete' ? brief(event.measurement_status?.[record.id]?.reason || statusLabels[record.status])
      : finite(record.value) ? `${record.value} ${record.definition.unit || ''}`.trim()
      : `${record.value.counts.length} histogram bins`;
    for (const text of [record.known ? String(event.step) : 'unknown', value, statusLabels[record.status],
      finite(event.seconds) ? String(event.seconds) : 'unknown', brief(deviceOf(event) ?? 'unrecorded', 64),
      brief(samplesOf(event) ?? 'unrecorded', 64), brief(event.attempt_id, 64),
      brief(event.stream_generation || event.evaluation_id, 64)]) row.append(node('td', text));
    body.append(row);
  }
  table.append(body);
  const scroll = node('div', undefined, 'table-scroll'); scroll.tabIndex = 0; scroll.append(table);
  return scroll;
}
function binTable(value, label) {
  const table = node('table');
  table.append(node('caption', `${label} · exact bin values`));
  const head = node('thead'), header = node('tr');
  for (const title of ['Bin start', 'Bin end', 'Value']) { const cell = node('th', title); cell.scope = 'col'; header.append(cell); }
  head.append(header); table.append(head);
  const body = node('tbody');
  value.counts.forEach((count, i) => {
    const row = node('tr');
    for (const cell of [value.edges[i], value.edges[i+1], count]) row.append(node('td', String(cell)));
    body.append(row);
  });
  table.append(body);
  const scroll = node('div', undefined, 'table-scroll'); scroll.tabIndex = 0; scroll.append(table);
  return scroll;
}
// Built only while the metric's details element is open; the protocol document
// of each result stays behind one more explicit expansion.
function resultBlocks(label, records) {
  return records.map(record => {
    const {event, definition} = record;
    const block = node('details', undefined, 'evaluation-result');
    block.dataset.evaluation = event.evaluation_id;
    block.append(node('summary', `${positionOf(record)} · ${statusLabels[record.status]} · evaluation ${brief(event.evaluation_id, 32)}`));
    if (record.status === 'complete' && definition.kind === 'histogram') {
      try { block.append(histogramPlot(record.value, label, definition.unit), binTable(record.value, label)); }
      catch (error) { block.append(node('p', error.message)); }
    }
    const link = node('a', 'Export raw evaluation ↗', 'text-link');
    link.href = `/api/v1${record.path}`; link.target = '_blank'; link.rel = 'noopener';
    block.append(link);
    const provenance = node('details');
    const content = node('div');
    provenance.append(node('summary', 'Definition and evaluation protocol'), content);
    provenance.addEventListener('toggle', () => {
      content.replaceChildren();
      if (provenance.open) content.append(node('pre', JSON.stringify({evaluation_id: event.evaluation_id,
        catalog: event.catalog, definition_hash: definition.definition_hash, definition,
        protocol_sha256: event.protocol_sha256, snapshot_sha256: event.snapshot_sha256,
        snapshot_identity: event.snapshot_identity, evaluation_protocol: event.evaluation_protocol},
        null, 2), 'numeric-preview'));
    });
    block.append(provenance);
    return block;
  });
}
export function evaluationShelf(api, base, changed = () => {}) {
  const records = new Map();
  const cards = new Map();
  let definitions = {}, run = {};
  const ready = () => [...records.values()].filter(entry => entry.status === 'ready');
  // Derived from the catalog specification and the run's schedule, both already
  // fetched; a run with nothing scheduled otherwise looks exactly like one that
  // has simply not reached its first interval yet.
  function renderUnscheduled(snapshots) {
    const target = document.getElementById('evaluation-unscheduled');
    if (!target) return;
    const manual = snapshots.filter(([, definition]) => (definition.specification || {}).trigger !== 'interval').map(([id]) => id);
    const scheduled = Object.keys(run.evaluation_schedule || {}).length > 0;
    if (!snapshots.length || manual.length !== snapshots.length || scheduled) {
      target.replaceChildren();
      target.hidden = true;
      return;
    }
    const named = manual.slice(0, 8).join(', ') + (manual.length > 8 ? `, and ${manual.length - 8} more` : '');
    target.replaceChildren(node('strong', 'No automatic evaluation is scheduled. '), node('span',
      `Snapshot ${manual.length > 1 ? 'metrics' : 'metric'} ${named} set trigger = "manual", so this run `
      + 'publishes no evaluation result while it trains, however many steps it reaches. '
      + 'Remove trigger = "manual" (or set trigger = "interval" with every_steps) in the '
      + 'configuration, then apply it with: hypergan resume RUN --config CONFIG'));
    target.hidden = false;
  }
  function renderSchedule() {
    const snapshots = Object.entries(definitions).filter(([, definition]) => definition.scope === 'snapshot');
    renderUnscheduled(snapshots);
    const items = snapshots.map(([id, definition]) => {
      const spec = definition.specification || {};
      const schedule = run.evaluation_schedule?.[id] || {};
      const interval = spec.trigger === 'interval';
      const card = node('li'); card.dataset.metric = id;
      card.append(node('h3', definition.label || id), node('p', id, 'quiet'));
      const states = {running: 'Running', complete: 'Complete', failed: 'Failed', skipped: 'Skipped', cancelled: 'Cancelled', pending: 'Pending', disabled: 'Disabled'};
      const evaluated = ready().some(({record}) => record.status === 'complete' && record.id === id);
      const status = states[schedule.status] || (evaluated ? 'Evaluated' : 'Not evaluated');
      card.append(node('p', `${status} · ${interval ? `every ${spec.every_steps} steps` : 'manual'}`, 'evaluation-schedule-status'));
      if (Number.isSafeInteger(schedule.source_step)) card.append(node('p', `Source step ${schedule.source_step}`, 'quiet'));
      if (interval && schedule.status !== 'disabled') {
        const next = Number.isSafeInteger(schedule.next_step) ? schedule.next_step :
          Number.isSafeInteger(spec.every_steps) && spec.every_steps > 0 ? (Math.floor((run.steps || 0) / spec.every_steps) + 1) * spec.every_steps : null;
        if (Number.isSafeInteger(next)) card.append(node('p', `Next evaluation at step ${next}${['running', 'training'].includes(run.status) ? '' : ' when training continues'}`, 'evaluation-next-step'));
        if (spec.on_busy === 'skip') card.append(node('p', 'If an evaluation is still running, the next scheduled evaluation is skipped.', 'quiet'));
      }
      if (spec.evaluation?.device) card.append(node('p', `Evaluation device: ${spec.evaluation.device}`, 'quiet'));
      if (Number.isSafeInteger(schedule.skipped_busy) && schedule.skipped_busy > 0) {
        card.append(node('p', `${schedule.skipped_busy} scheduled evaluations skipped while an evaluator was busy${Number.isSafeInteger(schedule.last_skipped_step) ? ` · last skipped step ${schedule.last_skipped_step}` : ''}`, 'evaluation-busy-skips'));
      }
      if (schedule.reason) card.append(node('p', schedule.reason === 'worker_busy' ? 'An evaluator was still running at the scheduled step.' : schedule.reason, 'evaluation-schedule-reason'));
      return card;
    });
    document.getElementById('evaluation-schedules').replaceChildren(...items);
    document.getElementById('evaluations').hidden = items.length === 0 && records.size === 0 && !inventory.some(s => /^evaluation:[0-9a-f]{32}$/.test(s.stream_id));
  }
  // One persistent card, chart instance and details element per metric, so new
  // results extend the chart in place and an expanded list stays expanded.
  function cardFor(id) {
    let card = cards.get(id);
    if (card) return card;
    const element = node('li', undefined, 'evaluation-metric');
    element.dataset.metric = id;
    const canvas = node('div', undefined, 'chart-canvas');
    canvas.setAttribute('role', 'img');
    const details = node('details', undefined, 'evaluation-results');
    const summary = node('summary'), content = node('div');
    details.append(summary, content);
    card = {element, canvas, chart: null, details, summary, content, label: id, records: [], signature: null};
    details.addEventListener('toggle', () => {
      content.replaceChildren();
      if (!details.open) return;
      try { content.append(resultsTable(card.label, card.records), ...resultBlocks(card.label, card.records)); }
      catch (error) { content.append(node('p', error.message)); }
    });
    cards.set(id, card);
    return card;
  }
  function renderCard(id, entries) {
    const card = cardFor(id);
    card.records = entries;
    // Charts are redrawn only when this metric gained a result, not on every
    // other stream that happens to finish loading.
    const signature = entries.map(record => record.event.evaluation_id).join(',');
    if (card.signature !== signature) { card.signature = signature; card.stale = true; }
    const latest = entries.filter(record => record.status === 'complete').at(-1) || entries.at(-1);
    const definition = latest.definition, label = definition.label || id;
    card.label = label;
    const complete = entries.filter(record => record.status === 'complete');
    const scalars = complete.filter(record => record.definition.kind === 'scalar');
    const children = [node('h3', label), node('p', id, 'quiet')];
    if (complete.length) {
      const last = complete.at(-1);
      children.push(node('p', finite(last.value) ? `${last.value} ${last.definition.unit || ''}`.trim()
        : `${last.value.counts.length} histogram bins`, 'evaluation-value'));
    }
    if (scalars.length) {
      card.canvas.setAttribute('aria-label',
        `${label} by source step; exact values are in the results list below`);
      children.push(card.canvas, node('p',
        `${scalars.length} completed ${scalars.length === 1 ? 'evaluation' : 'evaluations'}`
        + ' · source step on the horizontal axis · one point per evaluation', 'chart-note'));
    }
    const problems = entries.filter(record => record.status !== 'complete');
    if (problems.length) {
      const list = node('ul', undefined, 'evaluation-status');
      for (const record of problems) {
        const reason = brief(record.event.measurement_status?.[id]?.reason);
        const item = node('li', `${positionOf(record)} · ${statusLabels[record.status]}${reason ? ` · ${reason}` : ''}`,
          record.status === 'cancelled' ? 'evaluation-cancelled' : 'evaluation-failure');
        item.dataset.evaluation = record.event.evaluation_id;
        list.append(item);
      }
      children.push(list);
    }
    card.summary.textContent = `${entries.length} ${entries.length === 1 ? 'result' : 'results'}`
      + ' · steps, duration, device and protocol';
    children.push(card.details);
    card.element.dataset.steps = complete.map(record => record.event.step).join(',');
    card.element.replaceChildren(...children);
    // An expanded list is rebuilt in place; a collapsed one stays unallocated.
    if (card.details.open) card.details.dispatchEvent(new Event('toggle'));
    return {card, scalars};
  }
  function renderResults() {
    const grouped = new Map();
    for (const entry of ready()) {
      if (!grouped.has(entry.record.id)) grouped.set(entry.record.id, []);
      grouped.get(entry.record.id).push(entry.record);
    }
    const drawn = [];
    const items = [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([id, entries]) => {
      const {card, scalars} = renderCard(id, entries.sort(byStep));
      if (scalars.length && (card.stale || !card.chart)) drawn.push(card);
      return card.element;
    });
    for (const [id, card] of cards) if (!grouped.has(id)) { card.chart?.dispose(); cards.delete(id); }
    for (const entry of records.values()) {
      if (entry.status === 'ready') continue;
      const element = entry.element || (entry.element = node('li', undefined, 'evaluation-metric'));
      element.replaceChildren(...(entry.status === 'error'
        ? [node('h3', 'Evaluation unavailable'), node('p', entry.message)]
        : [node('p', 'Loading evaluation…')]));
      items.push(element);
    }
    document.getElementById('evaluation-items').replaceChildren(...items);
    // echarts measures the element, so it is initialized once the card is attached.
    for (const card of drawn) {
      card.chart ||= init(card.canvas, null, {renderer: 'canvas'});
      card.chart.setOption({...chartStyle(false), series: chartSeries(card.records)}, true);
      card.stale = false;
    }
  }
  function update(catalog, currentRun) {
    definitions = catalog?.metrics || {};
    run = currentRun || {};
    renderSchedule();
  }
  let pending = false, again = false, runPath = null, inventory = [];
  const items = document.getElementById('evaluation-items');
  if (items) new ResizeObserver(() => { for (const card of cards.values()) card.chart?.resize(); }).observe(items);
  async function refresh(streams) {
    if (streams) inventory = streams;
    if (pending) { again = true; return; }
    pending = true;
    try {
      do {
        again = false;
        const path = base();
        if (runPath !== path) {
          for (const card of cards.values()) card.chart?.dispose();
          cards.clear(); records.clear(); changed([]);
          document.getElementById('evaluation-items').replaceChildren(); runPath = path;
        }
        const selected = (inventory || []).filter(s => /^evaluation:[0-9a-f]{32}$/.test(s.stream_id)).slice(0, 64);
        renderSchedule();
        for (const stream of selected) {
          let entry = records.get(stream.stream_id);
          if (entry && !entry.retry) continue;
          if (!entry) { entry = {status: 'loading'}; records.set(stream.stream_id, entry); renderResults(); }
          entry.retry = false;
          const eventsPath = `${path}/events?${new URLSearchParams({stream_id: stream.stream_id, limit: '2'})}`;
          try {
            if (stream.error) throw new Error(stream.error);
            const response = await api(eventsPath, {signal: AbortSignal.timeout(15000)}), page = response.data;
            if (page.has_more || page.partial_tail || page.events?.length !== 1 || JSON.stringify(page).length > 131072) throw new Error('Expected one bounded, complete evaluation document');
            const event = page.events[0];
            if (event.event !== 'evaluation' || event.stream_id !== stream.stream_id || event.stream_generation !== stream.stream_id.slice(11)) throw new Error('Evaluation source identity differs');
            const catalog = (await api(`${path}/metrics/catalog?${new URLSearchParams({revision: event.catalog})}`, {signal: AbortSignal.timeout(15000)})).data;
            if (path !== base()) { again = true; break; }
            Object.assign(entry, {status: 'ready', event, catalog, record: readResult(event, catalog, eventsPath)});
            renderResults();
            changed(ready());
            renderSchedule();
          } catch (error) {
            Object.assign(entry, {status: 'error', retry: true, message: error.message});
            renderResults();
          }
        }
      } while (again);
    } finally { pending = false; }
  }
  return {refresh, update};
}
