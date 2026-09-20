// Immutable snapshot results use their own source catalog and position. Loading
// is serialized, bounded to the server's 64 streams, and never reduces history.
const node = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
};
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const svgNode = (tag, attrs, text) => {
  const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) result.setAttribute(key, value);
  if (text !== undefined) result.textContent = text;
  return result;
};
function plot(value, label, step, unit) {
  const svg = svgNode('svg', {viewBox: '0 0 600 180', role: 'img', 'aria-label': label, class: 'evaluation-plot'});
  svg.append(svgNode('title', {}, label));
  if (finite(value)) {
    svg.append(svgNode('circle', {cx: 300, cy: 65, r: 6}));
    svg.append(svgNode('text', {x: 300, y: 100, 'text-anchor': 'middle'}, `${value} ${unit || ''}`));
    svg.append(svgNode('text', {x: 300, y: 140, 'text-anchor': 'middle'}, `Source step ${step}`));
    return svg;
  }
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
function showResult(card, event, catalog, path) {
  const ids = [...new Set([...Object.keys(event.metrics || {}), ...Object.keys(event.distributions || {}), ...Object.keys(event.measurement_status || {})])];
  if (ids.length !== 1) throw new Error('Expected one metric per evaluation');
  const id = ids[0], definition = catalog.metrics[id];
  if (!definition) throw new Error('Evaluation definition is missing from its source catalog');
  const failed = event.status === 'failed';
  if (!failed && event.status !== 'complete') throw new Error('Evaluation is not terminal');
  const heading = node('h3', definition.label || id);
  const status = node('span', failed ? 'Failed' : 'Complete', 'badge');
  card.replaceChildren(heading, status, node('p', id, 'quiet'));
  const known = event.source_position_known === true;
  card.dataset.step = known ? String(event.step) : '';
  card.dataset.evaluation = event.evaluation_id;
  if (finite(event.seconds)) card.append(node('p', `Evaluation duration: ${event.seconds} seconds`, 'quiet'));
  card.append(node('p', known ? `Source step ${event.step} · Attempt ${event.attempt_id}` : 'Source position unknown'));
  const link = node('a', 'Export raw evaluation ↗', 'text-link');
  link.href = `/api/v1${path}`; link.target = '_blank'; link.rel = 'noopener';
  card.append(link);
  if (failed) {
    card.append(node('p', 'The evaluator did not complete.', 'evaluation-failure'));
    const failure = node('details');
    failure.append(node('summary', 'Failure details'));
    failure.append(node('pre', event.measurement_status?.[id]?.reason || 'Evaluation failed', 'numeric-preview'));
    card.append(failure);
  } else {
    if (!known || !Number.isSafeInteger(event.step) || event.step < 0) throw new Error('Complete evaluation needs a known source step');
    const value = definition.kind === 'histogram' ? event.distributions?.[id] : event.metrics?.[id];
    validateValue(value, definition.kind);
    card.append(node('p', finite(value) ? `${value} ${definition.unit || ''}` : `${value.counts.length} histogram bins`, 'evaluation-value'));
    const details = node('details'); details.append(node('summary', 'Plot and accessible values'));
    const content = node('div'); details.append(content);
    // Allocate charts/tables only on explicit expansion; release them on close.
    details.addEventListener('toggle', () => {
      content.replaceChildren();
      if (!details.open) return;
      try {
        content.append(plot(value, definition.label || id, event.step, definition.unit));
        const table = node('table'); table.append(node('caption', `${definition.label || id} · exact values`));
        const head = node('thead'), header = node('tr');
        for (const title of finite(value) ? ['Source step', 'Value'] : ['Bin start', 'Bin end', 'Value']) {
          const cell = node('th', title); cell.scope = 'col'; header.append(cell);
        }
        head.append(header); table.append(head);
        const body = node('tbody');
        const rows = finite(value) ? [[event.step, value]] : value.counts.map((count, i) => [value.edges[i], value.edges[i+1], count]);
        for (const row of rows) { const tr = node('tr'); for (const v of row) tr.append(node('td', String(v))); body.append(tr); }
        table.append(body); const scroll = node('div', undefined, 'table-scroll'); scroll.tabIndex = 0; scroll.append(table); content.append(scroll);
      } catch (error) { content.append(node('p', error.message)); }
    });
    card.append(details);
  }
  const provenance = node('details'); provenance.append(node('summary', 'Definition and evaluation protocol'));
  provenance.append(node('pre', JSON.stringify({evaluation_id: event.evaluation_id, catalog: event.catalog,
    definition_hash: definition.definition_hash, definition, protocol_sha256: event.protocol_sha256,
    snapshot_sha256: event.snapshot_sha256, snapshot_identity: event.snapshot_identity,
    evaluation_protocol: event.evaluation_protocol}, null, 2), 'numeric-preview'));
  card.append(provenance);
}
export function evaluationShelf(api, base, changed = () => {}) {
  const results = new Map();
  const records = new Map();
  let pending = false, again = false, runPath = null, inventory = [];
  async function refresh(streams) {
    if (streams) inventory = streams;
    if (pending) { again = true; return; }
    pending = true;
    try {
      do {
        again = false;
        const path = base();
        if (runPath !== path) { records.clear(); results.clear(); changed([]); document.getElementById('evaluation-items').replaceChildren(); runPath = path; }
        const selected = (inventory || []).filter(s => /^evaluation:[0-9a-f]{32}$/.test(s.stream_id)).slice(0, 64);
        document.getElementById('evaluations').hidden = selected.length === 0;
        for (const stream of selected) {
          let card = records.get(stream.stream_id);
          if (card && !card.dataset.retry) continue;
          if (!card) {
            card = node('li', 'Loading evaluation…'); records.set(stream.stream_id, card);
            document.getElementById('evaluation-items').append(card);
          }
          delete card.dataset.retry;
          const eventsPath = `${path}/events?${new URLSearchParams({stream_id: stream.stream_id, limit: '2'})}`;
          try {
            if (stream.error) throw new Error(stream.error);
            const response = await api(eventsPath, {signal: AbortSignal.timeout(15000)}), page = response.data;
            if (page.has_more || page.partial_tail || page.events?.length !== 1 || JSON.stringify(page).length > 131072) throw new Error('Expected one bounded, complete evaluation document');
            const event = page.events[0];
            if (event.event !== 'evaluation' || event.stream_id !== stream.stream_id || event.stream_generation !== stream.stream_id.slice(11)) throw new Error('Evaluation source identity differs');
            const catalog = (await api(`${path}/metrics/catalog?${new URLSearchParams({revision: event.catalog})}`, {signal: AbortSignal.timeout(15000)})).data;
            if (path !== base()) { again = true; break; }
            showResult(card, event, catalog, eventsPath);
            results.set(stream.stream_id, {event, catalog});
            const ordered = [...records.values()].sort((a, b) =>
              (a.dataset.step === '' || a.dataset.step === undefined ? Infinity : Number(a.dataset.step)) -
              (b.dataset.step === '' || b.dataset.step === undefined ? Infinity : Number(b.dataset.step)) ||
              (a.dataset.evaluation || '').localeCompare(b.dataset.evaluation || ''));
            document.getElementById('evaluation-items').replaceChildren(...ordered);
            changed([...results.values()]);
          } catch (error) { card.dataset.retry = 'true'; card.replaceChildren(node('h3', 'Evaluation unavailable'), node('p', error.message)); }
        }
      } while (again);
    } finally { pending = false; }
  }
  return {refresh};
}
