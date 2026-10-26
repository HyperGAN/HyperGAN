// Global runs dashboard. Kept external so the viewer's script policy stays intact.
const byId = id => document.getElementById(id);
const body = byId('runs-body');
const rows = new Map([...body.children].map(row => [row.dataset.id, row]));
const pendingActions = new Map();
const dateFormat = new Intl.DateTimeFormat(undefined, {dateStyle: 'medium', timeStyle: 'short'});
byId('time-zone').textContent = `(${dateFormat.resolvedOptions().timeZone})`;

function statusGroup(row) {
  if (row.dataset.active === 'true' || row.dataset.status === 'running') return 'running';
  const status = row.dataset.status;
  if (['complete', 'completed'].includes(status)) return 'complete';
  if (['stopped', 'interrupted', 'cancelled', 'canceled'].includes(status)) return 'stopped';
  return ['failed'].includes(status) ? status : 'unknown';
}

function formatCreated(row) {
  const time = row.querySelector('[data-field="created"]');
  const date = new Date(row.dataset.createdAt);
  const valid = row.dataset.createdAt && Number.isFinite(date.getTime());
  time.textContent = valid ? dateFormat.format(date) : 'Unknown';
  time.title = valid ? date.toISOString() : 'Creation time was not recorded';
  if (valid) time.dateTime = date.toISOString();
  else time.removeAttribute('datetime');
}

function filterRows() {
  const query = byId('search').value.trim().toLowerCase();
  const status = byId('status-filter').value;
  let visible = 0;
  for (const row of rows.values()) {
    const text = `${row.dataset.name} ${row.dataset.id} ${row.dataset.slug}`.toLowerCase();
    row.hidden = !(text.includes(query) && (!status || statusGroup(row) === status));
    if (!row.hidden) visible++;
  }
  const total = rows.size;
  const count = total === 1 ? 'run' : 'runs';
  const summary = query || status ? `${visible} of ${total} ${count}` : `${total} ${count}`;
  if (byId('run-count').textContent !== summary) byId('run-count').textContent = summary;
  byId('clear-filters').hidden = !(query || status);
  byId('empty-state').hidden = visible > 0;
  byId('empty-title').textContent = total ? 'No matching runs' : 'No training runs yet';
  byId('empty-description').textContent = total
    ? 'Try another search or clear the filters to see all runs.'
    : 'Start a training run in this workspace. It will appear here automatically.';
}

function sortRows() {
  const order = byId('sort-order').value;
  const sorted = [...rows.values()].sort((a, b) => {
    if (order !== 'name') {
      const left = Date.parse(a.dataset.createdAt);
      const right = Date.parse(b.dataset.createdAt);
      // Unknown dates belong at the end in both chronological orders.
      if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
      if (Number.isFinite(left) && left !== right) return order === 'oldest' ? left - right : right - left;
    }
    return a.dataset.name.localeCompare(b.dataset.name) || a.dataset.id.localeCompare(b.dataset.id);
  });
  const focused = document.activeElement;
  sorted.forEach((row, index) => {
    if (body.children[index] !== row) body.insertBefore(row, body.children[index] || null);
  });
  if (focused?.isConnected && document.activeElement !== focused) focused.focus({preventScroll: true});
}

function updateRow(row, run) {
  const id = String(run.run_id);
  const status = String(run.status || 'unknown').trim().toLowerCase();
  const active = Boolean(run.active || run.running || status === 'running');
  row.dataset.id = id;
  row.dataset.name = String(run.name || id);
  row.dataset.slug = String(run.slug || '');
  row.dataset.status = status;
  row.dataset.active = String(active);
  row.dataset.createdAt = run.created_at || '';
  row.querySelector('.run-link').textContent = row.dataset.name;
  row.querySelectorAll('a').forEach(link => { link.href = `/runs/${encodeURIComponent(id)}`; });
  row.querySelector('.slug-badge').textContent = row.dataset.slug;
  const shortId = row.querySelector('.id-badge');
  shortId.textContent = run.short_id || id.slice(0, 8);
  shortId.title = id;
  const badge = row.querySelector('.badge');
  badge.dataset.status = status;
  badge.textContent = status;
  formatCreated(row);
  const step = Number(run.step) || 0;
  const total = Number(run.total_steps || run.max_steps) || 0;
  row.querySelector('[data-field="steps"]').textContent = total > 0
    ? `${step.toLocaleString()} / ${total.toLocaleString()}` : step.toLocaleString();
  row.querySelector('.progress-bar').hidden = total <= 0;
  const percent = total > 0 ? Math.min(100, Math.max(0, step / total * 100)) : 0;
  row.querySelector('.progress-fill').style.width = `${percent}%`;
  const speed = run.steps_per_second;
  row.querySelector('[data-field="speed"]').textContent = typeof speed === 'number' && Number.isFinite(speed)
    ? `${speed.toFixed(1)} it/s` : '—';
  row.querySelector('[data-field="disk"]').textContent = run.disk_formatted || '—';
  updateActions(row, active);
}

function updateActions(row, active = row.dataset.active === 'true') {
  for (const button of row.querySelectorAll('[data-action]')) {
    const action = button.dataset.action;
    const label = action === 'stop' ? 'Stop' : action === 'sample' ? 'Sample' : 'Checkpoint';
    const pending = pendingActions.get(row.dataset.id);
    button.hidden = !active;
    button.disabled = Boolean(pending);
    button.textContent = pending === action ? `${label}…` : label;
    button.setAttribute('aria-label', `${label} ${row.dataset.name}`);
  }
  row.querySelector('.actions-cell a').setAttribute('aria-label', `Open run ${row.dataset.name}`);
}

// A token-mode dashboard serves no run data until this browser has a session.
function signedOut() { return !byId('login').hidden; }
function showLogin() {
  byId('login').hidden = false;
  byId('refresh-status').textContent = 'Sign in to see runs';
  byId('token').focus();
}

let refreshing = null;
function refreshRuns() {
  if (refreshing) return refreshing;
  byId('refresh').disabled = true;
  byId('refresh-status').textContent = 'Refreshing…';
  refreshing = (async () => {
    try {
      const response = await fetch('/api/v1/runs', {cache: 'no-store', credentials: 'same-origin'});
      if (response.status === 401) { showLogin(); return; }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const runs = await response.json();
      if (!Array.isArray(runs) || runs.some(run => !run || typeof run.run_id !== 'string')) {
        throw new Error('Invalid runs response');
      }
      const seen = new Set();
      for (const run of runs) {
        const id = run.run_id;
        seen.add(id);
        let row = rows.get(id);
        if (!row) {
          row = byId('run-row-template').content.firstElementChild.cloneNode(true);
          rows.set(id, row);
          body.append(row);
        }
        updateRow(row, run);
      }
      for (const [id, row] of rows) {
        if (!seen.has(id)) { row.remove(); rows.delete(id); }
      }
      sortRows();
      filterRows();
      byId('refresh-status').textContent = `Updated ${new Date().toLocaleTimeString()} · every 3 seconds`;
      byId('refresh-status').classList.remove('error');
    } catch (error) {
      byId('refresh-status').textContent = `Refresh failed (${error.message}). Retrying automatically.`;
      byId('refresh-status').classList.add('error');
    } finally {
      byId('refresh').disabled = false;
      refreshing = null;
    }
  })();
  return refreshing;
}

body.addEventListener('click', async event => {
  const button = event.target.closest('button[data-action]');
  if (!button || button.disabled) return;
  const row = button.closest('tr');
  const id = row.dataset.id;
  const name = row.dataset.name;
  const action = button.dataset.action;
  const notice = byId('action-status');
  pendingActions.set(id, action);
  updateActions(row);
  const actionLabel = action === 'stop' ? 'stop' : action === 'sample' ? 'sample' : 'checkpoint';
  const titleLabel = action === 'stop' ? 'Stop' : action === 'sample' ? 'Sample' : 'Checkpoint';
  notice.hidden = false;
  notice.classList.remove('error');
  notice.textContent = `Requesting ${actionLabel} for ${name}…`;
  try {
    const response = await fetch(`/api/v1/runs/${encodeURIComponent(id)}/${action}`,
      {method: 'POST', credentials: 'same-origin'});
    if (response.status === 401) showLogin();
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    notice.textContent = `${titleLabel} requested for ${name}.`;
    await refreshRuns();
  } catch (error) {
    notice.textContent = `Could not ${action} ${name}: ${error.message}`;
    notice.classList.add('error');
  } finally {
    pendingActions.delete(id);
    if (rows.has(id)) updateActions(rows.get(id));
  }
});

byId('login').addEventListener('submit', async event => {
  event.preventDefault();
  const token = byId('token').value;
  byId('token').value = '';
  byId('login-error').textContent = '';
  try {
    const response = await fetch('/api/v1/session', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token}),
    });
    if (!response.ok) throw new Error(response.status === 401 ? 'That token was not accepted.' : `HTTP ${response.status}`);
    byId('login').hidden = true;
    await refreshRuns();
  } catch (error) {
    byId('login-error').textContent = error.message;
  }
});
byId('search').addEventListener('input', filterRows);
byId('status-filter').addEventListener('change', filterRows);
byId('sort-order').addEventListener('change', sortRows);
byId('clear-filters').addEventListener('click', () => {
  byId('search').value = '';
  byId('status-filter').value = '';
  filterRows();
  byId('search').focus();
});
byId('refresh').addEventListener('click', refreshRuns);
for (const row of rows.values()) { formatCreated(row); updateActions(row); }
sortRows();
filterRows();
// Poll even an empty workspace so new and resumed runs appear without a reload.
const refreshTimer = setInterval(() => { if (!document.hidden && !signedOut()) refreshRuns(); }, 3000);
document.addEventListener('visibilitychange', () => { if (!document.hidden && !signedOut()) refreshRuns(); });
window.addEventListener('pagehide', event => { if (!event.persisted) clearInterval(refreshTimer); });
