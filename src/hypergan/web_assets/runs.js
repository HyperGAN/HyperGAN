// Global runs observatory index. Kept external so strict CSP stays intact.
const byId = id => document.getElementById(id);
const body = byId('runs-body');
const rows = new Map([...body.children].map(row => [row.dataset.id, row]));
const dateFormat = new Intl.DateTimeFormat(undefined, {dateStyle: 'medium', timeStyle: 'short'});
setText(byId('time-zone'), `(${dateFormat.resolvedOptions().timeZone})`);

function setText(el, next) {
  if (!el) return;
  const value = String(next ?? '');
  if (el.textContent !== value) el.textContent = value;
}

function setAttr(el, attr, next) {
  if (!el) return;
  if (next === null || next === undefined || next === '') {
    if (el.hasAttribute(attr)) el.removeAttribute(attr);
    return;
  }
  const value = String(next);
  if (el.getAttribute(attr) !== value) el.setAttribute(attr, value);
}

function setDataAttr(el, attr, next) {
  if (!el) return;
  const value = String(next ?? '');
  if (el.getAttribute(attr) !== value) el.setAttribute(attr, value);
}

function setHidden(el, hidden) {
  if (!el) return;
  const next = Boolean(hidden);
  if (el.hidden !== next) el.hidden = next;
}

function setClass(el, cls, enabled) {
  if (!el) return;
  const want = Boolean(enabled);
  if (el.classList.contains(cls) !== want) el.classList.toggle(cls, want);
}

function isFiniteNum(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNonNegFinite(value) {
  return isFiniteNum(value) && value >= 0;
}

function parseOptionalNumber(val) {
  if (val === undefined || val === null || val === '') return NaN;
  const num = Number(val);
  return Number.isFinite(num) ? num : NaN;
}

function formatEta(seconds) {
  if (!isFiniteNum(seconds) || seconds <= 0) return '';
  const totalSec = Math.max(1, Math.round(seconds));
  if (totalSec < 60) return `ETA ${totalSec}s`;
  if (totalSec < 3600) {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `ETA ${mins}m ${String(secs).padStart(2, '0')}s`;
  }
  if (totalSec < 86400) {
    const hours = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    return `ETA ${hours}h ${String(mins).padStart(2, '0')}m`;
  }
  const days = Math.floor(totalSec / 86400);
  const hours = Math.floor((totalSec % 86400) / 3600);
  return `ETA ${days}d ${String(hours).padStart(2, '0')}h`;
}

function statusGroup(row) {
  if (row.dataset.active === 'true' || ['running', 'training'].includes(row.dataset.status)) {
    return 'running';
  }
  const status = row.dataset.status;
  if (['complete', 'completed'].includes(status)) return 'complete';
  if (['stopped', 'interrupted', 'cancelled', 'canceled'].includes(status)) return 'stopped';
  return status === 'failed' ? 'failed' : 'unknown';
}

function formatCreated(row) {
  const time = row.querySelector('[data-field="created"]');
  if (!time) return;
  const raw = row.dataset.createdAt || '';
  const date = new Date(raw);
  const valid = Boolean(raw) && Number.isFinite(date.getTime());
  const iso = valid ? date.toISOString() : '';
  setText(time, valid ? dateFormat.format(date) : 'Unknown');
  setAttr(time, 'title', valid ? iso : 'Creation time was not recorded');
  setAttr(time, 'datetime', valid ? iso : null);
}

function filterRows() {
  const query = byId('search').value.trim().toLowerCase();
  const status = byId('status-filter').value;
  let visible = 0;
  for (const row of rows.values()) {
    const text = `${row.dataset.name || ''} ${row.dataset.id || ''} ${row.dataset.slug || ''}`.toLowerCase();
    const shouldHide = !(text.includes(query) && (!status || statusGroup(row) === status));
    setHidden(row, shouldHide);
    if (!shouldHide) visible++;
  }
  const total = rows.size;
  const count = total === 1 ? 'run' : 'runs';
  const summary = query || status ? `${visible} of ${total} ${count}` : `${total} ${count}`;
  setText(byId('run-count'), summary);
  setHidden(byId('clear-filters'), !(query || status));
  setHidden(byId('empty-state'), visible > 0);
  setText(byId('empty-title'), total ? 'No matching runs' : 'No training runs yet');
  setText(
    byId('empty-description'),
    total
      ? 'Try another search or clear the filters to see all runs.'
      : 'Start a training run in this workspace. It will appear here automatically.',
  );
}

function compareByNameAsc(a, b) {
  return (
    (a.dataset.name || '').localeCompare(b.dataset.name || '') ||
    (a.dataset.id || '').localeCompare(b.dataset.id || '')
  );
}

function compareByCreatedDesc(a, b) {
  const left = Date.parse(a.dataset.createdAt);
  const right = Date.parse(b.dataset.createdAt);
  if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
  if (Number.isFinite(left) && left !== right) return right - left;
  return compareByNameAsc(a, b);
}

function progressSortTuple(row) {
  const step = Number(row.dataset.step) || 0;
  const total = Number(row.dataset.totalSteps) || 0;
  const ratio = total > 0 ? step / total : step > 0 ? 0 : -1;
  return [ratio, step];
}

function syncSortHeaders(order) {
  for (const th of document.querySelectorAll('#runs-table thead th[data-sort-key]')) {
    const key = th.dataset.sortKey;
    let ariaSort = 'none';
    if (key === order) {
      ariaSort = order === 'name' || order === 'fid' ? 'ascending' : 'descending';
    } else if (key === 'newest' && order === 'oldest') {
      ariaSort = 'ascending';
    }
    setAttr(th, 'aria-sort', ariaSort);
  }
}

function sortRows() {
  const order = byId('sort-order').value || 'newest';
  syncSortHeaders(order);
  const sorted = [...rows.values()].sort((a, b) => {
    if (order === 'name') {
      return compareByNameAsc(a, b);
    }
    if (order === 'oldest') {
      const left = Date.parse(a.dataset.createdAt);
      const right = Date.parse(b.dataset.createdAt);
      if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
      if (Number.isFinite(left) && left !== right) return left - right;
      return compareByNameAsc(a, b);
    }
    if (order === 'fid') {
      const left = parseOptionalNumber(a.dataset.fid);
      const right = parseOptionalNumber(b.dataset.fid);
      if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
      if (Number.isFinite(left) && left !== right) return left - right;
      return compareByCreatedDesc(a, b);
    }
    if (order === 'progress') {
      const [lRatio, lStep] = progressSortTuple(a);
      const [rRatio, rStep] = progressSortTuple(b);
      if (lRatio !== rRatio) return rRatio - lRatio;
      if (lStep !== rStep) return rStep - lStep;
      return compareByCreatedDesc(a, b);
    }
    if (order === 'speed') {
      const lActive = a.dataset.active === 'true';
      const rActive = b.dataset.active === 'true';
      const left = lActive ? parseOptionalNumber(a.dataset.speed) : NaN;
      const right = rActive ? parseOptionalNumber(b.dataset.speed) : NaN;
      if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
      if (Number.isFinite(left) && left !== right) return right - left;
      if (lActive !== rActive) return lActive ? -1 : 1;
      return compareByCreatedDesc(a, b);
    }
    if (order === 'disk') {
      const left = parseOptionalNumber(a.dataset.diskBytes);
      const right = parseOptionalNumber(b.dataset.diskBytes);
      if (Number.isFinite(left) !== Number.isFinite(right)) return Number.isFinite(left) ? -1 : 1;
      if (Number.isFinite(left) && left !== right) return right - left;
      return compareByCreatedDesc(a, b);
    }
    return compareByCreatedDesc(a, b);
  });
  const focused = document.activeElement;
  sorted.forEach((row, index) => {
    if (body.children[index] !== row) body.insertBefore(row, body.children[index] || null);
  });
  if (focused?.isConnected && document.activeElement !== focused) focused.focus({preventScroll: true});
}

function updateRow(row, run) {
  const id = String(run.run_id);
  const name = String(run.name || id);
  const slug = String(run.slug || '');
  const status = String(run.status || 'unknown').trim().toLowerCase();
  const active =
    typeof run.active === 'boolean'
      ? run.active
      : typeof run.running === 'boolean'
        ? run.running
        : ['running', 'training', 'initializing', 'starting'].includes(status);
  const url = String(run.url || `/runs/${encodeURIComponent(id)}`);
  const step = Math.max(0, Math.trunc(Number(run.step) || 0));
  const total = Math.max(0, Math.trunc(Number(run.total_steps || run.max_steps) || 0));
  const bestFid = isFiniteNum(run.best_fid) ? run.best_fid : isFiniteNum(run.fid_best) ? run.fid_best : null;
  const latestFid = isFiniteNum(run.latest_fid) ? run.latest_fid : isFiniteNum(run.fid_latest) ? run.fid_latest : null;
  const sortFid = bestFid ?? latestFid;
  const rawSpeed = run.steps_per_second ?? run.speed_its;
  const hasActiveSpeed = active && isNonNegFinite(rawSpeed);
  const hasActiveSamples = hasActiveSpeed && isNonNegFinite(run.samples_per_second);
  const diskBytes = isNonNegFinite(run.disk_bytes) ? Math.trunc(run.disk_bytes) : null;

  setClass(row, 'run-row', true);
  setDataAttr(row, 'data-url', url);
  setDataAttr(row, 'data-id', id);
  setDataAttr(row, 'data-name', name);
  setDataAttr(row, 'data-slug', slug);
  setDataAttr(row, 'data-status', status);
  setDataAttr(row, 'data-active', String(active));
  setDataAttr(row, 'data-created-at', run.created_at || '');
  setDataAttr(row, 'data-fid', sortFid !== null ? String(sortFid) : '');
  setDataAttr(row, 'data-step', String(step));
  setDataAttr(row, 'data-total-steps', String(total));
  setDataAttr(row, 'data-speed', hasActiveSpeed ? String(rawSpeed) : '');
  setDataAttr(row, 'data-disk-bytes', diskBytes !== null ? String(diskBytes) : '');

  const runLink = row.querySelector('.run-link');
  setText(runLink, name);
  setAttr(runLink, 'href', url);

  const slugBadge = row.querySelector('.slug-badge');
  setText(slugBadge, slug);
  setHidden(slugBadge, !slug);

  const shortId = row.querySelector('.id-badge');
  setText(shortId, run.short_id || id.slice(0, 8));
  setAttr(shortId, 'title', id);

  const hostBadge = row.querySelector('[data-field="host"]');
  const showHost = Boolean(run.remote && run.hostname);
  setText(hostBadge, showHost ? String(run.hostname) : '');
  setAttr(hostBadge, 'title', showHost ? `Remote host: ${run.hostname}` : null);
  setHidden(hostBadge, !showHost);

  const thumbCell = row.querySelector('.thumbnail-cell');
  if (thumbCell) {
    const thumbUrl = run.thumbnail_url ? String(run.thumbnail_url) : '';
    const altText = `Preview of ${name}`;
    let img = thumbCell.querySelector('img.run-thumbnail');
    if (thumbUrl) {
      const placeholder = thumbCell.querySelector('.no-thumbnail');
      if (placeholder) placeholder.remove();
      if (!img) {
        img = document.createElement('img');
        img.className = 'run-thumbnail';
        img.loading = 'lazy';
        img.width = 48;
        img.height = 48;
        img.setAttribute('src', thumbUrl);
        img.setAttribute('alt', altText);
        thumbCell.append(img);
      } else {
        if (img.getAttribute('src') !== thumbUrl) {
          img.setAttribute('src', thumbUrl);
        }
        setAttr(img, 'alt', altText);
      }
    } else {
      if (img) img.remove();
      let placeholder = thumbCell.querySelector('.no-thumbnail');
      if (!placeholder) {
        placeholder = document.createElement('span');
        placeholder.className = 'no-thumbnail';
        placeholder.title = 'No previews yet';
        placeholder.textContent = '—';
        thumbCell.append(placeholder);
      }
    }
  }

  const badge = row.querySelector('.badge');
  setDataAttr(badge, 'data-status', status);
  setText(badge, status);

  const topo = run.topology && typeof run.topology === 'object' ? run.topology : {};
  const topoPrimary = [topo.label, run.batch_label].filter(Boolean).join(' · ');
  const topoSub = [topo.detail, run.batch_detail].filter(Boolean).join(' · ');
  const topoPrimaryEl = row.querySelector('[data-field="topology-primary"]');
  const topoSubEl = row.querySelector('[data-field="topology-sub"]');
  setText(topoPrimaryEl, topoPrimary || '—');
  setClass(topoPrimaryEl, 'muted', !topoPrimary);
  setText(topoSubEl, topoSub);
  setHidden(topoSubEl, !topoSub);

  const fidPrimaryEl = row.querySelector('[data-field="fid-primary"]');
  const fidLatestEl = row.querySelector('[data-field="fid-latest"]');
  if (fidPrimaryEl) {
    if (bestFid !== null) {
      if (fidPrimaryEl.className !== 'best-fid') fidPrimaryEl.className = 'best-fid';
      setAttr(fidPrimaryEl, 'title', 'Best FID');
      setText(fidPrimaryEl, bestFid.toFixed(2));
      const showLatestSub = latestFid !== null && Math.abs(latestFid - bestFid) > 0.01;
      setText(fidLatestEl, showLatestSub ? `(${latestFid.toFixed(2)})` : '');
      setHidden(fidLatestEl, !showLatestSub);
    } else if (latestFid !== null) {
      if (fidPrimaryEl.className !== 'latest-fid') fidPrimaryEl.className = 'latest-fid';
      setAttr(fidPrimaryEl, 'title', 'Latest FID');
      setText(fidPrimaryEl, latestFid.toFixed(2));
      setText(fidLatestEl, '');
      setHidden(fidLatestEl, true);
    } else {
      if (fidPrimaryEl.className !== 'no-fid muted') fidPrimaryEl.className = 'no-fid muted';
      setAttr(fidPrimaryEl, 'title', null);
      setText(fidPrimaryEl, '—');
      setText(fidLatestEl, '');
      setHidden(fidLatestEl, true);
    }
  }

  formatCreated(row);

  const progressText = total > 0 ? `${step.toLocaleString()} / ${total.toLocaleString()}` : step.toLocaleString();
  setText(row.querySelector('[data-field="steps"]'), progressText);
  setHidden(row.querySelector('.progress-bar'), total <= 0);
  const percent = total > 0 ? Math.min(100, Math.max(0, Math.round((step / total) * 100))) : 0;
  const progressFill = row.querySelector('.progress-fill');
  const widthStr = `${percent}%`;
  if (progressFill && progressFill.style.width !== widthStr) {
    progressFill.style.width = widthStr;
  }
  const etaText = active ? formatEta(run.eta_seconds) : '';
  const etaEl = row.querySelector('[data-field="eta"]');
  setText(etaEl, etaText);
  setHidden(etaEl, !etaText);

  const speedPrimaryEl = row.querySelector('[data-field="speed-primary"]');
  const speedSubEl = row.querySelector('[data-field="speed-sub"]');
  if (speedPrimaryEl) {
    const wantClass = hasActiveSpeed ? 'speed-primary' : 'muted';
    if (speedPrimaryEl.className !== wantClass) speedPrimaryEl.className = wantClass;
    setText(speedPrimaryEl, hasActiveSpeed ? `${rawSpeed.toFixed(1)} steps/s` : '—');
  }
  if (speedSubEl) {
    setText(speedSubEl, hasActiveSamples ? `${run.samples_per_second.toFixed(0)} samples/s` : '');
    setHidden(speedSubEl, !hasActiveSamples);
  }

  setText(row.querySelector('[data-field="disk"]'), run.disk_formatted || '—');
}

// A token-mode dashboard serves no run data until this browser has a session.
function signedOut() {
  return !byId('login').hidden;
}

function showLogin() {
  const statusEl = byId('refresh-status');
  const indicator = byId('live-indicator');
  setHidden(byId('login'), false);
  setText(statusEl, 'Sign in to see runs');
  setClass(statusEl, 'error', false);
  setClass(indicator, 'live', false);
  setClass(indicator, 'error', false);
  byId('token').focus();
}

let refreshing = null;
function refreshRuns({manual = false} = {}) {
  const refreshBtn = byId('refresh');
  const statusEl = byId('refresh-status');
  const indicator = byId('live-indicator');
  if (manual) {
    refreshBtn.disabled = true;
    setText(statusEl, 'Refreshing…');
  }
  if (refreshing) return refreshing;
  refreshing = (async () => {
    try {
      const response = await fetch('/api/v1/runs', {cache: 'no-store', credentials: 'same-origin'});
      if (response.status === 401) {
        showLogin();
        return;
      }
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
        if (!seen.has(id)) {
          row.remove();
          rows.delete(id);
        }
      }
      sortRows();
      filterRows();
      setText(statusEl, `Live · Updated ${new Date().toLocaleTimeString()}`);
      setClass(statusEl, 'error', false);
      setClass(indicator, 'error', false);
      setClass(indicator, 'live', true);
    } catch (error) {
      setText(statusEl, `Refresh failed (${error.message}). Retrying automatically.`);
      setClass(statusEl, 'error', true);
      setClass(indicator, 'live', false);
      setClass(indicator, 'error', true);
    } finally {
      if (refreshBtn.disabled) refreshBtn.disabled = false;
      refreshing = null;
    }
  })();
  return refreshing;
}

function shouldIgnoreRowNavigation(event) {
  if (event.target.closest('a, button, input, select, label')) return true;
  if (window.getSelection?.()?.toString()) return true;
  return false;
}

body.addEventListener('click', event => {
  if (event.button !== 0 || shouldIgnoreRowNavigation(event)) return;
  const row = event.target.closest('tr.run-row');
  const url = row?.dataset.url;
  if (!url) return;
  if (event.metaKey || event.ctrlKey) {
    window.open(url, '_blank', 'noopener');
  } else {
    window.location.assign(url);
  }
});

body.addEventListener('auxclick', event => {
  if (event.button !== 1 || shouldIgnoreRowNavigation(event)) return;
  const row = event.target.closest('tr.run-row');
  const url = row?.dataset.url;
  if (!url) return;
  event.preventDefault();
  window.open(url, '_blank', 'noopener');
});

byId('runs-table').querySelector('thead').addEventListener('click', event => {
  const button = event.target.closest('button.th-sort[data-sort]');
  if (!button) return;
  const sortSelect = byId('sort-order');
  const requested = button.dataset.sort;
  const next = requested === 'newest' && sortSelect.value === 'newest' ? 'oldest' : requested;
  if (sortSelect.value !== next) sortSelect.value = next;
  sortRows();
});

byId('login').addEventListener('submit', async event => {
  event.preventDefault();
  const token = byId('token').value;
  byId('token').value = '';
  setText(byId('login-error'), '');
  try {
    const response = await fetch('/api/v1/session', {
      method: 'POST',
      credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token}),
    });
    if (!response.ok) {
      throw new Error(response.status === 401 ? 'That token was not accepted.' : `HTTP ${response.status}`);
    }
    setHidden(byId('login'), true);
    await refreshRuns({manual: false});
  } catch (error) {
    setText(byId('login-error'), error.message);
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
byId('refresh').addEventListener('click', () => refreshRuns({manual: true}));

for (const row of rows.values()) formatCreated(row);
sortRows();
filterRows();

// Poll silently so active runs update in place without blinking or layout jitter.
const refreshTimer = setInterval(() => {
  if (!document.hidden && !signedOut()) refreshRuns({manual: false});
}, 3000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && !signedOut()) refreshRuns({manual: false});
});
window.addEventListener('pagehide', event => {
  if (!event.persisted) clearInterval(refreshTimer);
});
