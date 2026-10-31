"""Exercise the index through the real ASGI CSP, never a training process."""
import hashlib
import shutil
import threading
import time

import pytest
from playwright.sync_api import expect, sync_playwright
import uvicorn

from hypergan.image_grids import encode_png
from hypergan.run_state import atomic_json
from hypergan.web_launch import bind_loopback
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession


@pytest.fixture
def dashboard(tmp_path, request):
    auth = getattr(request, 'param', 'none')
    root = tmp_path / 'runs'
    root.mkdir()
    # A workspace can have other directories before any run is created. Keep
    # the existing pending-single-run routing heuristic out of this UI gate.
    (root / 'workspace-files').mkdir()

    def publish(run_id, *, status='complete', created_at='2026-10-17T00:00:00Z',
                steps=100, total_steps=1000, **values):
        atomic_json(root / run_id / 'manifest.json', dict(
            schema_version=1, run_id=run_id, name=run_id, status=status,
            created_at=created_at, steps=steps, total_steps=total_steps, **values))

    listener = bind_loopback()
    session = LocalSession(listener.getsockname()[1], auth=auth)
    server = uvicorn.Server(uvicorn.Config(create_app(root, session), log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, 'Dashboard did not start'
        time.sleep(.01)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={'width': 1280, 'height': 900},
                                          locale='en-US', timezone_id='America/Denver')
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('console', lambda message: errors.append(message.text)
                    if 'Content Security Policy' in message.text else None)
            page.clock.install()
            page.viewer_session = session
            yield page, session.origin, root, publish
            assert not errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), 'Dashboard did not stop'


def visible_ids(page):
    return page.locator('#runs-body tr:visible').evaluate_all('rows => rows.map(row => row.dataset.id)')


def test_index_filters_dates_theme_and_empty_feedback(dashboard):
    page, origin, root, publish = dashboard
    publish('alpha', status='running', slug='fleet-hawk')
    publish('beta', status='completed')
    publish('gamma', status='interrupted', created_at=None)
    publish('delta', status='failed')
    page.goto(origin)

    # Unified Warm Obsidian & Gold theme + masthead brand identity.
    assert page.evaluate('getComputedStyle(document.documentElement).backgroundColor') == 'rgb(18, 21, 22)'
    brand_text = page.locator('.masthead .brand').evaluate(
        "el => el.textContent.replace(/\\s+/g, ' ').trim()"
    )
    assert brand_text == 'HYPERGAN/EXPERIMENTS'
    expect(page.locator('.masthead .brand')).to_contain_text('HYPERGAN')
    expect(page.locator('.masthead .brand-detail')).to_have_text('EXPERIMENTS')

    expect(page.get_by_role('heading', name='Training runs', exact=True)).to_be_visible()
    assert page.get_by_role('columnheader', name='Created at', exact=False).count() == 1
    assert page.get_by_role('columnheader', name='FID', exact=True).count() == 1
    assert page.get_by_role('columnheader', name='Topology & Batch', exact=True).count() == 1
    expect(page.locator('#time-zone')).to_have_text('(America/Denver)')
    expect(page.locator('tr[data-id="alpha"] time')).to_have_text('Oct 16, 2026, 6:00 PM')
    expect(page.locator('tr[data-id="alpha"] time')).to_have_attribute('datetime', '2026-10-17T00:00:00.000Z')
    expect(page.locator('tr[data-id="gamma"] time')).to_have_text('Unknown')
    page.get_by_label('Status', exact=True).select_option('running')
    assert visible_ids(page) == ['alpha']
    expect(page.get_by_role('status').filter(has_text='1 of 4 runs')).to_be_visible()
    page.get_by_label('Search runs').fill(' FLEET-HAWK ')
    assert visible_ids(page) == ['alpha']
    page.get_by_label('Search runs').fill('missing')
    expect(page.get_by_role('heading', name='No matching runs')).to_be_visible()
    assert visible_ids(page) == []
    page.get_by_role('button', name='Clear filters').click()
    expect(page.get_by_label('Search runs')).to_be_focused()
    assert len(visible_ids(page)) == 4
    page.get_by_label('Status', exact=True).select_option('complete')
    assert visible_ids(page) == ['beta']
    page.get_by_label('Status', exact=True).select_option('stopped')
    assert visible_ids(page) == ['gamma']
    page.get_by_label('Status', exact=True).select_option('failed')
    assert visible_ids(page) == ['delta']


def test_index_active_only_steps_per_second_and_eta(dashboard):
    page, origin, root, publish = dashboard
    publish('active-run', status='running', steps=100, total_steps=1000,
            steps_per_second=12.4, global_batch_size=256)
    publish('done-run', status='complete', steps=100, total_steps=1000,
            steps_per_second=12.4, global_batch_size=256)
    publish('stopped-run', status='stopped', steps=100, total_steps=1000,
            steps_per_second=12.4, global_batch_size=256)
    page.goto(origin)

    active_row = page.locator('#runs-body tr[data-id="active-run"]')
    expect(active_row.locator('[data-field="speed-primary"]')).to_have_text('12.4 steps/s')
    expect(active_row.locator('[data-field="speed-sub"]')).to_be_visible()
    samples_text = active_row.locator('[data-field="speed-sub"]').inner_text()
    assert samples_text in ('3174 samples/s', '3,174 samples/s')
    expect(active_row.locator('[data-field="eta"]')).to_be_visible()
    expect(active_row.locator('[data-field="eta"]')).to_contain_text('ETA 1m 13s')

    for inactive_id in ('done-run', 'stopped-run'):
        inactive_row = page.locator(f'#runs-body tr[data-id="{inactive_id}"]')
        expect(inactive_row.locator('[data-field="speed-primary"]')).to_have_text('—')
        expect(inactive_row.locator('[data-field="speed-sub"]')).to_be_hidden()
        assert inactive_row.locator('[data-field="speed"]').inner_text().strip() == '—'
        expect(inactive_row.locator('[data-field="eta"]')).to_be_hidden()
        assert 'ETA' not in inactive_row.locator('td[data-label="Progress"]').inner_text()


def test_index_clickable_rows_and_absence_of_row_buttons(dashboard):
    page, origin, root, publish = dashboard
    publish('alpha', status='running')
    publish('beta', status='complete')
    page.goto(origin)

    assert page.locator('button[data-action], .actions-cell').count() == 0
    page.locator('#runs-body tr[data-id="alpha"] [data-field="topology"]').click()
    page.wait_for_url(f'{origin}/runs/alpha')
    expect(page.locator('#workspace')).to_be_visible()
    expect(page.locator('#run-id')).to_have_text('alpha')

    page.goto(origin)
    page.locator('#runs-body tr[data-id="beta"] [data-field="disk"]').click()
    page.wait_for_url(f'{origin}/runs/beta')
    expect(page.locator('#workspace')).to_be_visible()
    expect(page.locator('#run-id')).to_have_text('beta')


def test_index_topology_column_and_multicolumn_sorting(dashboard):
    page, origin, root, publish = dashboard
    publish(
        'alpha',
        status='complete',
        created_at='2026-10-14T00:00:00Z',
        steps=800,
        total_steps=1000,
        steps_per_second=99.0,  # inactive: must sort behind active runs on speed
        best_checkpoint={'metric': 'fid50k', 'value': 18.4, 'unit': 'FID', 'step': 800},
        execution={
            'world_size': 2,
            'nodes': 1,
            'ranks_per_node': 2,
            'global_batch_size': 256,
            'local_batch_size': 128,
            'name': 'cuda-replicated-nccl',
        },
        runtime={
            'world_size': 2,
            'device': 'cuda',
            'cuda': {
                'rank_devices': [
                    {'name': 'NVIDIA RTX A6000', 'rank': 0},
                    {'name': 'NVIDIA RTX A6000', 'rank': 1},
                ],
            },
        },
    )
    publish(
        'beta',
        status='running',
        created_at=1792281600,  # 2026-10-18 UTC
        steps=200,
        total_steps=1000,
        steps_per_second=14.2,
        best_checkpoint={'metric': 'fid50k', 'value': 9.75, 'unit': 'FID', 'step': 200},
    )
    publish(
        'gamma',
        status='running',
        created_at='invalid',
        steps=500,
        total_steps=1000,
        steps_per_second=25.0,
    )
    (root / 'alpha' / 'weights.bin').write_bytes(b'x' * 4096)
    (root / 'gamma' / 'weights.bin').write_bytes(b'x' * 16384)

    page.goto(origin)

    alpha_row = page.locator('#runs-body tr[data-id="alpha"]')
    expect(alpha_row.locator('[data-field="topology-primary"]')).to_have_text('2 GPUs · Batch 256')
    expect(alpha_row.locator('[data-field="topology-sub"]')).to_have_text('2× RTX A6000 · 128 / GPU')
    expect(alpha_row.locator('[data-field="steps"]')).to_have_text('1,600 / 2,000')
    expect(alpha_row.locator('[data-field="eta"]')).to_have_text('800 sync updates')
    expect(alpha_row.locator('[data-field="fid-primary"]')).to_have_text('18.40')
    expect(page.locator('#runs-body tr[data-id="beta"] [data-field="fid-primary"]')).to_have_text('9.75')
    expect(page.locator('#runs-body tr[data-id="gamma"] [data-field="fid-primary"]')).to_have_text('—')

    # Default sort is newest first with unknown dates last.
    expect(page.locator('th[data-sort-key="newest"]')).to_have_attribute('aria-sort', 'descending')
    assert visible_ids(page) == ['beta', 'alpha', 'gamma']

    # Clicking Created at toggles newest -> oldest while keeping unknown dates last.
    page.locator('button.th-sort[data-sort="newest"]').click()
    expect(page.locator('th[data-sort-key="newest"]')).to_have_attribute('aria-sort', 'ascending')
    assert visible_ids(page) == ['alpha', 'beta', 'gamma']

    # Clicking FID sorts lowest best FID first, missing FID last.
    page.locator('button.th-sort[data-sort="fid"]').click()
    expect(page.locator('th[data-sort-key="fid"]')).to_have_attribute('aria-sort', 'ascending')
    expect(page.locator('th[data-sort-key="newest"]')).to_have_attribute('aria-sort', 'none')
    assert visible_ids(page) == ['beta', 'alpha', 'gamma']

    # Clicking Progress sorts highest completion ratio first.
    page.locator('button.th-sort[data-sort="progress"]').click()
    expect(page.locator('th[data-sort-key="progress"]')).to_have_attribute('aria-sort', 'descending')
    assert visible_ids(page) == ['alpha', 'gamma', 'beta']

    # Clicking Speed sorts fastest active run first, inactive runs last.
    page.locator('button.th-sort[data-sort="speed"]').click()
    expect(page.locator('th[data-sort-key="speed"]')).to_have_attribute('aria-sort', 'descending')
    assert visible_ids(page) == ['gamma', 'beta', 'alpha']

    # Clicking Disk sorts largest disk footprint first.
    page.locator('button.th-sort[data-sort="disk"]').click()
    expect(page.locator('th[data-sort-key="disk"]')).to_have_attribute('aria-sort', 'descending')
    assert visible_ids(page) == ['gamma', 'alpha', 'beta']

    # Clicking Run / Name sorts alphabetically A-Z.
    page.locator('button.th-sort[data-sort="name"]').click()
    expect(page.locator('th[data-sort-key="name"]')).to_have_attribute('aria-sort', 'ascending')
    assert visible_ids(page) == ['alpha', 'beta', 'gamma']


def test_index_zero_blink_polling_preserves_thumbnail_dom_identity(dashboard):
    page, origin, root, publish = dashboard
    png = encode_png(bytes([216, 186, 125]), 1, 1, 3)
    preview_dir = root / 'live-run' / 'previews'
    preview_dir.mkdir(parents=True)
    (preview_dir / 'grid.png').write_bytes(png)
    atomic_json(preview_dir / 'index.json', {
        'previews': [{
            'step': 100,
            'sample_sequence': 1,
            'image_grid': {
                'path': 'previews/grid.png',
                'sha256': hashlib.sha256(png).hexdigest(),
            },
        }],
    })
    publish('live-run', status='running', steps=100, total_steps=1000)

    page.goto(origin)
    thumb = page.locator('#runs-body tr[data-id="live-run"] img.run-thumbnail')
    expect(thumb).to_be_visible()
    page.wait_for_function("() => document.querySelector('img.run-thumbnail')?.naturalWidth === 1")

    page.evaluate("""() => {
      const refreshBtn = document.getElementById('refresh');
      const statusEl = document.getElementById('refresh-status');
      window.__refreshDisabledSawTrue = refreshBtn.disabled;
      window.__sawRefreshingText = statusEl.textContent === 'Refreshing…';
      window.__thumbEl = document.querySelector('img.run-thumbnail');
      new MutationObserver(() => {
        if (refreshBtn.disabled) window.__refreshDisabledSawTrue = true;
      }).observe(refreshBtn, { attributes: true, attributeFilter: ['disabled'] });
      new MutationObserver(() => {
        if (statusEl.textContent === 'Refreshing…') window.__sawRefreshingText = true;
      }).observe(statusEl, { childList: true, characterData: true, subtree: true });
    }""")

    publish('live-run', status='running', steps=250, total_steps=1000)
    page.clock.fast_forward(3000)
    expect(page.locator('#runs-body tr[data-id="live-run"] [data-field="steps"]')).to_have_text('250 / 1,000')

    assert page.evaluate('window.__refreshDisabledSawTrue') is False
    assert page.evaluate('window.__sawRefreshingText') is False
    assert page.evaluate("document.querySelector('img.run-thumbnail') === window.__thumbEl") is True

    # Transition to 2-GPU active run via polling: Progress and Speed scale by world_size=2 without double-multiplying samples/s.
    publish(
        'live-run',
        status='running',
        steps=2303,
        total_steps=200000,
        steps_per_second=0.873,
        global_batch_size=256,
        execution={
            'name': 'cuda-replicated-nccl',
            'world_size': 2,
            'nodes': 1,
            'global_batch_size': 256,
            'local_batch_size': 128,
        },
        runtime={'device': 'cuda', 'world_size': 2},
    )
    page.clock.fast_forward(3000)
    live_row = page.locator('#runs-body tr[data-id="live-run"]')
    expect(live_row.locator('[data-field="steps"]')).to_have_text('4,606 / 400,000')
    expect(live_row.locator('[data-field="eta"]')).to_contain_text('2,303 sync updates · ETA')
    expect(live_row.locator('[data-field="speed-primary"]')).to_have_text('1.7 steps/s')
    expect(live_row.locator('[data-field="speed-sub"]')).to_have_text('223 samples/s')
    assert page.evaluate("document.querySelector('img.run-thumbnail') === window.__thumbEl") is True


def test_index_polling_discovers_removes_and_resumes_runs(dashboard):
    page, origin, root, publish = dashboard
    page.goto(origin)
    expect(page.get_by_role('heading', name='No training runs yet')).to_be_visible()
    publish('new-run')
    page.clock.fast_forward(3000)
    row = page.locator('#runs-body tr[data-id="new-run"]')
    expect(row).to_be_visible()
    expect(row).to_have_attribute('data-status', 'complete')
    assert page.locator('button[data-action]').count() == 0
    page.get_by_label('Status', exact=True).select_option('running')
    page.get_by_label('Search runs').fill('new')
    publish('new-run', status='running')
    page.clock.fast_forward(3000)
    expect(row).to_be_visible()
    expect(row).to_have_attribute('data-status', 'running')
    expect(row).to_have_attribute('data-active', 'true')
    expect(page.get_by_label('Search runs')).to_be_focused()
    expect(page.get_by_label('Status', exact=True)).to_have_value('running')
    shutil.rmtree(root / 'new-run')
    page.clock.fast_forward(3000)
    expect(row).to_have_count(0)
    expect(page.get_by_role('heading', name='No training runs yet')).to_be_visible()


def test_index_refresh_error_and_automatic_recovery_are_visible(dashboard):
    page, origin, root, publish = dashboard
    publish('active', status='running')
    page.goto(origin)
    assert page.locator('button[data-action]').count() == 0
    page.route('**/api/v1/runs', lambda route: route.fulfill(status=503, json={'error': 'Unavailable'}))
    page.get_by_role('button', name='Refresh', exact=True).click()
    expect(page.locator('#refresh-status')).to_contain_text('Refresh failed (HTTP 503)')
    assert visible_ids(page) == ['active']
    page.unroute('**/api/v1/runs')
    page.clock.fast_forward(3000)
    expect(page.locator('#refresh-status')).to_contain_text('Live · Updated')


@pytest.mark.parametrize('width', [360, 768, 1280])
def test_responsive_layouts_keep_controls_visible_without_horizontal_overflow(dashboard, width):
    page, origin, root, publish = dashboard
    publish(
        'very-long-run-name-' * 8,
        status='running',
        execution={'world_size': 2, 'global_batch_size': 256, 'local_batch_size': 128, 'name': 'cuda'},
    )
    page.set_viewport_size({'width': width, 'height': 900})
    page.goto(origin)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    for selector in ['#search', '#status-filter', '#sort-order', '.run-link', '[data-field="topology"]']:
        loc = page.locator(selector).first
        expect(loc).to_be_visible()
        box = loc.bounding_box()
        assert box is not None and box['x'] >= 0 and box['x'] + box['width'] <= width
    expect(page.get_by_label('Search runs')).to_be_visible()
    expect(page.get_by_label('Status', exact=True)).to_be_visible()


@pytest.mark.parametrize('dashboard', ['token'], indirect=True)
def test_token_dashboard_signs_in_before_listing_and_reaches_run_detail(dashboard):
    page, origin, root, publish = dashboard
    publish('active', status='running', slug='secret-run')
    page.goto(origin)
    # No run data reaches a signed-out browser, not even in the first HTML.
    expect(page.locator('#login')).to_be_visible()
    assert page.locator('#runs-body tr').count() == 0
    assert 'secret-run' not in page.content() and str(root) not in page.content()
    page.get_by_label('Session token').fill('wrong')
    page.get_by_role('button', name='Sign in').click()
    expect(page.locator('#login-error')).to_have_text('That token was not accepted.')
    page.get_by_label('Session token').fill(page.viewer_session._token)
    page.get_by_role('button', name='Sign in').click()
    expect(page.locator('#login')).to_be_hidden()
    expect(page.locator('tr[data-id="active"]')).to_be_visible()
    assert page.locator('button[data-action]').count() == 0

    # Clicking the row carries the authenticated session cookie to /runs/active.
    page.locator('tr[data-id="active"] [data-field="topology"]').click()
    page.wait_for_url(f'{origin}/runs/active')
    expect(page.locator('#login')).to_be_hidden()
    expect(page.locator('#workspace')).to_be_visible()
    expect(page.locator('#run-id')).to_have_text('active')
