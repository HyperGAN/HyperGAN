"""Exercise the index through the real ASGI CSP, never a training process."""
import shutil
import threading
import time

import pytest
from playwright.sync_api import expect, sync_playwright
import uvicorn

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

    def publish(run_id, *, status='complete', created_at='2026-10-17T00:00:00Z', **values):
        atomic_json(root / run_id / 'manifest.json', dict(
            schema_version=1, run_id=run_id, name=run_id, status=status,
            created_at=created_at, steps=100, total_steps=1000, **values))

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


def test_index_filters_dates_and_empty_feedback(dashboard):
    page, origin, root, publish = dashboard
    publish('alpha', status='running', slug='fleet-hawk')
    publish('beta', status='completed')
    publish('gamma', status='interrupted', created_at=None)
    publish('delta', status='failed')
    page.goto(origin)
    expect(page.get_by_role('heading', name='Training runs', exact=True)).to_be_visible()
    assert page.get_by_role('columnheader', name='Created at', exact=False).count() == 1
    assert 'FID' not in page.locator('main').inner_text()
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


def test_index_sort_order_keeps_unknown_dates_last(dashboard):
    page, origin, root, publish = dashboard
    publish('alpha', created_at='2026-10-14T00:00:00Z')
    publish('beta', created_at=1792281600)  # 2026-10-18 UTC
    publish('gamma', created_at='invalid')
    page.goto(origin)
    assert visible_ids(page) == ['beta', 'alpha', 'gamma']
    page.get_by_label('Sort by').select_option('oldest')
    assert visible_ids(page) == ['alpha', 'beta', 'gamma']
    page.get_by_label('Sort by').select_option('name')
    assert visible_ids(page) == ['alpha', 'beta', 'gamma']


def test_index_polling_discovers_removes_and_resumes_runs(dashboard):
    page, origin, root, publish = dashboard
    page.goto(origin)
    expect(page.get_by_role('heading', name='No training runs yet')).to_be_visible()
    publish('new-run')
    page.clock.fast_forward(3000)
    row = page.locator('#runs-body tr[data-id="new-run"]')
    expect(row).to_be_visible()
    expect(row.get_by_role('button', name='Stop new-run')).to_be_hidden()
    page.get_by_label('Status', exact=True).select_option('running')
    page.get_by_label('Search runs').fill('new')
    publish('new-run', status='running')
    page.clock.fast_forward(3000)
    expect(row).to_be_visible()
    expect(row.get_by_role('button', name='Stop new-run')).to_be_visible()
    expect(page.get_by_label('Search runs')).to_be_focused()
    expect(page.get_by_label('Status', exact=True)).to_have_value('running')
    shutil.rmtree(root / 'new-run')
    page.clock.fast_forward(3000)
    expect(row).to_have_count(0)
    expect(page.get_by_role('heading', name='No training runs yet')).to_be_visible()


def test_index_refresh_and_action_errors_are_visible(dashboard):
    page, origin, root, publish = dashboard
    publish('active', status='running')
    page.goto(origin)
    page.route('**/api/v1/runs/active/checkpoint', lambda route: route.fulfill(
        status=400, json={'error': 'Run is not active'}))
    page.get_by_role('button', name='Checkpoint active', exact=True).click()
    expect(page.locator('#action-status')).to_have_text('Could not checkpoint active: Run is not active')
    expect(page.get_by_role('button', name='Checkpoint active', exact=True)).to_be_enabled()
    page.route('**/api/v1/runs/active/checkpoint', lambda route: route.fulfill(json={'ok': True}))
    page.get_by_role('button', name='Checkpoint active', exact=True).click()
    expect(page.locator('#action-status')).to_have_text('Checkpoint requested for active.')
    page.route('**/api/v1/runs', lambda route: route.fulfill(status=503, json={'error': 'Unavailable'}))
    page.get_by_role('button', name='Refresh', exact=True).click()
    expect(page.locator('#refresh-status')).to_contain_text('Refresh failed (HTTP 503)')
    assert visible_ids(page) == ['active']
    page.unroute('**/api/v1/runs')
    page.clock.fast_forward(3000)
    expect(page.locator('#refresh-status')).to_contain_text('Updated')


@pytest.mark.parametrize('width', [390, 768, 1280])
def test_index_responsive_layout(dashboard, width):
    page, origin, root, publish = dashboard
    publish('very-long-run-name-' * 8, status='running')
    page.set_viewport_size({'width': width, 'height': 900})
    page.goto(origin)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    for selector in ['#search', '#status-filter', '#sort-order', '.run-link', '.actions-cell']:
        box = page.locator(selector).bounding_box()
        assert box['x'] >= 0 and box['x'] + box['width'] <= width
    expect(page.get_by_label('Search runs')).to_be_visible()
    expect(page.get_by_label('Status', exact=True)).to_be_visible()
    expect(page.get_by_role('button', name='Stop very-long-run-name-', exact=False)).to_be_visible()


@pytest.mark.parametrize('dashboard', ['token'], indirect=True)
def test_token_dashboard_signs_in_before_listing_and_its_buttons_authenticate(dashboard):
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
    # The control reaches its handler with the session cookie (the run has no
    # live trainer, so the handler itself refuses it), never a 401.
    page.get_by_role('button', name='Checkpoint active', exact=True).click()
    expect(page.locator('#action-status')).to_have_text("Could not checkpoint active: Run 'active' is not active")
