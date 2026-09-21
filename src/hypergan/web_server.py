"""Optional loopback ASGI server; browser and external clients share one API."""
from contextlib import asynccontextmanager
import json
from pathlib import Path

from .web_dev import RELOAD_ASSET, RELOAD_SCRIPT, VERSION_ROUTE, dev_assets
from .web_files import read_artifact, read_json, _open
from .web_service import ObservationService, MapSpec, ViewSpec
from .web_session import LocalSession

STATIC_ASSETS = {'index.html', 'app.js', 'view-worker.js', 'style.css', 'styles.css', 'uplot.js', 'uplot.css'}


def create_app(run_dir, session, *, poll_seconds=.25, history_timeout=180, dev=None):
    """Construct without importing torch, running maps or computing measurements."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response, StreamingResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise RuntimeError('Local serving requires hypergan[web]') from exc
    class ClosingStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                return await super().__call__(scope, receive, send)
            finally:
                await self.body_iterator.aclose()

    service = ObservationService(run_dir, poll_seconds=poll_seconds, history_timeout=history_timeout)
    dev = dev_assets() if dev is None else dev
    if dev.active:
        import sys
        print(f'Viewer development mode: assets read per request from {dev.describe()}',
              file=sys.stderr, flush=True)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    def check_run(request):
        if request.path_params.get('run_id') != service.run_id:
            raise FileNotFoundError('Unknown run identity')

    async def session_exchange(request):
        if request.headers.get('content-type', '').split(';')[0] != 'application/json':
            return JSONResponse({'error': 'Expected application/json'}, status_code=415)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                return JSONResponse({'error': 'Session body exceeds 4096 bytes'}, status_code=413)
        try:
            value = json.loads(raw)
            if type(value) is not dict or set(value) != {'token'}:
                raise ValueError('Expected token object')
            cookie = session.exchange(value['token'])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({'error': 'Invalid or expired viewer credential'}, status_code=401)
        result = JSONResponse({'authenticated': True})
        # Cookies are keyed by host, so the proxied and the direct authority
        # already hold separate ones. Only the https channel may carry Secure:
        # a browser discards a Secure cookie offered over plain-HTTP loopback.
        secure = session.cookie_secure and request.scope.get('state', {}).get('viewer_channel') == 'public'
        result.set_cookie(session.cookie_name, cookie, httponly=True, samesite='strict',
                          secure=secure, max_age=86400, path='/')
        return result

    async def capabilities(request):
        from .metrics_reducer import descriptor
        return JSONResponse({'schema_version': 1, 'api_version': 'v1', 'run_id': service.run_id,
                             'status': service.manifest.get('status'), 'server_instance_id': session.instance_id,
                             'auth_mode': session.auth_mode, 'public_origin': session.public_origin,
                             'transports': ['sse'], 'controls': [], 'reducer': descriptor(),
                             'limits': {'subscribers': 32, 'queue_bytes': 1048576, 'groups': 2048,
                                        'bootstrap_bytes': 1048576, 'stream_count': 64,
                                        'history_seconds': service.history_timeout}})

    async def run(request):
        check_run(request)
        return JSONResponse(service.public_manifest())

    async def catalog(request):
        check_run(request)
        from .metrics import read_catalog
        import asyncio
        revision = request.query_params.get('revision') or service.manifest.get('metrics_catalog')
        result = await asyncio.to_thread(read_catalog, service.root, revision,
            _open_file=lambda path: _open(service.root, path.relative_to(service.root).as_posix()))
        return JSONResponse(result)

    async def events(request):
        check_run(request)
        import asyncio
        cursor = request.query_params.get('cursor')
        stream_id = request.query_params.get('stream_id', 'training')
        limit = int(request.query_params.get('limit', '100'))
        if not 1 <= limit <= 1000:
            raise ValueError('limit must be in 1..1000')
        _, _, result = await asyncio.to_thread(service.page, stream_id, cursor, limit=limit)
        return JSONResponse(result)

    async def views(request):
        check_run(request)
        mapping = MapSpec()
        view = ViewSpec(mapping.revision)
        available = ['projection:' + mapping.revision in service.streams]
        return JSONResponse({'schema_version': 1, 'map_revision': mapping.revision,
                             'view_revision': view.revision, 'status': 'available' if available[0] else 'missing',
                             'descriptor': view.descriptor(), 'discovery_error': service.discovery_error,
                             'streams': [{'stream_id': s.stream_id, 'cursor': s.cursor,
                                          'caught_up': s.caught_up, 'error': s.error} for s in service.streams.values()]})

    async def bootstrap(request):
        check_run(request)
        params = request.query_params
        series = params.get('series', '').split(',')
        kwargs = dict(series=series, bucket_steps=int(params.get('bucket_steps', '1')),
                      step_from=int(params.get('step_from', '0')))
        if 'step_to' in params:
            kwargs['step_to'] = int(params['step_to'])
        try:
            job = service.bootstrap(request.path_params['map_revision'], **kwargs)
        except LookupError as exc:
            return JSONResponse({'status': 'indexing', 'reason': str(exc)}, status_code=202)
        if job.error:
            return JSONResponse({'status': 'error', 'error': job.error, 'job_id': job.key}, status_code=422)
        if job.result is None:
            return JSONResponse({'status': 'reducing', 'job_id': job.key}, status_code=202)
        return JSONResponse(job.result)

    async def stream(request):
        if 'run_id' in request.path_params:
            check_run(request)
        stream_id = request.query_params.get('stream_id', '*')
        cursor = request.query_params.get('cursor')
        # Validate selectors before StreamingResponse sends its 200 headers.
        temporary, _ = service.subscribe(stream_id)
        service.subscribers.remove(temporary)
        if cursor is not None:
            from .web_service import cursor_offset
            cursor_offset(cursor)
        return ClosingStream(service.events(stream_id, cursor), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    async def artifacts(request):
        check_run(request)
        import asyncio
        await service.refresh_artifacts()
        artifact_id = request.path_params.get('artifact_id')
        if artifact_id is None:
            return JSONResponse(service.artifact_index())
        record = service.artifacts.get(artifact_id)
        if record is None:
            raise FileNotFoundError('Artifact is not indexed')
        if record.get('status') == 'unavailable':
            raise ValueError(record['reason'])
        data = await asyncio.to_thread(read_artifact, service.root, record)
        if record.get('media_type') == 'image/png' and record.get('modality') == 'image':
            from .image_grids import inspect_png
            await asyncio.to_thread(inspect_png, data)
            return Response(data, media_type='image/png',
                            headers={'Content-Disposition': 'inline; filename="grid.png"'})
        if record.get('media_type') == 'application/json':
            return Response(data, media_type='application/json',
                            headers={'Content-Disposition': 'attachment; filename="samples.json"'})
        return Response(data, media_type='application/octet-stream',
                        headers={'Content-Disposition': 'attachment; filename="artifact.bin"'})

    async def static(request):
        asset = request.path_params.get('asset', 'index.html')
        if asset not in STATIC_ASSETS:
            raise FileNotFoundError('Unknown static asset')
        resource = dev.web_asset(asset)
        if not resource.is_file():
            if asset == 'index.html':
                return Response('HyperGAN viewer assets are not installed.', media_type='text/plain', status_code=503)
            raise FileNotFoundError('Unknown static asset')
        media = 'text/html' if asset.endswith('.html') else 'text/css' if asset.endswith('.css') else 'text/javascript'
        body = resource.read_bytes()
        if dev.active and asset == 'index.html':
            body = body.replace(b'</body>', f'<script type="module" src="{RELOAD_ASSET}"></script>\n  </body>'.encode(), 1)
        return Response(body, media_type=media, headers=dev.headers)

    async def reducer_asset(request):
        asset = request.path_params['asset']
        if asset not in {'host.js', 'client.js', 'worker.js', 'reducer.wasm', 'reducer.json'}:
            raise FileNotFoundError('Unknown reducer asset')
        media = 'application/wasm' if asset.endswith('.wasm') else 'application/json' if asset.endswith('.json') else 'text/javascript'
        return Response(dev.reducer_asset(asset).read_bytes(), media_type=media, headers=dev.headers)

    async def dev_version(request):
        """Identity of the served browser assets, so a dev page can reload itself."""
        return JSONResponse({'version': dev.version(sorted(STATIC_ASSETS))}, headers=dev.headers)

    async def dev_reload(request):
        return Response(RELOAD_SCRIPT, media_type='text/javascript', headers=dev.headers)

    async def openapi(request):
        return JSONResponse(openapi_schema(cookie_name=session.cookie_name, auth_mode=session.auth_mode))

    async def error_handler(request, exc):
        code = 404 if isinstance(exc, FileNotFoundError) else 400
        return JSONResponse({'error': str(exc)}, status_code=code)

    routes = [Route('/', static), Route('/assets/{asset}', static), Route('/reducers/{asset}', reducer_asset),
              Route('/api/v1/session', session_exchange, methods=['POST']),
              Route('/api/v1/capabilities', capabilities), Route('/api/v1/openapi.json', openapi),
              Route('/api/v1/stream', stream), Route('/api/v1/runs/{run_id}', run),
              Route('/api/v1/runs/{run_id}/metrics/catalog', catalog),
              Route('/api/v1/runs/{run_id}/events', events),
              Route('/api/v1/runs/{run_id}/views', views),
              Route('/api/v1/runs/{run_id}/views/{map_revision}/bootstrap', bootstrap),
              Route('/api/v1/runs/{run_id}/stream', stream),
              Route('/api/v1/runs/{run_id}/artifacts', artifacts),
              Route('/api/v1/runs/{run_id}/artifacts/{artifact_id}', artifacts)]
    if dev.active:
        routes.extend([Route(VERSION_ROUTE, dev_version), Route(RELOAD_ASSET, dev_reload)])
    app = Starlette(routes=routes, lifespan=lifespan,
                    exception_handlers={ValueError: error_handler, FileNotFoundError: error_handler})
    app.state.observations = service

    class AuthenticatedApp:
        state = app.state
        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await app(scope, receive, send)
            request = Request(scope, receive=receive)
            # Forwarded headers are read only when a public origin is configured;
            # `match_request` ignores them otherwise, so nothing here is trusted
            # by default. The channel decides the session cookie's Secure flag.
            channel = session.match_request(request.headers.get('host'), request.headers.get('origin'),
                                            forwarded_proto=request.headers.get('x-forwarded-proto'),
                                            forwarded_host=request.headers.get('x-forwarded-host'))
            if channel is None:
                return await JSONResponse({'error': 'Invalid local Host or Origin'}, status_code=403)(scope, receive, send)
            scope.setdefault('state', {})['viewer_channel'] = channel
            path = scope['path']
            public = (path == '/' or path.startswith('/assets/') or path.startswith('/reducers/')
                      or path == '/api/v1/session' or (dev.active and path.startswith('/dev/')))
            if not public and not session.authenticated(authorization=request.headers.get('authorization'),
                                                       cookie=request.cookies.get(session.cookie_name)):
                return await JSONResponse({'error': 'Viewer credential required'}, status_code=401)(scope, receive, send)
            async def secure_send(message):
                if message['type'] == 'http.response.start':
                    headers = list(message.get('headers', []))
                    present = {name.lower() for name, _ in headers}
                    headers.extend([(b'x-content-type-options', b'nosniff'),
                                    (b'referrer-policy', b'no-referrer'),
                                    # Every directive is 'self' and every URL the page
                                    # requests is relative, so 'self' resolves to the
                                    # public origin behind a proxy with no host pinned
                                    # here and nothing widened for it.
                                    (b'content-security-policy', b"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'; form-action 'self'; frame-ancestors 'none'")])
                    # A route that already stated its own policy keeps exactly one value.
                    if b'cache-control' not in present:
                        headers.append((b'cache-control', b'no-store'))
                    message = dict(message, headers=headers)
                if message['type'] == 'http.response.body' and scope['path'].endswith('/stream'):
                    import asyncio
                    try:
                        await asyncio.wait_for(send(message), timeout=5)
                    except asyncio.TimeoutError as exc:
                        raise OSError('Stream client exceeded the send deadline') from exc
                else:
                    await send(message)
            return await app(scope, receive, secure_send)
    return AuthenticatedApp()


def openapi_schema(*, auth_mode='token', cookie_name=LocalSession.cookie_name):
    """Versioned machine-readable route contract; rich event schemas are separate."""
    paths = {}
    descriptions = {
        '/capabilities': 'Authenticated server identity and resource limits',
        '/runs/{run_id}': 'Current run manifest',
        '/runs/{run_id}/metrics/catalog': 'Immutable metric definition catalog',
        '/runs/{run_id}/events': 'Bounded raw source or projection frame page',
        '/runs/{run_id}/views': 'Available view descriptors and stream watermarks',
        '/runs/{run_id}/views/{map_revision}/bootstrap': 'Historical shared-reducer state through a fixed projection cursor; 202 while indexing/reducing',
        '/runs/{run_id}/stream': 'SSE frame/control stream; reconnect only from last applied cursor',
        '/stream': 'SSE run discovery and future stream registration',
        '/runs/{run_id}/artifacts': 'Explicit artifact index',
        '/runs/{run_id}/artifacts/{artifact_id}': 'Bounded digest-verified indexed artifact download',
    }
    for route, description in descriptions.items():
        params = [{'name': name, 'in': 'path', 'required': True, 'schema': {'type': 'string'}}
                  for name in ('run_id', 'map_revision', 'artifact_id') if '{' + name + '}' in route]
        paths['/api/v1' + route] = {'get': {'description': description, 'parameters': params,
            'security': [{'bearerAuth': []}, {'cookieAuth': []}],
            'responses': {'200': {'description': 'Success'}, '202': {'description': 'Background indexing or reduction'},
                          '400': {'description': 'Invalid query or cursor'}, '401': {'description': 'Credential required'},
                          '404': {'description': 'Run, projection or artifact not available'}}}}
    paths['/api/v1/session'] = {'post': {'description': 'Exchange local bearer for HttpOnly SameSite session cookie',
        'requestBody': {'required': True, 'content': {'application/json': {'schema': {'type': 'object', 'required': ['token'],
                        'additionalProperties': False, 'properties': {'token': {'type': 'string'}}}}}},
        'responses': {'200': {'description': 'Authenticated'}, '401': {'description': 'Invalid credential'}}}}
    if auth_mode == 'none':
        for path in paths.values():
            for operation in path.values():
                operation.pop('security', None)
    from .web_schema import enrich
    return enrich({'openapi': '3.1.0', 'info': {'title': 'HyperGAN local observation API', 'version': '1.0.0'},
            'paths': paths, 'components': {'securitySchemes': {
                'bearerAuth': {'type': 'http', 'scheme': 'bearer'},
                'cookieAuth': {'type': 'apiKey', 'in': 'cookie', 'name': cookie_name}}}})


def run_socket(run_dir, bound_socket, session, *, history_seconds=180):
    """Serve a prebound loopback socket; the caller owns credentials/lifecycle."""
    import uvicorn
    app = create_app(run_dir, session, history_timeout=history_seconds)
    server = uvicorn.Server(uvicorn.Config(app, host=session.bind_host, port=bound_socket.getsockname()[1],
                                        access_log=False, log_level='warning'))
    server.run(sockets=[bound_socket])
