"""Versioned JSON schemas for the public observation API (OpenAPI 3.1)."""

HASH = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
CURSOR = {'type': 'string', 'maxLength': 4096}
SAFE_INTEGER = {'type': 'integer', 'minimum': 0, 'maximum': 9007199254740991}


def ref(name):
    return {'$ref': '#/components/schemas/' + name}


def object_schema(properties, required=(), **extra):
    return {'type': 'object', 'properties': properties, 'required': list(required), **extra}


def schemas():
    source = object_schema({'run_id': {'type': 'string'}, 'stream_id': {'type': 'string'},
        'stream_generation': {'type': 'string'}, 'attempt_id': {'type': 'string'},
        'sequence': {'type': 'integer', 'minimum': 1}},
        ('run_id', 'stream_id', 'stream_generation', 'attempt_id', 'sequence'))
    emission = object_schema({'id': HASH, 'definition_hash': HASH, 'value': {'type': 'number'},
        'key': {'type': 'array', 'prefixItems': [{'type': 'string'}, {'type': 'string'}, SAFE_INTEGER],
                'minItems': 3, 'maxItems': 3}}, ('id', 'definition_hash', 'value', 'key'), additionalProperties=False)
    frame = object_schema({'schema_version': {'const': 1}, 'map_revision': HASH,
        'projection_sequence': {'type': 'integer', 'minimum': 1}, 'source_cursor': CURSOR,
        'source': ref('SourceIdentity'), 'emissions': {'type': 'array', 'maxItems': 128, 'items': ref('Emission')}},
        ('schema_version', 'map_revision', 'projection_sequence', 'source_cursor', 'source', 'emissions'))
    histogram = object_schema({'edges': {'type': 'array', 'minItems': 2, 'maxItems': 513, 'items': {'type': 'number'}},
        'counts': {'type': 'array', 'minItems': 1, 'maxItems': 512, 'items': {'type': 'number', 'minimum': 0}}},
        ('edges', 'counts'), additionalProperties=False)
    event = object_schema({**source['properties'], 'schema_version': {'const': 2}, 'event': {'type': 'string'},
        'step': SAFE_INTEGER, 'catalog': HASH, 'metrics': {'type': 'object', 'additionalProperties': {'type': 'number'}},
        'measurement_status': {'type': 'object'},
        'distributions': {'type': 'object', 'additionalProperties': ref('Histogram')},
        'evaluation_id': {'type': 'string', 'pattern': '^[0-9a-f]{32}$'},
        'status': {'enum': ['complete', 'failed']},
        'source_position_known': {'type': 'boolean', 'description': 'False on failures before snapshot load; step is not a measurement position.'},
        'snapshot_sha256': HASH, 'snapshot_identity': {'type': 'object'}, 'protocol_sha256': HASH,
        'evaluation_protocol': {'type': 'object', 'description': 'Immutable numerical, data, factory, RNG, sample count and runtime protocol.'}}, (*source['required'], 'schema_version', 'event', 'step', 'catalog'))
    point = object_schema({'value': {'type': 'number'}, 'position': {'type': 'array',
        'prefixItems': [SAFE_INTEGER, {'type': 'string', 'maxLength': 128}], 'minItems': 2, 'maxItems': 2}},
        ('value', 'position'))
    envelope = object_schema({'reducer': {'const': 'envelope/v1'}, 'version': {'const': 1}, 'count': SAFE_INTEGER,
        **{key: {'oneOf': [ref('Point'), {'type': 'null'}]} for key in ('first', 'min', 'max', 'last')}},
        ('reducer', 'version', 'count', 'first', 'min', 'max', 'last'), additionalProperties=False)
    group = object_schema({'key': {'type': 'array', 'prefixItems': [{'type': 'string'}, HASH,
        {'type': 'string'}, SAFE_INTEGER], 'minItems': 4, 'maxItems': 4}, 'state': ref('EnvelopeState')}, ('key', 'state'))
    lineage = {'type': 'array', 'maxItems': 4096, 'items': object_schema({
        'attempt_id': {'type': 'string'}, 'through_step': {'oneOf': [SAFE_INTEGER, {'type': 'null'}]}},
        ('attempt_id', 'through_step'))}
    bootstrap = object_schema({'schema_version': {'const': 1}, 'run_id': {'type': 'string'},
        'map_revision': HASH, 'view_revision': HASH, 'module_sha256': HASH, 'bucket_steps': SAFE_INTEGER,
        'lineage_revision': HASH, 'lineage': lineage, 'groups': {'type': 'array', 'maxItems': 2048, 'items': ref('Group')},
        'cursor': CURSOR, 'projection_sequence': SAFE_INTEGER, 'coverage': object_schema({'complete': {'const': True}}, ('complete',)),
        'step_from': SAFE_INTEGER, 'step_to': {'oneOf': [SAFE_INTEGER, {'type': 'null'}]}},
        ('schema_version', 'run_id', 'map_revision', 'view_revision', 'module_sha256', 'bucket_steps',
         'lineage_revision', 'lineage', 'groups', 'cursor', 'projection_sequence', 'coverage'))
    page = object_schema({'cursor': CURSOR, 'has_more': {'type': 'boolean'}, 'partial_tail': {'type': 'boolean'},
        'events': {'type': 'array', 'maxItems': 1000, 'items': ref('Event')},
        'event_cursors': {'type': 'array', 'items': CURSOR},
        'frames': {'type': 'array', 'maxItems': 1000, 'items': ref('ProjectionFrame')},
        'frame_cursors': {'type': 'array', 'items': CURSOR}}, ('cursor', 'has_more', 'partial_tail'))
    catalog = object_schema({'schema_version': {'const': 1}, 'metrics': {'type': 'object',
        'additionalProperties': object_schema({'definition_hash': HASH, 'kind': {'enum': ['scalar', 'histogram']},
            'source': {'type': 'string'}, 'label': {'type': 'string'}, 'unit': {'type': 'string'}},
            ('definition_hash', 'kind', 'source'))}}, ('schema_version', 'metrics'))
    stream_frame = object_schema({'stream_id': {'type': 'string'}, 'cursor': CURSOR,
        'frame': {'oneOf': [ref('Event'), ref('ProjectionFrame')]}}, ('stream_id', 'cursor', 'frame'))
    control = object_schema({'stream_id': {'type': 'string'}, 'run': ref('Run'), 'reason': {'type': 'string'},
                            'status': {'type': 'string'}, 'job_id': HASH})
    run = object_schema({'run_id': {'type': 'string'}, 'attempt_id': {'type': 'string'}, 'name': {'type': 'string'},
        'status': {'type': 'string'}, 'steps': SAFE_INTEGER, 'total_steps': SAFE_INTEGER,
        'last_durable_step': {'oneOf': [SAFE_INTEGER, {'type': 'null'}]}, 'metrics_catalog': HASH}, ('status',))
    return {'Histogram': histogram, 'SourceIdentity': source, 'Emission': emission, 'ProjectionFrame': frame, 'Event': event,
            'Point': point, 'EnvelopeState': envelope, 'Group': group, 'Bootstrap': bootstrap,
            'Page': page, 'Catalog': catalog, 'StreamFrame': stream_frame, 'StreamControl': control,
            'Run': run, 'Error': object_schema({'error': {'type': 'string'}}, ('error',)),
            'Pending': object_schema({'status': {'enum': ['indexing', 'reducing']},
                                     'reason': {'type': 'string'}, 'job_id': HASH}, ('status',))}


def enrich(document):
    document['components']['schemas'] = schemas()
    for path, operations in document['paths'].items():
        operation = operations.get('get')
        if not operation:
            continue
        selected = 'Bootstrap' if path.endswith('/bootstrap') else 'Catalog' if path.endswith('/catalog') else 'Page' if path.endswith('/events') else 'Run' if path.endswith('/{run_id}') else None
        if selected:
            operation['responses']['200']['content'] = {'application/json': {'schema': ref(selected)}}
        operation['responses']['202']['content'] = {'application/json': {'schema': ref('Pending')}}
        for status in ('400', '401', '404'):
            operation['responses'][status]['content'] = {'application/json': {'schema': ref('Error')}}
        query = {}
        if path.endswith('/bootstrap'):
            query = {'series': {'type': 'string', 'description': 'Comma-separated 1..32 metric IDs'},
                     'bucket_steps': {'type': 'integer', 'minimum': 1}, 'step_from': SAFE_INTEGER, 'step_to': SAFE_INTEGER}
        elif path.endswith('/events'):
            query = {'cursor': CURSOR, 'stream_id': {'type': 'string'}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 1000}}
        elif path.endswith('/catalog'):
            query = {'revision': HASH}
        elif path.endswith('/stream'):
            query = {'stream_id': {'type': 'string', 'description': 'training, projection:<map revision>, evaluation:<id>, or *'}, 'cursor': CURSOR}
            operation['responses']['200']['content'] = {'text/event-stream': {'schema': {'type': 'string'}}}
            operation['x-sse-events'] = {'frame': ref('StreamFrame'), **{name: ref('StreamControl') for name in
                ('ready', 'heartbeat', 'stream_added', 'metadata', 'artifacts', 'bootstrap_ready', 'gap', 'reset_required', 'discovery_error')}}
        operation['parameters'].extend({'name': name, 'in': 'query', 'schema': schema,
                                         'required': name == 'series'} for name, schema in query.items())
    return document
