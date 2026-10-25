"""Unit tests for multi-node distributed topology resolution, profile contracts, and device mapping.

Covers:
1. Profile resolution of canonical multi-node keys: nodes, node_rank, ranks_per_node, rendezvous.
2. Invariant validation: world_size % nodes == 0, 0 <= node_rank < nodes, ranks_per_node == world_size // nodes.
3. Strict error rejection: non-dividing topologies, out-of-bound node_rank, invalid types.
4. Rendezvous URI parsing and validation: tcp://<host>:<port>, file://<path>, and rejection of unsupported schemes.
5. Local device remapping formulas: local_rank = global_rank % ranks_per_node, node_rank = global_rank // ranks_per_node.
6. Batch geometry resolution and partitioning across nodes and accumulation steps.
7. cpu-single invariant preservation (nodes=1, node_rank=0, ranks_per_node=1, and rejection of tcp:// rendezvous).
"""
from __future__ import annotations

import pytest

from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile


def make_raw_profile(name='cpu-replicated-gloo', **execution):
    return {'schema_version': 1, 'execution': {'name': name, **execution}}


def make_cfg(device='cpu', batch_size=16):
    return resolve_config({'training': {'device': device, 'batch_size': batch_size}})


# ==============================================================================
# 1. Multi-Node Topology Resolution & Invariants
# ==============================================================================

class TestMultiNodeProfileResolution:
    """Validate parsing, derivation, and invariants of multi-node profile keys."""

    def test_default_multi_node_keys_when_omitted(self):
        """When multi-node keys are omitted, profile applies canonical GLOSSARY defaults."""
        cfg = make_cfg(batch_size=8)
        # cpu-single
        single = resolve_execution_profile(make_raw_profile('cpu-single'), cfg)
        assert single['execution']['nodes'] == 1
        assert single['execution']['node_rank'] == 0
        assert single['execution']['ranks_per_node'] == 1
        assert single['execution']['rendezvous'] == 'file://'

        # cpu-replicated-gloo
        replicated = resolve_execution_profile(make_raw_profile('cpu-replicated-gloo', world_size=2), cfg)
        assert replicated['execution']['nodes'] == 1
        assert replicated['execution']['node_rank'] == 0
        assert replicated['execution']['ranks_per_node'] == 2
        assert replicated['execution']['rendezvous'] == 'file://'

    @pytest.mark.parametrize('world_size,nodes,node_rank,expected_rpn', [
        (2, 2, 0, 1),
        (2, 2, 1, 1),
        (4, 2, 0, 2),
        (4, 2, 1, 2),
        (4, 4, 3, 1),
        (8, 2, 1, 4),
        (8, 4, 2, 2),
        (8, 8, 7, 1),
        (16, 4, 3, 4),
        (64, 8, 5, 8),
    ])
    def test_valid_topologies_parametrized(self, world_size, nodes, node_rank, expected_rpn):
        """Profile resolves valid topologies and derives ranks_per_node = world_size // nodes."""
        cfg = make_cfg(batch_size=world_size * 4)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=world_size, nodes=nodes, node_rank=node_rank)
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['world_size'] == world_size
        assert exec_tab['nodes'] == nodes
        assert exec_tab['node_rank'] == node_rank
        assert exec_tab['ranks_per_node'] == expected_rpn

    def test_explicit_ranks_per_node_matching_derived(self):
        """Passing explicit ranks_per_node matching world_size // nodes is accepted."""
        cfg = make_cfg(batch_size=16)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=4, nodes=2, node_rank=0, ranks_per_node=2)
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['ranks_per_node'] == 2

    def test_explicit_ranks_per_node_mismatch_rejected(self):
        """Passing explicit ranks_per_node differing from world_size // nodes is rejected."""
        cfg = make_cfg(batch_size=16)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=4, nodes=2, node_rank=0, ranks_per_node=3)
        with pytest.raises(ValueError, match=r'execution\.ranks_per_node \(3\) must equal world_size // nodes \(2\)'):
            resolve_execution_profile(raw, cfg)

    @pytest.mark.parametrize('world_size,nodes', [
        (3, 2),   # 3 not divisible by 2
        (5, 2),   # 5 not divisible by 2
        (7, 3),   # 7 not divisible by 3
        (10, 3),  # 10 not divisible by 3
    ])
    def test_nodes_not_dividing_world_size_rejected(self, world_size, nodes):
        """Rejects configurations where world_size does not divide evenly by nodes."""
        cfg = make_cfg(batch_size=world_size * 4)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=world_size, nodes=nodes, node_rank=0)
        with pytest.raises(ValueError, match=r'must divide evenly across execution\.nodes'):
            resolve_execution_profile(raw, cfg)

    def test_nodes_greater_than_world_size_rejected(self):
        """Rejects configurations where nodes > world_size."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, nodes=4, node_rank=0)
        with pytest.raises(ValueError, match=r'execution\.nodes cannot exceed execution\.world_size'):
            resolve_execution_profile(raw, cfg)

    @pytest.mark.parametrize('nodes,node_rank', [
        (2, -1),
        (2, 2),
        (2, 3),
        (4, 4),
        (4, 10),
    ])
    def test_node_rank_out_of_bounds_rejected(self, nodes, node_rank):
        """Rejects node_rank < 0 or node_rank >= nodes."""
        cfg = make_cfg(batch_size=16)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=4, nodes=nodes, node_rank=node_rank)
        with pytest.raises(ValueError, match=r'node_rank'):
            resolve_execution_profile(raw, cfg)

    def test_cpu_single_invariants(self):
        """cpu-single strictly enforces world_size=1, nodes=1, node_rank=0, and rejects tcp rendezvous."""
        cfg = make_cfg(batch_size=8)
        with pytest.raises(ValueError, match=r'cpu-single requires nodes=1'):
            resolve_execution_profile(make_raw_profile('cpu-single', nodes=2), cfg)
        with pytest.raises(ValueError, match=r'cpu-single requires world_size=1'):
            resolve_execution_profile(make_raw_profile('cpu-single', world_size=2), cfg)
        with pytest.raises(ValueError, match=r'cpu-single requires file:// rendezvous'):
            resolve_execution_profile(make_raw_profile('cpu-single', rendezvous='tcp://127.0.0.1:29500'), cfg)

    @pytest.mark.parametrize('invalid_val', [True, False, 1.5, '2', 0, -1])
    def test_nodes_type_and_bounds_validation(self, invalid_val):
        """Rejects non-positive integer types for nodes."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, nodes=invalid_val)
        with pytest.raises(ValueError, match=r'execution\.nodes'):
            resolve_execution_profile(raw, cfg)

    @pytest.mark.parametrize('invalid_val', [True, False, 0.5, '0', -1])
    def test_node_rank_type_validation(self, invalid_val):
        """Rejects non-integer types for node_rank."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, nodes=2, node_rank=invalid_val)
        with pytest.raises(ValueError, match=r'execution\.node_rank'):
            resolve_execution_profile(raw, cfg)


# ==============================================================================
# 2. Rendezvous URI Validation
# ==============================================================================

class TestRendezvousURIValidation:
    """Validate acceptance of file:// and tcp:// URIs and rejection of malformed URIs."""

    @pytest.mark.parametrize('uri', [
        'tcp://127.0.0.1:29500',
        'tcp://localhost:12345',
        'tcp://192.168.1.100:8080',
        'tcp://coordinator.internal:65535',
        'file:///tmp/hypergan-rdzv',
        'file:///var/run/hypergan/rdzv-store',
    ])
    def test_valid_rendezvous_uris(self, uri):
        """Valid tcp:// and file:// URIs are accepted verbatim."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, nodes=2, node_rank=0, rendezvous=uri)
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['rendezvous'] == uri

    @pytest.mark.parametrize('invalid_scheme', [
        'http://127.0.0.1:29500',
        'https://127.0.0.1:29500',
        'udp://127.0.0.1:29500',
        'ftp://127.0.0.1:29500',
        'redis://127.0.0.1:6379',
    ])
    def test_unsupported_rendezvous_schemes_rejected(self, invalid_scheme):
        """Rejects non-file/non-tcp URI schemes."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, rendezvous=invalid_scheme)
        with pytest.raises(ValueError, match=r'execution\.rendezvous must be a file:// or tcp:// URI'):
            resolve_execution_profile(raw, cfg)

    @pytest.mark.parametrize('malformed_uri,match', [
        ('tcp://', 'must be formatted as tcp://<host>:<port>'),
        ('tcp://localhost', 'must be formatted as tcp://<host>:<port>'),
        ('tcp://:29500', 'must have a non-empty host'),
        ('tcp://localhost:abc', 'port must be an integer'),
        ('tcp://localhost:0', 'port must be between 1 and 65535'),
        ('tcp://localhost:65536', 'port must be between 1 and 65535'),
        ('tcp://localhost:-1', 'port must be between 1 and 65535'),
        ('', 'must be a non-empty string'),
        (12345, 'must be a non-empty string'),
    ])
    def test_malformed_tcp_uris_rejected(self, malformed_uri, match):
        """Rejects malformed host/port components in tcp:// URIs."""
        cfg = make_cfg(batch_size=8)
        raw = make_raw_profile('cpu-replicated-gloo', world_size=2, rendezvous=malformed_uri)
        with pytest.raises(ValueError, match=match):
            resolve_execution_profile(raw, cfg)


# ==============================================================================
# 3. Local Device Index Mapping & Batch Geometry
# ==============================================================================

class TestDeviceMappingAndBatchGeometry:
    """Validate local rank formulas and batch partitioning across multi-node topologies."""

    @pytest.mark.parametrize('world_size,nodes,ranks_per_node,global_rank,expected_node_rank,expected_local_rank', [
        # 2 nodes, 1 rank/node
        (2, 2, 1, 0, 0, 0),
        (2, 2, 1, 1, 1, 0),
        # 2 nodes, 2 ranks/node
        (4, 2, 2, 0, 0, 0),
        (4, 2, 2, 1, 0, 1),
        (4, 2, 2, 2, 1, 0),
        (4, 2, 2, 3, 1, 1),
        # 4 nodes, 2 ranks/node
        (8, 4, 2, 0, 0, 0),
        (8, 4, 2, 1, 0, 1),
        (8, 4, 2, 4, 2, 0),
        (8, 4, 2, 5, 2, 1),
        (8, 4, 2, 7, 3, 1),
    ])
    def test_local_device_remapping_formulas(
        self, world_size, nodes, ranks_per_node, global_rank, expected_node_rank, expected_local_rank
    ):
        """Verifies local_rank = global_rank % ranks_per_node and node_rank = global_rank // ranks_per_node."""
        derived_local_rank = global_rank % ranks_per_node
        derived_node_rank = global_rank // ranks_per_node
        assert derived_local_rank == expected_local_rank
        assert derived_node_rank == expected_node_rank

        # Invariant: round-trip reconstructing global_rank
        reconstructed_global = derived_node_rank * ranks_per_node + derived_local_rank
        assert reconstructed_global == global_rank

    def test_batch_geometry_partitioning(self):
        """Verifies global_batch_size, local_batch_size, and microbatch_size resolution."""
        cfg = make_cfg(batch_size=32)
        raw = make_raw_profile(
            'cpu-replicated-gloo',
            world_size=4,
            nodes=2,
            node_rank=0,
            accumulation_steps=2,
        )
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['global_batch_size'] == 32
        assert exec_tab['local_batch_size'] == 8      # 32 // 4
        assert exec_tab['microbatch_size'] == 4      # 8 // 2
        assert exec_tab['ranks_per_node'] == 2
        # Aggregate samples processed per node = local_batch_size * ranks_per_node = 16
        assert exec_tab['local_batch_size'] * exec_tab['ranks_per_node'] == 32 // 2
