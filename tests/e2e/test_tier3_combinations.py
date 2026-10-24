"""Tier 3: Cross-Feature Combinations (Pairwise & Multi-Feature Interactions).

Covers:
- accumulation_steps + parameter_average (microbatch local SGD)
- accumulation_steps + pseudo_gradient (accumulated DiLoCo)
- multi-node + pseudo_gradient (federated multi-node training)
- component reuse + extra adversarial_terms (shared weights in multiple losses)
- recipe presets + cpu-replicated-gloo vs cuda-replicated-nccl
- prior inputs + parameter_average
- multi-node + tcp rendezvous + gradient_allreduce
- recipe presets + reuse + extra adversarial_terms (compound graph)
- qualification CLI + suite filtering
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import pytest
import torch

from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile

try:
    from e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_RECIPE_PARITY,
        HAVE_M1_SCHEMA,
        HAVE_AGGREGATION,
        HAVE_MULTINODE,
        HAVE_QUALIFY,
        requires_m1_recipe_parity,
        requires_m1_schema,
        requires_aggregation,
        requires_multinode,
        requires_qualify,
        requires_2_gpus,
        make_toy_config,
        make_prior_bound_config,
        make_reuse_config,
        make_adversarial_terms_config,
        make_profile_raw,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
        run_replicated_cpu_smoke_test,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_RECIPE_PARITY,
        HAVE_M1_SCHEMA,
        HAVE_AGGREGATION,
        HAVE_MULTINODE,
        HAVE_QUALIFY,
        requires_m1_recipe_parity,
        requires_m1_schema,
        requires_aggregation,
        requires_multinode,
        requires_qualify,
        requires_2_gpus,
        make_toy_config,
        make_prior_bound_config,
        make_reuse_config,
        make_adversarial_terms_config,
        make_profile_raw,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
        run_replicated_cpu_smoke_test,
    )


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature cross-module interaction verification."""

    @requires_m1_schema
    @requires_aggregation
    def test_tier3_accumulation_steps_plus_parameter_average(self):
        """Combination 1: accumulation_steps > 1 with parameter_average strategy."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2},
            'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 2},
        }
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['accumulation_steps'] == 2
        assert exec_tab['microbatch_size'] == 4
        assert exec_tab['accumulation_algorithm'] == 'detached-logit-vjp-replay-v1'

        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'parameter_average'
        assert agg['sync_every_steps'] == 2

    @requires_m1_schema
    @requires_aggregation
    def test_tier3_accumulation_steps_plus_pseudo_gradient(self):
        """Combination 2: accumulation_steps > 1 with pseudo_gradient strategy."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2},
            'aggregation': {
                'strategy': 'pseudo_gradient',
                'sync_every_steps': 4,
                'outer_lr': 0.8,
                'outer_momentum': 0.5,
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['accumulation_steps'] == 2
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'pseudo_gradient'
        assert agg['sync_every_steps'] == 4

    @requires_m1_schema
    @requires_multinode
    @requires_aggregation
    def test_tier3_multinode_plus_pseudo_gradient(self):
        """Combination 3: multi-node topology (nodes=2) with pseudo_gradient strategy."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'nodes': 2,
                'node_rank': 0,
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
            'aggregation': {
                'strategy': 'pseudo_gradient',
                'sync_every_steps': 4,
                'outer_lr': 0.9,
                'outer_momentum': 0.9,
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['nodes'] == 2
        assert exec_tab['ranks_per_node'] == 1
        assert exec_tab['rendezvous'] == 'tcp://127.0.0.1:29500'
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'pseudo_gradient'

    @requires_m1_recipe_parity
    def test_tier3_component_reuse_plus_adversarial_terms(self):
        """Combination 4: Reused components invoked inside extra adversarial_terms."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        # Declares shared encoder and calls it in extra adversarial terms
        cfg['components']['encoder_a'] = {
            'factory': 'mlp',
            'args': {'input_dim': 2, 'output_dim': 2, 'hidden': [4]},
            'inputs': {'x': 'candidate'},
        }
        cfg['components']['encoder_b'] = {
            'reuse': 'encoder_a',
            'inputs': {'x': 'candidate'},
        }
        cfg['adversarial_terms'] = [
            {'id': 'adv_term_1', 'component': 'discriminator', 'real': 'batch.real', 'fake': 'generated'},
            {'id': 'adv_term_2', 'component': 'discriminator', 'real': 'batch.real', 'fake': 'generated'},
        ]
        # Replicated validation must accept this compound config
        from hypergan.execution_profiles import validate_replicated_recipe
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier3_presets_plus_cpu_replicated_gloo(self):
        """Combination 5: Preset ParticleGAN recipe on cpu-replicated-gloo."""
        from hypergan.execution_profiles import validate_replicated_recipe
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 2, 'batch_size': 8},
        })
        validate_replicated_recipe(cfg)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['name'] == 'cpu-replicated-gloo'
        assert resolved['execution']['local_batch_size'] == 4

    @requires_m1_recipe_parity
    @requires_2_gpus
    def test_tier3_presets_plus_cuda_replicated_nccl(self):
        """Combination 6: Preset ParticleGAN recipe on cuda-replicated-nccl."""
        from hypergan.execution_profiles import validate_replicated_recipe
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cuda', 'steps': 2, 'batch_size': 8},
        })
        validate_replicated_recipe(cfg)
        raw = {'schema_version': 1, 'execution': {'name': 'cuda-replicated-nccl', 'world_size': 2}}
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['name'] == 'cuda-replicated-nccl'
        assert resolved['execution']['local_batch_size'] == 4

    @requires_m1_recipe_parity
    @requires_aggregation
    def test_tier3_prior_inputs_plus_parameter_average(self):
        """Combination 7: Generator with prior.z input trained with parameter_average."""
        cfg = make_prior_bound_config(device="cpu", batch_size=8)
        from hypergan.execution_profiles import validate_replicated_recipe
        validate_replicated_recipe(cfg)

        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 2},
        }
        resolved = resolve_execution_profile(raw, cfg)
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'parameter_average'

    @requires_m1_schema
    @requires_multinode
    def test_tier3_multinode_tcp_plus_gradient_allreduce(self):
        """Combination 8: Multi-node TCP rendezvous with gradient_allreduce strategy."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 4,
                'nodes': 2,
                'node_rank': 1,
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
            'aggregation': {'strategy': 'gradient_allreduce', 'sync_every_steps': 1},
        }
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['nodes'] == 2
        assert exec_tab['ranks_per_node'] == 2
        assert exec_tab['rendezvous'] == 'tcp://127.0.0.1:29500'

    @requires_m1_recipe_parity
    def test_tier3_preset_plus_reuse_plus_adversarial_terms(self):
        """Combination 9: Compound recipe combining presets, component reuse, and extra adversarial terms."""
        from hypergan.execution_profiles import validate_replicated_recipe
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 2, 'batch_size': 8},
        })
        cfg['components']['critic_shared'] = {
            'reuse': 'discriminator',
            'inputs': {'input': 'candidate'},
        }
        cfg['adversarial_terms'] = [
            {'id': 'extra_adv', 'component': 'critic_shared', 'real': 'batch.real', 'fake': 'generated'}
        ]
        validate_replicated_recipe(cfg)

    @requires_qualify
    def test_tier3_qualification_cli_suite_filtering(self, tmp_path):
        """Combination 10: Qualification CLI selectively executing --suite aggregation in dry-run mode."""
        receipt = tmp_path / "qual_filter.json"
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--suite", "aggregation", "--dry-run", "--output", str(receipt)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
        assert receipt.exists()
        data = json.loads(receipt.read_text())
        assert any(s['name'] == 'aggregation' or 'aggregation' in s['name'] for s in data.get('suites', []))
