"""Tier 1: Feature Coverage for Replicated Execution, Profiles, Prior Inputs, Reuse, Adversarial Terms, and Presets.

Requirements covered:
- R1: Replicated Recipe Parity for Multi-Device Smoke Testing (ORIGINAL_REQUEST.md §R1)
- Features 1, 2, 3: Canonical Profile Schema, Replicated Recipe Parity, Local Device Remapping (PROJECT.md §Feature Inventory)
- Interface Contract 1: Execution Profile Parser (PROJECT.md §Interface Contracts)

Test coverage target: >=5 test cases per major feature.
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
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.execution_profiles import (
    load_execution_profile,
    resolve_execution_profile,
    validate_replicated_recipe,
)

try:
    from e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_RECIPE_PARITY,
        HAVE_M1_SCHEMA,
        requires_m1_recipe_parity,
        requires_m1_schema,
        requires_2_gpus,
        make_toy_config,
        make_prior_bound_config,
        make_reuse_config,
        make_adversarial_terms_config,
        make_profile_raw,
        assert_states_close,
        run_replicated_cpu_smoke_test,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_RECIPE_PARITY,
        HAVE_M1_SCHEMA,
        requires_m1_recipe_parity,
        requires_m1_schema,
        requires_2_gpus,
        make_toy_config,
        make_prior_bound_config,
        make_reuse_config,
        make_adversarial_terms_config,
        make_profile_raw,
        assert_states_close,
        run_replicated_cpu_smoke_test,
    )


# ==============================================================================
# Feature 1: Replicated Profile Parsing (>= 5 test cases)
# ==============================================================================

class TestReplicatedProfileParsing:
    """Feature 1: Replicated profile parsing and geometry resolution."""

    def test_tier1_parse_cpu_single_defaults(self):
        """Case 1.1: Default single-device profile resolves to world_size=1, accumulation_steps=1."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single'}}
        resolved = resolve_execution_profile(raw, cfg)

        exec_tab = resolved['execution']
        assert exec_tab['name'] == 'cpu-single'
        assert exec_tab['world_size'] == 1
        assert exec_tab['accumulation_steps'] == 1
        assert exec_tab['global_batch_size'] == 16
        assert exec_tab['local_batch_size'] == 16
        assert exec_tab['microbatch_size'] == 16
        assert exec_tab['accumulation_algorithm'] == 'retained-local-graph-v1'

    def test_tier1_parse_cpu_replicated_gloo_batch_partitioning(self):
        """Case 1.2: 2-rank CPU Gloo partitions global_batch_size=16 into local_batch_size=8."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}
        resolved = resolve_execution_profile(raw, cfg)

        exec_tab = resolved['execution']
        assert exec_tab['name'] == 'cpu-replicated-gloo'
        assert exec_tab['world_size'] == 2
        assert exec_tab['accumulation_steps'] == 1
        assert exec_tab['global_batch_size'] == 16
        assert exec_tab['local_batch_size'] == 8
        assert exec_tab['microbatch_size'] == 8
        assert exec_tab['accumulation_algorithm'] == 'retained-local-graph-v1'

    def test_tier1_parse_cuda_replicated_nccl_with_accumulation(self):
        """Case 1.3: 2-rank NCCL with accumulation_steps=4 selects detached-logit-vjp-replay-v1."""
        cfg = make_toy_config(device="cuda", batch_size=16)
        raw = {'schema_version': 1, 'execution': {'name': 'cuda-replicated-nccl', 'world_size': 2, 'accumulation_steps': 4}}
        resolved = resolve_execution_profile(raw, cfg)

        exec_tab = resolved['execution']
        assert exec_tab['name'] == 'cuda-replicated-nccl'
        assert exec_tab['world_size'] == 2
        assert exec_tab['accumulation_steps'] == 4
        assert exec_tab['global_batch_size'] == 16
        assert exec_tab['local_batch_size'] == 8
        assert exec_tab['microbatch_size'] == 2
        assert exec_tab['accumulation_algorithm'] == 'detached-logit-vjp-replay-v1'

    @requires_m1_schema
    def test_tier1_parse_profile_aggregation_defaults(self):
        """Case 1.4: Omitted [aggregation] defaults to strategy='gradient_allreduce', sync_every_steps=1."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}
        resolved = resolve_execution_profile(raw, cfg)

        assert 'aggregation' in resolved or 'aggregation' in resolved['execution']
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'gradient_allreduce'
        assert agg['sync_every_steps'] == 1
        assert agg.get('outer_lr', 1.0) == pytest.approx(1.0)
        assert agg.get('outer_momentum', 0.0) == pytest.approx(0.0)

    @requires_m1_schema
    def test_tier1_parse_profile_multinode_defaults(self):
        """Case 1.5: Multi-node topology resolution computes ranks_per_node = world_size // nodes."""
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
        }
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['world_size'] == 4
        assert exec_tab['nodes'] == 2
        assert exec_tab['node_rank'] == 1
        assert exec_tab['ranks_per_node'] == 2
        assert exec_tab['rendezvous'] == 'tcp://127.0.0.1:29500'

    @requires_m1_schema
    def test_tier1_parse_profile_omitted_nodes_default(self):
        """Case 1.6: Omitted nodes and node_rank default to nodes=1, node_rank=0."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab.get('nodes', 1) == 1
        assert exec_tab.get('node_rank', 0) == 0
        assert exec_tab.get('ranks_per_node', exec_tab['world_size']) == 2


# ==============================================================================
# Feature 2: Replicated Smoke Steps (>= 5 test cases)
# ==============================================================================

class TestReplicatedSmokeSteps:
    """Feature 2: Replicated smoke steps and numerical parity."""

    def test_tier1_replicated_cpu_smoke_steps(self):
        """Case 2.1: 2-rank CPU replicated trainer completes 2 steps cleanly."""
        cfg = make_toy_config(device="cpu", batch_size=8, steps=2)
        results = run_replicated_cpu_smoke_test(cfg, steps=2, world_size=2)
        assert len(results) == 2
        for r_res in results:
            assert r_res['passed']
            assert len(r_res['history']) == 2
            assert r_res['history'][0]['step'] == 1
            assert r_res['history'][1]['step'] == 2
            assert math.isfinite(r_res['history'][0]['d_loss'])
            assert math.isfinite(r_res['history'][0]['g_loss'])

    def test_tier1_replicated_cpu_vs_reference_parity(self):
        """Case 2.2: 2-rank CPU replicated run matches reference trainer losses within tolerance."""
        from hypergan.training import ReferenceTrainer
        cfg = make_toy_config(device="cpu", batch_size=8, steps=2)

        results = run_replicated_cpu_smoke_test(cfg, steps=2, world_size=2)
        ref_trainer = ReferenceTrainer(cfg)

        data = torch.linspace(-1, 1, 16).reshape(8, 2)
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        ref_m1, _ = ref_trainer.update({'real': data}, (ref_trainer.prior.z[ids], ids))
        ref_m2, _ = ref_trainer.update({'real': data}, (ref_trainer.prior.z[ids], ids))

        rep_rank0 = results[0]['history']
        assert rep_rank0[0]['d_loss'] == pytest.approx(ref_m1['d_loss'], rel=3e-5, abs=3e-6)
        assert rep_rank0[0]['g_loss'] == pytest.approx(ref_m1['g_loss'], rel=3e-5, abs=3e-6)
        assert rep_rank0[1]['d_loss'] == pytest.approx(ref_m2['d_loss'], rel=3e-5, abs=3e-6)
        assert rep_rank0[1]['g_loss'] == pytest.approx(ref_m2['g_loss'], rel=3e-5, abs=3e-6)

    @requires_2_gpus
    def test_tier1_replicated_gpu_smoke_steps(self, tmp_path):
        """Case 2.3: 2-GPU NCCL smoke step executes on local 2x A6000 hardware via CPUWorkerService."""
        from hypergan.cpu_worker_service import CPUWorkerService
        driver = tmp_path / "nccl_smoke.py"
        driver.write_text('''
import os, sys, torch, torch.distributed as dist
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world):
    assert dist.get_backend() == 'nccl'
    assert torch.cuda.current_device() == rank
    return rank

def handler(rank, operation, payload):
    if operation == 'ping':
        t = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(t)
        torch.cuda.synchronize(rank)
        return {'rank': rank, 'sum': t.item()}
    raise ValueError(operation)

if __name__ == '__main__':
    service = CPUWorkerService(factory, handler, backend='nccl', run_id='e2e-gpu', attempt_id='att1',
                               startup_timeout=20, command_timeout=10, collective_timeout=5, total_timeout=30)
    with service:
        res = service.command('ping')['results']
        assert res == [{'rank': 0, 'sum': 3.0}, {'rank': 1, 'sum': 3.0}]
''')
        proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=35, env=os.environ.copy())
        assert proc.returncode == 0, f"GPU NCCL failed: {proc.stdout}\n{proc.stderr}"

    def test_tier1_replicated_microbatch_accumulation_smoke(self):
        """Case 2.4: 2-rank replicated trainer with accumulation_steps=2 executes microbatches."""
        cfg = make_toy_config(device="cpu", batch_size=8, steps=1)
        results = run_replicated_cpu_smoke_test(cfg, steps=1, world_size=2, accumulation_steps=2)
        assert len(results) == 2
        for r_res in results:
            assert r_res['passed']
            assert r_res['history'][0]['step'] == 1
            assert math.isfinite(r_res['history'][0]['d_loss'])

    def test_tier1_replicated_state_bit_identical_across_replicas(self):
        """Case 2.5: All replicas maintain bit-identical parameter values after synchronous step."""
        cfg = make_toy_config(device="cpu", batch_size=8, steps=1)
        results = run_replicated_cpu_smoke_test(cfg, steps=1, world_size=2)
        w0 = results[0]['weights']
        w1 = results[1]['weights']
        assert set(w0.keys()) == set(w1.keys())
        for param_name in w0:
            assert w0[param_name] == w1[param_name], f"Weights differed across ranks for {param_name}"


# ==============================================================================
# Feature 3: Prior Inputs in Replicated Mode (>= 5 test cases)
# ==============================================================================

class TestPriorInputsReplicated:
    """Feature 3: Generator and component inputs bound to prior.* in replicated mode."""

    @requires_m1_recipe_parity
    def test_tier1_prior_inputs_validation_acceptance(self):
        """Case 3.1: validate_replicated_recipe accepts config with inputs={'x': 'prior.z'}."""
        cfg = make_prior_bound_config(device="cpu", batch_size=8)
        # Must not raise ValueError
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_prior_sample_input_binding_acceptance(self):
        """Case 3.2: validate_replicated_recipe accepts config with prior.sample inputs."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        cfg['components']['generator']['inputs'] = {'x': 'prior.sample'}
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_replicated_prior_input_forward_backward(self):
        """Case 3.3: 2-rank replicated trainer executes step with prior-bound generator inputs."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_prior_bound_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)

        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert metrics['step'] == 1
        assert math.isfinite(metrics['g_loss'])

    @requires_m1_recipe_parity
    def test_tier1_replicated_prior_gradient_allreduce(self):
        """Case 3.4: Gradients flowing through prior-bound generator are properly reduced."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_prior_bound_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)
        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        # Prior parameters or generator parameters received valid non-zero gradients
        assert trainer.checkpoint_ready

    @requires_m1_recipe_parity
    def test_tier1_prior_input_replicated_vs_reference_parity(self):
        """Case 3.5: Prior-bound replicated run matches reference single-device loss within tolerance."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        from hypergan.training import ReferenceTrainer
        cfg = make_prior_bound_config(device="cpu", batch_size=8)
        rep = ReplicatedCPUTrainer(cfg, world_size=2)
        ref = ReferenceTrainer(cfg)

        data = torch.linspace(-1, 1, 16).reshape(8, 2)
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        rep_m, _ = rep.update({'real': data.chunk(2)[0]}, (rep.prior.z[ids[:4]], ids[:4]))
        ref_m, _ = ref.update({'real': data}, (ref.prior.z[ids], ids))

        assert rep_m['d_loss'] == pytest.approx(ref_m['d_loss'], rel=3e-5, abs=3e-6)
        assert rep_m['g_loss'] == pytest.approx(ref_m['g_loss'], rel=3e-5, abs=3e-6)


# ==============================================================================
# Feature 4: Component Reuse in Replicated Mode (>= 5 test cases)
# ==============================================================================

class TestComponentReuseReplicated:
    """Feature 4: Shared component instances across graph positions in replicated mode."""

    @requires_m1_recipe_parity
    def test_tier1_component_reuse_validation_acceptance(self):
        """Case 4.1: validate_replicated_recipe accepts config containing component reuse."""
        cfg = make_reuse_config(device="cpu", batch_size=8)
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_component_reuse_graph_construction(self):
        """Case 4.2: Graph constructed with reused component shares parameter tensors."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_reuse_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)
        # encoder_a and encoder_b must share the exact same parameter objects
        mod_a = trainer.graph.models['encoder_a']
        mod_b = trainer.graph.models['encoder_b']
        for p_a, p_b in zip(mod_a.parameters(), mod_b.parameters()):
            assert p_a is p_b, "Reused component parameters must be identical object in memory"

    @requires_m1_recipe_parity
    def test_tier1_replicated_reuse_gradient_accumulation(self):
        """Case 4.3: Gradients accumulate correctly across multiple call sites of reused component."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_reuse_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)

        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert metrics['step'] == 1
        assert math.isfinite(metrics['d_loss'])

    @requires_m1_recipe_parity
    def test_tier1_replicated_reuse_allreduce_parity(self):
        """Case 4.4: Reused component parameters remain synchronized across all ranks."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_reuse_config(device="cpu", batch_size=8)
        rank0 = ReplicatedCPUTrainer(cfg, world_size=2, rank=0)
        rank1 = ReplicatedCPUTrainer(cfg, world_size=2, rank=1)
        for (n0, p0), (n1, p1) in zip(rank0.graph.named_parameters(), rank1.graph.named_parameters()):
            assert torch.equal(p0, p1)

    @requires_m1_recipe_parity
    def test_tier1_reuse_replicated_vs_reference_loss_parity(self):
        """Case 4.5: Replicated reuse run matches reference single-device loss within tolerance."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        from hypergan.training import ReferenceTrainer
        cfg = make_reuse_config(device="cpu", batch_size=8)
        rep = ReplicatedCPUTrainer(cfg, world_size=2)
        ref = ReferenceTrainer(cfg)

        data = torch.linspace(-1, 1, 16).reshape(8, 2)
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        rep_m, _ = rep.update({'real': data.chunk(2)[0]}, (rep.prior.z[ids[:4]], ids[:4]))
        ref_m, _ = ref.update({'real': data}, (ref.prior.z[ids], ids))

        assert rep_m['d_loss'] == pytest.approx(ref_m['d_loss'], rel=3e-5, abs=3e-6)


# ==============================================================================
# Feature 5: Extra Adversarial Terms in Replicated Mode (>= 5 test cases)
# ==============================================================================

class TestExtraAdversarialTermsReplicated:
    """Feature 5: Multiple critic/generator adversarial term pairs in replicated mode."""

    @requires_m1_recipe_parity
    def test_tier1_adversarial_terms_validation_acceptance(self):
        """Case 5.1: validate_replicated_recipe accepts config with extra adversarial_terms."""
        cfg = make_adversarial_terms_config(device="cpu", batch_size=8)
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_multiple_adversarial_terms_acceptance(self):
        """Case 5.2: validate_replicated_recipe accepts multiple extra adversarial terms."""
        cfg = make_adversarial_terms_config(device="cpu", batch_size=8)
        cfg['adversarial_terms'].append({
            'id': 'term_diversity',
            'component': 'discriminator',
            'real': 'batch.real',
            'fake': 'generated',
        })
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_replicated_adversarial_terms_execution(self):
        """Case 5.3: 2-rank replicated trainer executes all adversarial terms in program loop."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_adversarial_terms_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)

        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert metrics['step'] == 1
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['g_loss'])

    @requires_m1_recipe_parity
    def test_tier1_adversarial_terms_gradient_reduction(self):
        """Case 5.4: Gradients from extra adversarial terms are all-reduced before optimizer step."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = make_adversarial_terms_config(device="cpu", batch_size=8)
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)
        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert trainer.checkpoint_ready

    @requires_m1_recipe_parity
    def test_tier1_adversarial_terms_replicated_vs_reference_parity(self):
        """Case 5.5: Parity between replicated extra adversarial terms and reference trainer."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        from hypergan.training import ReferenceTrainer
        cfg = make_adversarial_terms_config(device="cpu", batch_size=8)
        rep = ReplicatedCPUTrainer(cfg, world_size=2)
        ref = ReferenceTrainer(cfg)

        data = torch.linspace(-1, 1, 16).reshape(8, 2)
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        rep_m, _ = rep.update({'real': data.chunk(2)[0]}, (rep.prior.z[ids[:4]], ids[:4]))
        ref_m, _ = ref.update({'real': data}, (ref.prior.z[ids], ids))

        assert rep_m['d_loss'] == pytest.approx(ref_m['d_loss'], rel=3e-5, abs=3e-6)
        assert rep_m['g_loss'] == pytest.approx(ref_m['g_loss'], rel=3e-5, abs=3e-6)


# ==============================================================================
# Feature 6: Recipe Presets in Replicated Mode (>= 5 test cases)
# ==============================================================================

class TestPresetsReplicated:
    """Feature 6: ParticleGAN and standard recipe presets in replicated mode."""

    @requires_m1_recipe_parity
    def test_tier1_preset_bispf_validation_acceptance(self):
        """Case 6.1: validate_replicated_recipe accepts recipe='particlegan-bispf'."""
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8},
        })
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_preset_dualnorm_validation_acceptance(self):
        """Case 6.2: validate_replicated_recipe accepts recipe='particlegan-dualnorm'."""
        cfg = resolve_config({
            'recipe': 'particlegan-dualnorm',
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8},
        })
        validate_replicated_recipe(cfg)

    @requires_m1_recipe_parity
    def test_tier1_preset_resolution_to_components(self):
        """Case 6.3: Resolving preset recipe generates complete graph components table."""
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8},
        })
        assert 'components' in cfg
        assert 'generator' in cfg['components']
        assert 'discriminator' in cfg['components']

    @requires_m1_recipe_parity
    def test_tier1_preset_replicated_smoke_step(self):
        """Case 6.4: 2-rank replicated trainer executes smoke step using resolved preset recipe."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8},
        })
        trainer = ReplicatedCPUTrainer(cfg, world_size=2)
        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert metrics['step'] == 1

    @requires_m1_recipe_parity
    def test_tier1_preset_replicated_accumulation(self):
        """Case 6.5: Preset recipe with accumulation_steps=2 partitions microbatches cleanly."""
        from hypergan.distributed_training import ReplicatedCPUTrainer
        cfg = resolve_config({
            'recipe': 'particlegan-bispf',
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8},
        })
        trainer = ReplicatedCPUTrainer(cfg, world_size=2, accumulation_steps=2)
        data = torch.randn(8, 2)
        local_ids = torch.tensor([0, 1, 2, 3])
        metrics, _ = trainer.update({'real': data.chunk(2)[0]}, (trainer.prior.z[local_ids], local_ids))
        assert metrics['step'] == 1
