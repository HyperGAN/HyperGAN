#!/usr/bin/env python3
"""Opt-in local two-GPU NCCL diagnostic; does not run HyperGAN or qualify GANs.

Run with a CUDA-enabled interpreter from a guarded file entry point. The parent
bounds each two-rank job and reaps its own children, but this standalone smoke
is not the production coordinator-death supervisor. Do not interrupt the parent
with SIGKILL or use this diagnostic as a training launcher. Other GPU processes
are neither stopped nor modified. Only small tensors and a linear Adam step are
allocated. Reports/logs are retained in an exclusive output namespace.
"""
import argparse
from datetime import timedelta
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _rank(rank, devices, rendezvous, directory, collective_timeout, fault):
    # Keep NCCL watchdog output separate from the parent JSON result.
    with (Path(directory) / f'rank-{rank}.log').open('w') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            import torch
            import torch.distributed as dist
            from torch.distributed.nn.functional import all_gather
            if not torch.cuda.is_available() or not dist.is_nccl_available():
                raise RuntimeError('A CUDA-enabled Torch installation with NCCL is required')
            if max(devices) >= torch.cuda.device_count():
                raise ValueError('Requested device indices exceed visible CUDA devices')
            torch.set_num_threads(1)
            device = torch.device('cuda', devices[rank])
            torch.cuda.set_device(device)
            dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                                    timeout=timedelta(seconds=collective_timeout), device_id=device)
            if fault:
                dist.barrier()
                _write(Path(directory) / f'rank-{rank}.phase.json',
                       {'pid': os.getpid(), 'phase': 'missing-collective' if rank == 1 else 'waiting-collective'})
                if rank == 1:
                    time.sleep(120)
                else:
                    value = torch.ones(1, device=device)
                    dist.all_reduce(value)
                    torch.cuda.synchronize(device)
                raise AssertionError('The deliberately missing peer collective unexpectedly completed')

            value = torch.tensor([float(rank + 1)], device=device)
            dist.all_reduce(value)
            torch.testing.assert_close(value, torch.tensor([3.0], device=device), rtol=0, atol=0)
            value.fill_(7 if rank == 0 else -1)
            dist.broadcast(value, src=0)
            assert value.item() == 7
            gathered = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(gathered, torch.tensor([float(rank)], device=device))
            assert [item.item() for item in gathered] == [0, 1]

            # Same differentiable gather shape used by global-mean GAN terms:
            # each rank consumes the full mean; backward sums rank consumers.
            local = torch.tensor([[float(rank + 1)]], device=device, requires_grad=True)
            global_mean = torch.cat(all_gather(local)).mean()
            first, = torch.autograd.grad(global_mean.square(), local, create_graph=True)
            second, = torch.autograd.grad(first.sum(), local)
            torch.testing.assert_close(first, torch.tensor([[3.0]], device=device), rtol=0, atol=0)
            torch.testing.assert_close(second, torch.tensor([[2.0]], device=device), rtol=0, atol=0)

            model = torch.nn.Linear(2, 1, bias=False, device=device)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[.5, -.25]], device=device))
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            full = torch.arange(1, 9, dtype=torch.float32, device=device).reshape(4, 2)
            model(full[rank * 2:(rank + 1) * 2]).square().mean().backward()
            dist.all_reduce(model.weight.grad)
            model.weight.grad.div_(2)
            baseline = torch.nn.Linear(2, 1, bias=False, device=device)
            with torch.no_grad():
                baseline.weight.copy_(model.weight)
            baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=.001)
            baseline(full).square().mean().backward()
            torch.testing.assert_close(model.weight.grad, baseline.weight.grad, rtol=1e-6, atol=1e-6)
            gradient = model.weight.grad.detach().cpu().tolist()
            optimizer.step()
            baseline_optimizer.step()
            torch.testing.assert_close(model.weight, baseline.weight, rtol=0, atol=0)
            for key, expected in baseline_optimizer.state[baseline.weight].items():
                torch.testing.assert_close(optimizer.state[model.weight][key], expected, rtol=0, atol=0)
            peers = [torch.empty_like(model.weight) for _ in range(2)]
            dist.all_gather(peers, model.weight.detach())
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
            torch.cuda.synchronize(device)
            properties = torch.cuda.get_device_properties(device)
            receipt = {'rank': rank, 'pid': os.getpid(), 'device_index': devices[rank],
                       'name': properties.name, 'uuid': str(properties.uuid),
                       'capability': list(torch.cuda.get_device_capability(device)),
                       'total_memory_bytes': properties.total_memory,
                       'peer_access': torch.cuda.can_device_access_peer(devices[rank], devices[1-rank]),
                       'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                       'nccl': list(torch.cuda.nccl.version()), 'python': sys.version.split()[0],
                       'gradient': gradient, 'adam_weight': model.weight.detach().cpu().tolist(),
                       'differentiable_gather_first': first.item(), 'differentiable_gather_second': second.item(),
                       'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            dist.destroy_process_group()
            _write(Path(directory) / f'rank-{rank}.json', receipt)
        except BaseException as error:
            _write(Path(directory) / f'rank-{rank}.error.json', {'error': f'{type(error).__name__}: {error}'})
            traceback.print_exc()
            raise


def _job(directory, devices, timeout, collective_timeout, fault):
    directory.mkdir()
    context, processes = mp.get_context('spawn'), []
    started, failure = time.monotonic(), None
    try:
        for rank in range(2):
            process = context.Process(target=_rank, args=(rank, devices, (directory / 'rendezvous').as_uri(),
                                                         str(directory), collective_timeout, fault))
            process.start()
            processes.append(process)
        while True:
            exits = [process.exitcode for process in processes]
            if any(code not in (None, 0) for code in exits):
                failure = 'rank-failed'
                break
            if all(code == 0 for code in exits):
                break
            if time.monotonic() - started >= timeout:
                failure = 'supervisor-deadline'
                break
            time.sleep(.02)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 2
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            raise RuntimeError('OS did not reap diagnostic GPU ranks')
    result = {'seconds': time.monotonic() - started, 'failure': failure,
              'ranks': [{'pid': p.pid, 'exit_code': p.exitcode, 'reaped': not p.is_alive()} for p in processes]}
    for process in processes:
        process.close()
    if fault:
        phases = [directory / f'rank-{rank}.phase.json' for rank in range(2)]
        logs = ''.join((directory / f'rank-{rank}.log').read_text(errors='replace') for rank in range(2))
        result['all_ranks_entered_fault'] = all(path.exists() for path in phases)
        result['nccl_timeout_reported'] = 'Watchdog caught collective operation timeout' in logs
        result['passed'] = bool(failure == 'rank-failed' and result['all_ranks_entered_fault']
                                and result['nccl_timeout_reported'])
    else:
        result['passed'] = failure is None
        if result['passed']:
            result['devices'] = [json.loads((directory / f'rank-{rank}.json').read_text()) for rank in range(2)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0,1', help='Two distinct visible CUDA indices')
    parser.add_argument('--timeout', type=float, default=45)
    parser.add_argument('--collective-timeout', type=float, default=5)
    parser.add_argument('--fault-timeout', action='store_true', help='Also test missing-peer NCCL timeout and rank reaping')
    args = parser.parse_args()
    try:
        devices = [int(value) for value in args.devices.split(',')]
    except ValueError:
        parser.error('--devices must be two distinct nonnegative indices')
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error('--devices must be two distinct nonnegative indices')
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.collective_timeout)):
        parser.error('Timeouts must be finite positive seconds')
    if args.collective_timeout >= args.timeout:
        parser.error('--collective-timeout must be shorter than --timeout')
    output = args.output.resolve()
    if output.exists():
        parser.error('--output already exists; use a fresh receipt path')
    directory = output.with_suffix(output.suffix + '.artifacts')
    directory.mkdir(parents=True, exist_ok=False)
    result = {'kind': 'hypergan-two-gpu-nccl-smoke', 'scope': 'collectives-and-linear-adam-only',
              'devices_requested': devices, 'collective_timeout': args.collective_timeout,
              'supervisor_timeout': args.timeout, 'artifacts': str(directory),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'nccl_environment': {key: value for key, value in os.environ.items()
                                   if key.startswith(('TORCH_NCCL_', 'NCCL_'))}}
    for label, options in [('gpu_inventory', ['--query-gpu=index,name,uuid,driver_version,memory.used', '--format=csv']),
                           ('topology', ['topo', '-m'])]:
        result[label] = subprocess.check_output(['nvidia-smi', *options], text=True, timeout=10)
    result['collectives'] = _job(directory / 'collectives', devices, args.timeout, args.collective_timeout, False)
    if args.fault_timeout and result['collectives']['passed']:
        result['missing_peer'] = _job(directory / 'missing-peer', devices, args.timeout, args.collective_timeout, True)
    result['passed'] = result['collectives']['passed'] and (not args.fault_timeout or result.get('missing_peer', {}).get('passed', False))
    _write(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
