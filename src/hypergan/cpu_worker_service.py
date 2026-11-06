"""Internal fixed-group CPU commands with an independent worker-owning broker.

The parent and broker import no numerical runtime. This is not a run service or
checkpoint commit authority; commands execute trusted Python in disposable ranks.
"""
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import re
import socket
import struct
import sys
import tempfile
import time
import traceback


MAX_FRAME_BYTES = 65536
# Experimental multi-node: follower nodes mirror node 0's command stream, which
# global rank 0 publishes in the process group's TCP store before each command.
FOLLOW_OPERATION = 'follow-node-0'
_LEADER_COMMAND = 'hypergan/node-0-command/{}'


class CPUServiceCancelled(RuntimeError):
    """The caller explicitly cancelled a supervised command."""


_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
_OPERATIONS = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')


def _json(value):
    def check(item):
        if item is None or type(item) in (str, int, float, bool):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError('CPU service messages require JSON values and string object keys')
    try:
        check(value)
        data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f'Invalid CPU service JSON: {exc}') from exc
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError(f'CPU service frame exceeds {MAX_FRAME_BYTES} bytes')
    return data


class _Channel:
    """At most one queued output; socket I/O never blocks the broker monitor."""
    def __init__(self, sock):
        self.socket = sock
        sock.setblocking(False)
        self.incoming = bytearray()
        self.outgoing = b''
        self.closed = False

    def send(self, value):
        if self.outgoing:
            raise RuntimeError('CPU service output queue is full')
        data = _json(value)
        self.outgoing = struct.pack('!I', len(data)) + data

    def pump(self):
        # Drain terminal diagnostics before a queued health/next-command write
        # can encounter a peer that has already closed its end of the socket.
        try:
            data = self.socket.recv(MAX_FRAME_BYTES + 4 - len(self.incoming))
            if not data:
                self.closed = True
            self.incoming.extend(data)
        except BlockingIOError:
            pass
        rows = []
        while len(self.incoming) >= 4:
            size, = struct.unpack('!I', self.incoming[:4])
            if not 1 <= size <= MAX_FRAME_BYTES:
                raise ValueError('Invalid CPU service frame length')
            if len(self.incoming) < size + 4:
                break
            data = bytes(self.incoming[4:size + 4])
            del self.incoming[:size + 4]
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ValueError('Invalid CPU service JSON frame') from exc
            _json(value)
            rows.append(value)
        if self.outgoing and not self.closed:
            try:
                count = self.socket.send(self.outgoing)
                self.outgoing = self.outgoing[count:]
            except BlockingIOError:
                pass
            except (BrokenPipeError, ConnectionResetError):
                self.outgoing = b''
                # Read remaining peer data/EOF on the next pump. A buffered
                # error frame has priority over this secondary write failure.
        return rows

    def close(self):
        self.socket.close()


def _envelope(identity, sequence, operation, kind, **values):
    return dict(values, run_id=identity[0], attempt_id=identity[1], sequence=sequence,
                operation=operation, kind=kind)


def _validate(row, identity, sequence, operation=None):
    if (not isinstance(row, dict) or row.get('run_id') != identity[0]
            or row.get('attempt_id') != identity[1]
            or type(row.get('sequence')) is not int or row['sequence'] != sequence
            or (operation is not None and row.get('operation') != operation)):
        raise ValueError('Stale or invalid CPU service run/attempt/sequence/operation')


def _receive(channel, timeout=None, cancellation_event=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        rows = channel.pump()
        if rows:
            if len(rows) != 1:
                raise ValueError('CPU service permits one outstanding command')
            return rows[0]
        if channel.closed:
            raise EOFError('CPU service channel closed')
        # Already-received worker failures/results take precedence over a late
        # cancellation request; cancellation must not disguise actual failures.
        if cancellation_event is not None and cancellation_event.is_set():
            raise CPUServiceCancelled('CPU service command cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('CPU service response deadline exceeded')
        time.sleep(0.01)


def _flush(channel, timeout=2):
    deadline = time.monotonic() + timeout
    while channel.outgoing and time.monotonic() < deadline:
        # Flushing also detects closure but never accepts an unsolicited message.
        if channel.pump():
            raise ValueError('Unexpected CPU service message during shutdown')
        time.sleep(0.01)


def _agree_command(row, world_size):
    import torch.distributed as dist
    shared_payload = row.get('payload')
    if isinstance(shared_payload, dict):
        shared_payload = {k: v for k, v in shared_payload.items()
                          if k not in ('controller_id', 'metadata', 'identity', 'bundle_dir')}
    shared_command = {
        'sequence': row.get('sequence'),
        'operation': row.get('operation'),
        'kind': row.get('kind'),
        'payload': shared_payload,
    }
    digest = hashlib.sha256(_json(shared_command)).hexdigest()
    agreed = [None] * world_size
    dist.all_gather_object(agreed, digest)
    if any(item != digest for item in agreed):
        raise ValueError('Ranks disagree on CPU command identity or payload')


def _store():
    from torch.distributed.distributed_c10d import _get_default_store
    return _get_default_store()


def _publish_leader_command(sequence, row):
    # Every rank read command N-1 before agreeing on it, so it can be dropped.
    store = _store()
    store.set(_LEADER_COMMAND.format(sequence), _json(row))
    if sequence > 1:
        store.delete_key(_LEADER_COMMAND.format(sequence - 1))


_FOLLOW_POLL_MAX = 0.1


def _follow_leader(channel, state, handler, world_size, backend, local_rank):
    """Run node 0's commands on a follower rank until node 0 shuts the group down.

    Waiting is idle, like a node-0 rank between commands: no collective runs, so no
    deadline applies. A dead node 0 takes its TCP store with it and ends the wait.
    """
    import torch
    import torch.distributed as dist
    store, sequence, count = _store(), 0, 0
    while True:
        sequence += 1
        key = _LEADER_COMMAND.format(sequence)
        # Back off while idle so large worlds do not flood node 0's store with checks.
        delay = 0.005
        while not store.check([key]):
            if channel.pump() or channel.closed:
                raise EOFError('Follower broker closed while waiting for node 0')
            time.sleep(delay)
            delay = min(delay * 2, _FOLLOW_POLL_MAX)
        row = json.loads(store.get(key))
        if (not isinstance(row, dict) or row.get('sequence') != sequence or row.get('kind') != 'command'
                or not isinstance(row.get('operation'), str) or 'payload' not in row):
            raise ValueError('Invalid node 0 command in the process group store')
        if backend == 'nccl' and torch.cuda.current_device() != local_rank:
            raise ValueError('NCCL handler changed the rank-owned current CUDA device')
        _agree_command(row, world_size)
        operation = row['operation']
        if operation == '__shutdown__':
            dist.barrier()
            dist.destroy_process_group()
            return {'commands': count}
        if not _OPERATIONS.fullmatch(operation):
            raise ValueError('Invalid node 0 operation')
        try:
            handler(state, operation, row['payload'])
        except Exception as exc:
            raise RuntimeError(f'Following node 0 command {operation!r} failed: {type(exc).__name__}: {exc}') from exc
        count += 1


def _rank_main(sock, rank, world_size, identity, rendezvous, collective_timeout, bootstrap, initialize_process_group, backend='gloo', ranks_per_node=None,
               nodes=1):
    channel = _Channel(sock)
    sequence, operation = 0, '__start__'
    if ranks_per_node is None:
        ranks_per_node = world_size
    local_rank = rank % ranks_per_node
    group_open = initialize_process_group
    publish = initialize_process_group and nodes > 1 and rank == 0
    follower = initialize_process_group and nodes > 1 and rank >= ranks_per_node
    try:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        if initialize_process_group:
            import torch
            import torch.distributed as dist
            torch.set_num_threads(1)
            options = {}
            if backend == 'nccl':
                if not torch.cuda.is_available() or not dist.is_nccl_available():
                    raise RuntimeError('NCCL execution requires CUDA-enabled PyTorch and available NVIDIA GPUs')
                if ranks_per_node > torch.cuda.device_count():
                    raise ValueError(f'NCCL ranks_per_node={ranks_per_node} requires at least {ranks_per_node} visible GPUs; found {torch.cuda.device_count()}')
                torch.cuda.set_device(local_rank)
                options['device_id'] = torch.device('cuda', local_rank)
            dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world_size,
                                    timeout=timedelta(seconds=collective_timeout), **options)
        factory, handler, args = pickle.loads(bootstrap)
        state = factory(rank, world_size, *args)
        if initialize_process_group and backend == 'nccl' and torch.cuda.current_device() != local_rank:
            raise ValueError('NCCL factory changed the rank-owned current CUDA device')
        channel.send(_envelope(identity, sequence, operation, 'ready', rank=rank, pid=os.getpid()))
        while True:
            row = _receive(channel)  # Idle outside Gloo; broker owns deadlines.
            sequence += 1
            operation = row.get('operation') if isinstance(row, dict) else None
            _validate(row, identity, sequence)
            if set(row) != {'run_id', 'attempt_id', 'sequence', 'operation', 'kind', 'payload'} or row['kind'] != 'command':
                raise ValueError('Invalid CPU service command fields')
            if operation == FOLLOW_OPERATION:
                if not follower or not group_open:
                    raise ValueError('Only ranks on a follower node of an open multi-node group can follow node 0')
                value = _follow_leader(channel, state, handler, world_size, backend, local_rank)
                group_open = False
                channel.send(_envelope(identity, sequence, operation, 'result', rank=rank, value=value))
                continue
            if group_open:
                if backend == 'nccl' and torch.cuda.current_device() != local_rank:
                    raise ValueError('NCCL handler changed the rank-owned current CUDA device')
                if publish:
                    _publish_leader_command(sequence, row)
                _agree_command(row, world_size)
            if operation == '__shutdown__':
                if group_open:
                    dist.barrier()
                    dist.destroy_process_group()
                channel.send(_envelope(identity, sequence, operation, 'result', rank=rank, value=None))
                _flush(channel)
                return
            if not isinstance(operation, str) or not _OPERATIONS.fullmatch(operation):
                raise ValueError('Invalid CPU service operation')
            value = handler(state, operation, row['payload'])
            channel.send(_envelope(identity, sequence, operation, 'result', rank=rank, value=value))
    except BaseException as exc:
        try:
            channel.outgoing = b''
            hint = (" Install training dependencies (CUDA-enabled Torch for NCCL) with pip install 'hypergan[train]'."
                    if isinstance(exc, ModuleNotFoundError)
                    and (exc.name or '').split('.')[0] in ('torch', 'numpy', 'particlegan') else '')
            channel.send(_envelope(identity, sequence, operation, 'error', rank=rank,
                                   error=f'{type(exc).__name__}: {exc}'[:2000] + hint,
                                   traceback=traceback.format_exc()[-6000:]))
            _flush(channel)
        except BaseException:
            pass
        raise
    finally:
        channel.close()


def _reap(processes):
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
        raise RuntimeError('Operating system did not reap a terminated CPU rank')


def _broker_main(sock, identity, world_size, limits, bootstrap, started, initialize_process_group, backend='gloo',
                 node_rank=0, nodes=1, ranks_per_node=None, rendezvous=None):
    # Terminal process-group signals belong to the controller. It completes the
    # current all-rank command before saving; its independently monitored death
    # still makes this broker reap every rank. Isolate before spawning ranks.
    if os.name != 'nt':
        os.setsid()
    if ranks_per_node is None:
        ranks_per_node = world_size // nodes
    start_rank = node_rank * ranks_per_node
    local_ranks_count = ranks_per_node
    parent = mp.parent_process()
    control = _Channel(sock)
    context = mp.get_context('spawn')
    processes, channels = [], []
    sequence, operation, phase = 0, '__start__', 'startup'
    results = {}
    deadline = started + limits['startup_timeout']
    terminal = None
    try:
        is_unspecified = rendezvous in (None, 'file://')
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix='hypergan-worker-service-') if is_unspecified else nullcontext()
        with temp_dir_ctx as temporary:
            effective_rendezvous = (Path(temporary) / 'rendezvous').as_uri() if is_unspecified else rendezvous
            try:
                for rank in range(start_rank, start_rank + local_ranks_count):
                    if not parent.is_alive():
                        raise EOFError('CPU service coordinator died during startup')
                    if control.pump() or control.closed:
                        raise EOFError('CPU service coordinator disconnected during startup')
                    local, remote = socket.socketpair()
                    process = context.Process(target=_rank_main, name=f'hypergan-service-rank-{rank}',
                                              args=(remote, rank, world_size, identity, effective_rendezvous,
                                                    limits['collective_timeout'], bootstrap, initialize_process_group, backend, ranks_per_node,
                                                    nodes))
                    try:
                        process.start()
                    except BaseException:
                        local.close()
                        raise
                    finally:
                        remote.close()
                    processes.append(process)
                    channels.append(_Channel(local))
                while True:
                    if not parent.is_alive():
                        raise EOFError('CPU service coordinator died; stopping owned ranks')
                    now = time.monotonic()
                    total_timeout = limits.get('total_timeout')
                    if total_timeout is not None and math.isfinite(total_timeout):
                        if now >= started + total_timeout:
                            raise TimeoutError('CPU service total deadline exceeded (including idle time)')
                    if phase != 'idle' and now >= deadline:
                        raise TimeoutError(f'CPU service {phase} deadline exceeded')
                    for idx, channel in enumerate(channels):
                        rank = start_rank + idx
                        try:
                            rows = channel.pump()
                        except (OSError, ValueError) as exc:
                            raise RuntimeError(f'CPU rank {rank} channel failed: {exc}') from exc
                        for row in rows:
                            _validate(row, identity, sequence, operation)
                            if type(row.get('rank')) is not int or row['rank'] != rank:
                                raise ValueError('CPU service rank identity mismatch')
                            if row.get('kind') == 'error':
                                raise RuntimeError(f"rank {rank}: {row.get('error')}\n{row.get('traceback', '')}")
                            expected = 'ready' if phase == 'startup' else 'result'
                            if phase == 'idle' or row.get('kind') != expected or rank in results:
                                raise ValueError(f'Unexpected or duplicate CPU result from rank {rank}')
                            results[rank] = row
                        if channel.closed and phase != 'closing':
                            raise RuntimeError(f'CPU rank {rank} command channel closed unexpectedly '
                                               f'(exit code {processes[idx].exitcode})')
                    for idx, process in enumerate(processes):
                        rank = start_rank + idx
                        if process.exitcode is not None and not (phase == 'closing' and process.exitcode == 0):
                            raise RuntimeError(f'CPU rank {rank} exited unexpectedly with code {process.exitcode}')
                    if phase != 'idle' and len(results) == local_ranks_count:
                        if phase == 'startup':
                            control.send(_envelope(identity, sequence, operation, 'ready',
                                                   worker_pids=[process.pid for process in processes]))
                        elif phase == 'closing':
                            if all(process.exitcode == 0 for process in processes):
                                terminal = _envelope(identity, sequence, operation, 'closed')
                                break
                            time.sleep(0.01)
                            continue
                        else:
                            control.send(_envelope(identity, sequence, operation, 'result',
                                                   results=[results[r]['value'] for r in range(start_rank, start_rank + local_ranks_count)]))
                        phase, results = 'idle', {}
                    for row in control.pump():
                        if isinstance(row, dict) and row.get('kind') == 'abort':
                            _validate(row, identity, sequence)
                            raise EOFError('CPU service aborted by coordinator')
                        if phase != 'idle':
                            raise ValueError('CPU service permits one outstanding command')
                        if isinstance(row, dict) and row.get('kind') == 'health':
                            _validate(row, identity, sequence, '__health__')
                            control.send(_envelope(identity, sequence, '__health__', 'healthy'))
                            continue
                        _validate(row, identity, sequence + 1)
                        if set(row) != {'run_id', 'attempt_id', 'sequence', 'operation', 'kind', 'payload'} or row['kind'] != 'command':
                            raise ValueError('Invalid CPU command envelope')
                        sequence += 1
                        operation = row['operation']
                        if operation != '__shutdown__' and (not isinstance(operation, str) or not _OPERATIONS.fullmatch(operation)):
                            raise ValueError('Invalid CPU service operation')
                        for channel in channels:
                            channel.send(row)
                        results = {}
                        phase = 'closing' if operation == '__shutdown__' else 'command'
                        # Following node 0 lasts the whole run; its ranks are idle between
                        # node 0 commands and fail when node 0 or its store disappears.
                        deadline = math.inf if operation == FOLLOW_OPERATION else time.monotonic() + limits['command_timeout']
                    if control.closed:
                        raise EOFError('CPU service coordinator channel closed')
                    time.sleep(0.01)
            finally:
                _reap(processes)
    except BaseException as exc:
        terminal = _envelope(identity, sequence, operation, 'error',
                             error=f'{type(exc).__name__}: {exc}'[:10000])
    finally:
        for channel in channels:
            channel.close()
        for process in processes:
            if not process.is_alive():
                process.close()
        if terminal is not None and parent.is_alive():
            try:
                control.outgoing = b''
                control.send(terminal)
                _flush(control)
            except BaseException:
                pass
        control.close()


class CPUWorkerService:
    """One sequential caller, fixed identity and group, no automatic retries.

    Factory and handler must be importable picklable callables. Start from a Python
    main guard. The broker owns/reaps ranks even after abrupt coordinator death.
    It is itself adopted/reaped by the operating system when its parent is gone.
    """
    def __init__(self, factory, handler, *, args=(), run_id, attempt_id, world_size=2,
                 startup_timeout=60, command_timeout=30, collective_timeout=15, total_timeout=300,
                 initialize_process_group=True, backend='gloo', cancellation_event=None,
                 node_rank=0, nodes=1, ranks_per_node=None, rendezvous=None):
        if not callable(factory) or not callable(handler) or not isinstance(args, tuple):
            raise ValueError('CPU service requires callable factory/handler and tuple args')
        if any(not isinstance(value, str) or not _ID.fullmatch(value) for value in (run_id, attempt_id)):
            raise ValueError('CPU service run/attempt IDs require 1-128 ASCII letters/digits, underscores or hyphens')
        if type(backend) is not str or backend not in ('gloo', 'nccl'):
            raise ValueError('Worker service backend must be gloo or nccl')
        if backend == 'nccl' and not initialize_process_group:
            raise ValueError('NCCL backend requires an initialized process group')
        if type(initialize_process_group) is not bool:
            raise ValueError('initialize_process_group must be a boolean')
        if type(world_size) is not int or not (2 <= world_size <= 64 if initialize_process_group else world_size == 1):
            raise ValueError('CPU service requires world_size 2..64 with a process group, or world_size=1 without a process group')
        limits = dict(startup_timeout=startup_timeout, command_timeout=command_timeout,
                      collective_timeout=collective_timeout, total_timeout=total_timeout)
        for name, value in limits.items():
            if name == 'total_timeout' and value is None:
                continue
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f'{name} must be finite positive seconds')
        if ranks_per_node is None:
            ranks_per_node = world_size // nodes
        self._node_rank, self._nodes, self._ranks_per_node = node_rank, nodes, ranks_per_node
        self._rendezvous = rendezvous
        self._identity, self._world_size, self._limits = (run_id, attempt_id), world_size, limits
        self._initialize_process_group = initialize_process_group
        self._backend = backend
        self._cancellation_event = cancellation_event
        self.factory, self.handler, self.args = factory, handler, args
        self._process = self._channel = None
        self._sequence = 0
        self._closed = False
        self.worker_pids = []
        self._broker_pid = None

    @property
    def broker_pid(self):
        return self._broker_pid

    @property
    def identity(self):
        return self._identity

    @property
    def limits(self):
        return dict(self._limits)

    @property
    def world_size(self):
        return self._world_size

    @property
    def backend(self):
        return self._backend

    @property
    def sequence(self):
        return self._sequence

    @property
    def next_sequence(self):
        return self._sequence + 1

    def _abort_preserving(self, error):
        try:
            self.abort()
        except BaseException as cleanup:
            note = f'CPU service cleanup also failed: {type(cleanup).__name__}: {cleanup}'
            if hasattr(error, 'add_note'):
                error.add_note(note)
            else:  # Python 3.10 retains the primary exception and attached evidence.
                error.cleanup_error = note

    def start(self):
        if self._closed or self._process is not None:
            raise RuntimeError('CPU service cannot be started again')
        started = time.monotonic()
        bootstrap = pickle.dumps((self.factory, self.handler, self.args), protocol=5)
        local, remote = socket.socketpair()
        process = mp.get_context('spawn').Process(target=_broker_main, name='hypergan-cpu-broker',
                    args=(remote, self._identity, self._world_size, self._limits, bootstrap, started, self._initialize_process_group, self._backend,
                          self._node_rank, self._nodes, self._ranks_per_node, self._rendezvous))
        main_mod = sys.modules.get('__main__')
        main_file = getattr(main_mod, '__file__', None)
        restore_main_file = main_file is not None and (main_file == '<stdin>' or not os.path.exists(main_file))
        try:
            if restore_main_file:
                main_mod.__file__ = None
            process.start()
            self._process, self._channel = process, _Channel(local)
            self._broker_pid = process.pid
        except BaseException:
            local.close()
            self._closed = True
            raise
        finally:
            if restore_main_file:
                main_mod.__file__ = main_file
            remote.close()
        try:
            total_limit = self._limits['total_timeout'] if (self._limits.get('total_timeout') is not None and math.isfinite(self._limits['total_timeout'])) else float('inf')
            result = self._reply('__start__', max(0, started + min(self._limits['startup_timeout'], total_limit) - time.monotonic()) + 5)
            if result['kind'] != 'ready':
                raise RuntimeError('CPU service did not become ready')
            self.worker_pids = result['worker_pids']
            return self
        except BaseException as exc:
            self._abort_preserving(exc)
            raise

    def _reply(self, operation, timeout):
        result = _receive(self._channel, timeout, self._cancellation_event)
        if isinstance(result, dict) and result.get('kind') == 'error':
            raise RuntimeError(f"CPU service run={self._identity[0]} attempt={self._identity[1]} "
                               f"sequence={result.get('sequence')} operation={result.get('operation')}: {result.get('error')}")
        _validate(result, self._identity, self._sequence, operation)
        expected = {'run_id', 'attempt_id', 'sequence', 'operation', 'kind'}
        if operation == '__start__':
            pids = result.get('worker_pids')
            valid = (result.get('kind') == 'ready' and set(result) == expected | {'worker_pids'}
                     and type(pids) is list and len(pids) == self._ranks_per_node
                     and all(type(pid) is int and pid > 0 for pid in pids)
                     and len(set(pids)) == len(pids))
        elif operation in ('__health__', '__shutdown__'):
            valid = (set(result) == expected and result.get('kind') ==
                     ('healthy' if operation == '__health__' else 'closed'))
        else:
            valid = (result.get('kind') == 'result' and set(result) == expected | {'results'}
                     and type(result.get('results')) is list and len(result['results']) == self._ranks_per_node)
        if not valid:
            raise RuntimeError('Invalid CPU service command result schema or rank count')
        return result

    def _request(self, operation, payload=None, *, follow=False):
        if self._closed or self._channel is None:
            raise RuntimeError('CPU service is not running')
        row = _envelope(self._identity, self._sequence + 1, operation, 'command', payload=payload)
        _json(row)  # Caller errors do not submit or consume a sequence number.
        self._sequence += 1
        try:
            self._channel.send(row)
            return self._reply(operation, None if follow else self._limits['command_timeout'] + 5)
        except BaseException as exc:
            self._abort_preserving(exc)
            raise

    def command(self, operation, payload=None):
        if not isinstance(operation, str) or not _OPERATIONS.fullmatch(operation):
            raise ValueError('Operation must start with a letter and contain at most 64 ASCII letters/digits, underscores or hyphens')
        result = self._request(operation, payload)
        return {key: result[key] for key in ('run_id', 'attempt_id', 'sequence', 'operation', 'results')}

    def follow(self):
        """Mirror node 0's command stream on this follower node until node 0 shuts down.

        Experimental multi-node execution. Node 0's controller alone decides previews,
        evaluations, checkpoints, stops and inference; follower ranks run the same
        collectives without a schedule of their own. Returns one result per local rank.
        """
        if not self._initialize_process_group or self._node_rank == 0 or self._nodes < 2:
            raise ValueError('Only a follower node (node_rank > 0) of a multi-node group can follow node 0')
        return self._request(FOLLOW_OPERATION, follow=True)['results']

    def assert_healthy(self):
        if self._closed or self._channel is None:
            raise RuntimeError('CPU service is not running')
        try:
            self._channel.send(_envelope(self._identity, self._sequence, '__health__', 'health'))
            result = self._reply('__health__', self._limits['command_timeout'] + 5)
            if result['kind'] != 'healthy':
                raise RuntimeError('CPU service health check failed')
        except BaseException as exc:
            self._abort_preserving(exc)
            raise

    def close(self):
        if self._closed:
            return
        if self._process is None:
            self._closed = True
            return
        try:
            result = self._request('__shutdown__')
            if result['kind'] != 'closed':
                raise RuntimeError('CPU service shutdown did not complete')
        except BaseException as exc:
            self._abort_preserving(exc)
            raise
        self.abort()

    def abort(self):
        if self._closed and self._process is None:
            return
        self._closed = True
        if self._channel is not None:
            # Closing the channel is independent of queued/partial messages. The
            # broker observes EOF and reaps ranks; never kill their owning broker.
            self._channel.close()
        if self._process is not None:
            self._process.join(6)
            if self._process.is_alive():
                raise RuntimeError('CPU broker has not finished cleanup; it was left alive to reap its ranks')
            self._process.close()
            self._process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._abort_preserving(exc)
