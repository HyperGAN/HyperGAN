"""Best-effort, bounded CLI output independent of terminal consumer speed.

Only training CLI commands install these process-global stream/FD redirects.
Each destination has a separate disposable drain process: a reader continually
accepts bounded lines while its writer may block in the destination. Backlogs
retain the newest lines. Durable run files, not this lossy stream, are the
complete observation/result contract.
"""
from contextlib import contextmanager
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time

MAX_LINE_BYTES = 65536
MAX_PENDING_LINES = 16
CLOSE_SECONDS = 1.0
# A run whose snapshot metrics are all manual repeats its reminder on the first
# printed progress line and every Nth one after it, never on every update.
REMINDER_EVERY_PROGRESS_LINES = 10


def human_duration(seconds):
    """Bounded, human training duration: '45s', '12m 05s' or '1h 23m'."""
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        return None
    total = int(seconds)
    if total < 60:
        return f'{total}s'
    if total < 3600:
        return f'{total // 60}m {total % 60:02d}s'
    return f'{total // 3600}h {total % 3600 // 60:02d}m'


def _progress_details(event):
    """Throughput, training time and samples seen, each only when published."""
    rate = event.get('steps_per_second')
    if type(rate) in (int, float) and math.isfinite(rate) and rate >= 0:
        yield f'{rate:.1f} steps/s' if rate >= 1 else f'{rate:.3g} steps/s'
    elapsed = human_duration(event.get('training_seconds'))
    if elapsed is not None:
        yield elapsed
    samples = event.get('samples_seen')
    if type(samples) is int and samples >= 0:
        yield f'{samples:,} samples'
    warmup = event.get('g_lr_warmup')
    if isinstance(warmup, dict):
        step, steps, rate = warmup.get('completed_steps'), warmup.get('steps'), warmup.get('g_lr')
        if type(step) is int and type(steps) is int and steps >= 2 and 0 <= step <= steps:
            yield 'G LR warmup complete' if warmup.get('status') == 'complete' else f'G LR warmup {step}/{steps}'
            if type(rate) in (int, float) and math.isfinite(rate) and rate >= 0:
                yield f'G LR {rate:.6g}'


def _tuning_message(tuning):
    status = tuning.get('status')
    parts = [{'pending': 'Preparing startup tuning', 'running': 'Tuning startup', 'complete': 'Startup tuning complete',
              'failed': 'Startup tuning failed'}.get(status, 'Tuning startup')]
    def factor(value):
        return f'{value:.4g}' if type(value) in (float, int) and math.isfinite(value) and value > 0 else None
    def brief(value):
        return ''.join(char if char.isprintable() else ' ' for char in value).strip()[:240]
    if status == 'running':
        phase = {'initialization': 'Initialization', 'dynamics': 'Trial updates'}.get(tuning.get('phase'))
        if phase:
            parts.append(phase)
    candidate, total = tuning.get('candidate'), tuning.get('total_candidates')
    if status == 'running' and type(candidate) is int and type(total) is int and 0 < candidate <= total:
        parts.append(f'Candidate {candidate} of {total}')
    if status == 'running' and tuning.get('phase') == 'dynamics':
        step, steps = tuning.get('trial_step'), tuning.get('trial_steps')
        if type(step) is int and type(steps) is int and 0 <= step <= steps and steps > 0:
            parts.append(f'Trial step {step} of {steps}')
        g_factor = tuning.get('g_lr_factor', tuning.get('lr_factor'))
        if factor(g_factor):
            parts.append('G learning rate × ' + factor(g_factor))
        if factor(tuning.get('d_lr_factor')):
            parts.append('D learning rate × ' + factor(tuning['d_lr_factor']))
    if status == 'complete':
        outcome = {'selected': 'Applied tuned initialization',
                   'kept_baseline': 'Kept baseline initialization'}.get(tuning.get('outcome'))
        if outcome:
            parts.append(outcome)
        dynamics = tuning.get('dynamics_outcome')
        if dynamics == 'selected':
            if factor(tuning.get('selected_g_lr_factor')):
                parts.append('Selected G learning rate × ' + factor(tuning['selected_g_lr_factor']))
            if factor(tuning.get('selected_d_lr_factor')):
                parts.append('Selected D learning rate × ' + factor(tuning['selected_d_lr_factor']))
        elif dynamics == 'kept_baseline':
            parts.append('Configured G and D learning rates passed' if factor(tuning.get('selected_d_lr_factor'))
                         else 'Configured G learning rate passed')
        elif dynamics == 'unresolved':
            parts[0] = 'Startup tuning unresolved'
            parts.append('No rate candidate passed; kept configured G and D learning rates' if factor(tuning.get('selected_d_lr_factor'))
                         else 'No rate candidate passed; kept configured G learning rate')
        elif dynamics == 'skipped':
            parts.append('Dynamics check skipped')
        reason = tuning.get('dynamics_reason')
        if isinstance(reason, str) and reason.strip():
            parts.append(brief(reason))
        warmup = tuning.get('g_lr_warmup')
        if isinstance(warmup, dict) and type(warmup.get('steps')) is int and warmup['steps'] >= 2:
            parts.append(f"G learning-rate warmup enabled for {warmup['steps']} retained updates")
    message = tuning.get('message')
    if isinstance(message, str) and message.strip():
        parts.append(brief(message))
    return ' | '.join(parts)


def _put_latest(pending, value):
    try:
        pending.put_nowait(value)
        return False
    except queue.Full:
        try:
            pending.get_nowait()
        except queue.Empty:
            pass
        while True:
            try:
                pending.put_nowait(value)
                return True
            except queue.Full:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    pass


def _write_all(fd, data):
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise BrokenPipeError('Output pipe closed')
        data = data[written:]


def _drain(parent_pid):
    """Child reader never waits for its destination writer or a full queue."""
    pending = queue.Queue(MAX_PENDING_LINES)
    ended = threading.Event()

    def watch():
        # Opening the Windows process handle once also fences PID reuse.
        handle = None
        if os.name == 'nt':
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            handle = kernel.OpenProcess(0x100000, False, parent_pid)
            if not handle:
                os._exit(0)
        while not ended.wait(0.1):
            alive = (kernel.WaitForSingleObject(handle, 0) == 258 if handle
                     else os.getppid() == parent_pid)
            if not alive:
                os._exit(0)
        # EOF must release a writer stuck in an unread destination too.
        time.sleep(CLOSE_SECONDS)
        os._exit(0)

    def read():
        partial = bytearray()
        oversize = False
        try:
            while True:
                chunk = os.read(0, MAX_LINE_BYTES)
                if not chunk:
                    break
                for piece in chunk.splitlines(keepends=True):
                    if not oversize:
                        if len(partial) + len(piece) <= MAX_LINE_BYTES:
                            partial.extend(piece)
                        else:
                            partial.clear()
                            oversize = True
                    if piece.endswith(b'\n'):
                        if not oversize:
                            _put_latest(pending, bytes(partial))
                        partial.clear()
                        oversize = False
            if partial and not oversize:
                _put_latest(pending, bytes(partial))
        except OSError:
            pass
        finally:
            # Preserve queued terminal output while bounding blocked writes.
            ended.set()

    threading.Thread(target=watch, daemon=True).start()
    threading.Thread(target=read, daemon=True).start()
    try:
        while not (ended.is_set() and pending.empty()):
            try:
                line = pending.get(timeout=0.05)
            except queue.Empty:
                continue
            _write_all(1, line)
    except OSError:
        # The receiver continues until EOF or coordinator death; native writers
        # must not receive SIGPIPE just because the user's terminal disappeared.
        while not ended.wait(0.05):
            pass


class _Stream:
    encoding = 'utf-8'
    errors = 'replace'

    def __init__(self, original, fd):
        self.original, self.fd = original, fd
        self.pending = queue.Queue(MAX_PENDING_LINES)
        self.partial = ''
        self.oversize = False
        self.dropped_lines = 0
        self.closed = False
        self.lock = threading.Lock()
        self.process = self.feeder = None
        self.saved_fd = None
        try:
            descriptor = original.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # Real in-memory capture streams cannot block on external I/O.
            if not (type(original) is io.StringIO or
                    isinstance(original, io.TextIOWrapper) and type(original.buffer) is io.BytesIO):
                raise ValueError('Training output requires a file descriptor or an in-memory text stream')
            self.memory = original
            return
        self.memory = None
        self.saved_fd = os.dup(fd)
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'hypergan.bounded_cli_output', '--drain', str(os.getpid())],
                stdin=subprocess.PIPE, stdout=descriptor, stderr=subprocess.DEVNULL,
                bufsize=0, close_fds=True,
                # Keep terminal-group stop signals on the coordinator so its
                # complete-boundary shutdown can still publish final output.
                start_new_session=os.name != 'nt')
            # Native libraries and spawned workers use the independently drained
            # pipe too. Python writes use the bounded queue below.
            os.dup2(self.process.stdin.fileno(), fd)
            self.feeder = threading.Thread(target=self._feed, daemon=True)
            self.feeder.start()
        except BaseException:
            self.close(time.monotonic())
            raise

    def _feed(self):
        try:
            while True:
                data = self.pending.get()
                if data is None:
                    return
                _write_all(self.process.stdin.fileno(), data)
        except (OSError, ValueError):
            pass

    def _line(self, text):
        encoded = text.encode('utf-8', errors='replace')
        if len(encoded) > MAX_LINE_BYTES:
            self.dropped_lines += 1
            return
        if self.memory is not None:
            self.memory.write(text)
        elif _put_latest(self.pending, encoded):
            self.dropped_lines += 1

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError('Output must be text')
        with self.lock:
            if self.closed:
                return len(text)
            for piece in text.splitlines(keepends=True):
                if not self.oversize:
                    # Reject oversized character strings before encoding, then
                    # bound the partial line by its actual UTF-8 size too.
                    if (len(self.partial) + len(piece) <= MAX_LINE_BYTES and
                            len(self.partial.encode('utf-8', errors='replace')) +
                            len(piece.encode('utf-8', errors='replace')) <= MAX_LINE_BYTES):
                        self.partial += piece
                    else:
                        self.partial = ''
                        self.oversize = True
                        self.dropped_lines += 1
                if piece.endswith('\n'):
                    if not self.oversize:
                        self._line(self.partial)
                    self.partial = ''
                    self.oversize = False
        return len(text)

    def flush(self):
        with self.lock:
            if not self.closed and self.partial:
                self._line(self.partial + '\n')
                self.partial = ''

    def isatty(self):
        return False

    def fileno(self):
        return self.fd

    def close(self, deadline):
        self.flush()
        self.closed = True
        if self.saved_fd is not None:
            os.dup2(self.saved_fd, self.fd)
            os.close(self.saved_fd)
            self.saved_fd = None
        if self.process is None:
            return
        if self.feeder is not None:
            # A sentinel follows queued data; if full, sacrifice oldest output.
            _put_latest(self.pending, None)
            self.feeder.join(max(0, deadline - time.monotonic()))
        if self.feeder is None or not self.feeder.is_alive():
            self.process.stdin.close()
        try:
            self.process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)
        # Killing a child closes its reader and releases any blocked raw write.
        if self.feeder is not None:
            self.feeder.join(1.0)
        self.process.stdin.close()


def _native_standard_streams():
    """Resolve only the owning C runtime's standard stdout/stderr streams.

    Never fflush(NULL): unrelated FILEs may target unrelated blocking devices.
    Custom native streams, separate CRTs and externally held FILE locks are
    not managed here.
    """
    import ctypes

    libraries = []
    if os.name == 'nt':
        modern = ctypes.CDLL('ucrtbase')
        modern.__acrt_iob_func.argtypes = [ctypes.c_uint]
        modern.__acrt_iob_func.restype = ctypes.c_void_p
        libraries.append((modern, [modern.__acrt_iob_func(index) for index in (1, 2)]))
    else:
        libc = ctypes.CDLL(None)
        names = ('__stdoutp', '__stderrp') if sys.platform == 'darwin' else ('stdout', 'stderr')
        libraries.append((libc, [ctypes.c_void_p.in_dll(libc, name).value for name in names]))
    for library, streams in libraries:
        if any(not stream for stream in streams):
            raise RuntimeError('Native stdout/stderr pointer is unavailable')
        library.fflush.argtypes = [ctypes.c_void_p]
        library.fflush.restype = ctypes.c_int
    return libraries


class _NativeStreams:
    def __init__(self):
        self.libraries = _native_standard_streams()

    def flush(self):
        for library, streams in self.libraries:
            for stream in streams:
                library.fflush(stream)


def _flush_cached_streams(originals):
    # Some dependencies cache the original objects (or sys.__stdout__) before
    # CLI redirection. Their buffers must not survive restoration of a full
    # terminal pipe. They still point at the independently drained FDs here.
    seen = set()
    for original in (*originals, sys.__stdout__, sys.__stderr__):
        if original is None or id(original) in seen:
            continue
        seen.add(id(original))
        try:
            original.flush()
        except (OSError, ValueError):
            # A disconnected/caller-closed stream is optional output as well.
            pass


class CLIProgress:
    """Exact internal sink type allowed to bypass arbitrary callback isolation."""
    def __init__(self, output):
        self.output = output
        self.progress_lines = 0

    def __call__(self, event):
        self.output.policy.refresh(self.output.stderr)
        warmup = event.get('g_lr_warmup', {})
        warmup_boundary = isinstance(warmup, dict) and bool(warmup) and event.get('step') in (1, warmup.get('steps'))
        if event.get('event') == 'train' and event['step'] % self.output.policy.every and not warmup_boundary:
            return
        reminder = None
        if event.get('event') == 'train' and self.output.evaluation_reminder:
            if not self.progress_lines % REMINDER_EVERY_PROGRESS_LINES:
                reminder = self.output.evaluation_reminder
            self.progress_lines += 1
        if self.output.progress_json:
            # An added field only; existing progress keys and their meaning are unchanged.
            row = dict(event, evaluation_reminder=reminder) if reminder else event
            self.output.stdout.write(json.dumps(row, allow_nan=False) + '\n')
        elif event.get('event') == 'tuning':
            self.output.stderr.write(_tuning_message(event.get('tuning', {})) + '\n')
        elif event.get('event') == 'train':
            metrics = event.get('metrics', {})
            values = ' '.join(f'{label}={metrics[key]:.6g}' for key, label in
                              (('loss/d_total', 'D'), ('loss/g_total', 'G')) if key in metrics)
            details = ' | '.join(_progress_details(event))
            self.output.stderr.write(f"step {event['step']}" + (f': {values}' if values else '')
                                     + (f' | {details}' if details else '') + '\n')
            if reminder:
                self.output.stderr.write(f'reminder: {reminder}\n')

        else:
            message = event.get('error') or event.get('reason') or event.get('stop_reason')
            self.output.stderr.write(f"{event.get('event', 'status')} at step {event.get('step', '?')}" +
                                     (f": {message}" if message else '') + '\n')

    def deliver(self, event):
        self(event)
        return True

    @property
    def disabled(self):
        return False

    def close(self):
        # The CLI owns the lifetime, including the final result after execution.
        pass


class TrainingOutput:
    def __init__(self, stdout, stderr, progress_json, progress_every=None):
        from .console_settings import ConsolePolicy
        self.policy = ConsolePolicy(progress_every=progress_every)
        self.stdout, self.stderr = stdout, stderr
        self.progress_json = progress_json
        self.evaluation_reminder = None
        self.progress = CLIProgress(self)

    def configure(self, run_dir, *, progress_every=None, evaluation_reminder=None):
        from .console_settings import ConsolePolicy
        self.policy.close()
        self.policy = ConsolePolicy(run_dir, progress_every=progress_every)
        self.evaluation_reminder = evaluation_reminder

    def result(self, result, *, run_dir=None):
        value = {'event': 'result', 'manifest': result} if self.progress_json else result
        encoded = json.dumps(value, allow_nan=False, sort_keys=not self.progress_json) + '\n'
        if len(encoded.encode('utf-8')) > MAX_LINE_BYTES:
            self.stderr.write('warning: CLI result exceeds 65536 bytes; read the durable run manifest.json\n')
            encoded = json.dumps({'event': 'output_omitted', 'reason': 'result_exceeds_output_limit',
                                  'read': str(os.path.join(run_dir, 'manifest.json')) if run_dir is not None else 'manifest.json in the run directory'}) + '\n'
        self.stdout.write(encoded)


@contextmanager
def training_output(*, progress_json=False, progress_every=None):
    """Bound all train/resume output; close and reap drains within finite grace.

    Per stream: 16 queued lines in the parent, 16 in the child, one partial and
    one in-flight line per stage; each line is at most 64 KiB. Full queues drop
    oldest lines, oversized lines are omitted, and closed consumers do not fail
    training. Shutdown gets one shared second plus forced process cleanup.
    """
    originals = sys.stdout, sys.stderr
    stdout = stderr = native = output = None
    try:
        native = _NativeStreams()
        stdout = _Stream(originals[0], 1)
        stderr = _Stream(originals[1], 2)
        sys.stdout, sys.stderr = stdout, stderr
        output = TrainingOutput(stdout, stderr, progress_json, progress_every)
        yield output
    finally:
        if output is not None:
            output.policy.close()
        # Flush cached Python and C stdio while both independent input drains
        # are alive. Otherwise interpreter/libc finalization could write those
        # buffers into the restored, already full destination and hang exit.
        try:
            if stdout is not None and stderr is not None:
                _flush_cached_streams(originals)
                native.flush()
        finally:
            sys.stdout, sys.stderr = originals
            deadline = time.monotonic() + CLOSE_SECONDS
            try:
                if stdout is not None:
                    stdout.close(deadline)
            finally:
                if stderr is not None:
                    stderr.close(deadline)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--drain':
        raise SystemExit('Internal output drain requires a parent PID')
    _drain(int(sys.argv[2]))
