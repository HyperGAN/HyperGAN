"""Ephemeral RunPod qualification runner: GPU selection, cost guardrails and pod cleanup.

Experimental. The live mode is disabled: the runner cannot yet run ``hypergan qualify``
on a pod and fetch the receipt it writes, so a live run would pay for a pod that does
nothing. ``RunPodRunner`` therefore refuses live mode with ``LiveModeUnavailable``
before it creates anything.

What works today:
- ``--dry-run`` plans a run without contacting RunPod: it selects the cheapest GPU that
  meets the VRAM and cost limits from a built-in, illustrative price list, checks the
  180-second lifetime limit, and writes a receipt whose status is ``dry_run``. It creates
  no pod and reports no suite as passed.
- With an injected client that implements ``execute_qualification(pod_id)``, the runner
  drives the pod lifecycle and validates the receipt the client returns. It never fills
  in missing results.
- Cleanup: a pod is dropped from the exit/signal cleanup registry only once its
  termination is confirmed. When termination cannot be confirmed the pod stays
  registered and the run fails with ``PodCleanupError`` naming the pod. The SIGINT and
  SIGTERM handlers are installed only while a pod is registered, chain to the previous
  handlers, and are restored afterwards.
- ``RunPodHttpClient`` speaks the RunPod GraphQL API with the standard library, sends the
  API key only in the Authorization header, retries termination, and looks for an orphaned
  pod by its unique name when a create request fails ambiguously (e.g. a timeout).
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import uuid

from .qualify import QualificationRunner, compute_receipt_sha256, default_receipt_path, validate_qualification_receipt, write_receipt

# ---------------------------------------------------------------------------
# Constants & Guardrails
# ---------------------------------------------------------------------------

MAX_ALLOWED_TIMEOUT: int = 180
DEFAULT_TIMEOUT: int = 180
DEFAULT_MIN_VRAM_GB: int = 16
DEFAULT_POD_IMAGE: str = "hypergan/hypergan:latest"
RUNPOD_GRAPHQL_ENDPOINT: str = "https://api.runpod.io/graphql"
RUNPOD_CONSOLE_URL: str = "https://www.runpod.io/console/pods"

LIVE_MODE_UNAVAILABLE_MESSAGE: str = (
    "Live RunPod qualification is not implemented: the runner cannot yet run `hypergan qualify` "
    "on the pod and fetch its receipt, so it refuses to create a paid pod. Use --dry-run to plan a run."
)

# Illustrative prices for dry-run planning only; a live client must supply its own catalog.
DEFAULT_GPU_CATALOG: List[Dict[str, Any]] = [
    {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
    {"id": "NVIDIA RTX A4500", "displayName": "RTX A4500", "vram_gb": 20, "cost_per_hr": 0.34, "available": True},
    {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
    {"id": "NVIDIA RTX 4090", "displayName": "RTX 4090", "vram_gb": 24, "cost_per_hr": 0.69, "available": True},
    {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "vram_gb": 48, "cost_per_hr": 0.79, "available": True},
    {"id": "NVIDIA A100 80GB PCIe", "displayName": "A100 PCIe", "vram_gb": 80, "cost_per_hr": 1.89, "available": True},
    {"id": "NVIDIA H100 SXM", "displayName": "H100 SXM", "vram_gb": 80, "cost_per_hr": 3.89, "available": True},
]


class LiveModeUnavailable(NotImplementedError):
    """Live RunPod execution was requested but cannot run qualification on a pod."""


class PodCleanupError(RuntimeError):
    """A pod's termination could not be confirmed; it may still be running and billing."""

    def __init__(self, pod_id: str, cause: Any):
        self.pod_id = pod_id
        super().__init__(
            f"Could not confirm termination of RunPod pod {pod_id} ({cause}); it may still be running "
            f"and billing. Terminate it manually at {RUNPOD_CONSOLE_URL}"
        )


class RunPodApiError(RuntimeError):
    """A RunPod API request failed. ``definitive`` is True when RunPod rejected it outright."""

    def __init__(self, message: str, definitive: bool):
        super().__init__(message)
        self.definitive = definitive


# ---------------------------------------------------------------------------
# Guardrail Validators
# ---------------------------------------------------------------------------

def validate_timeout_guardrail(requested_timeout: int) -> int:
    """Enforce hard 180s timeout limit on any RunPod cloud session.

    Args:
        requested_timeout: Proposed timeout in seconds.

    Returns:
        The validated timeout integer.

    Raises:
        ValueError: If requested_timeout <= 0 or requested_timeout > 180.
    """
    if requested_timeout <= 0 or requested_timeout > MAX_ALLOWED_TIMEOUT:
        raise ValueError(
            f"RunPod timeout must be between 1 and 180 seconds ({MAX_ALLOWED_TIMEOUT}s), got {requested_timeout}"
        )
    return requested_timeout


def select_cheapest_gpu(
    catalog: List[Dict[str, Any]],
    min_vram_gb: int = DEFAULT_MIN_VRAM_GB,
    gpu_type: Optional[str] = None,
    target_gpu: Optional[str] = None,
    max_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Select the lowest cost GPU from catalog satisfying constraints.

    Args:
        catalog: List of GPU type dictionaries.
        min_vram_gb: Minimum required VRAM in GB (default: 16).
        gpu_type: Optional substring filter for GPU type name or ID.
        target_gpu: Optional alias for gpu_type.
        max_cost: Optional maximum cost per hour in USD ($/hr).

    Returns:
        The matching GPU dictionary with the lowest cost_per_hr.

    Raises:
        RuntimeError: If no qualified GPU is found meeting the constraints.
    """
    target = target_gpu or gpu_type
    qualified: List[Dict[str, Any]] = []

    for g in catalog:
        if not g.get("available", True):
            continue
        vram = g.get("vram_gb", 0)
        if vram < min_vram_gb:
            continue
        if target:
            t = target.strip().lower()
            gid = str(g.get("id", "")).lower()
            name = str(g.get("displayName", "")).lower()
            if t not in gid and t not in name:
                continue
        if max_cost is not None:
            cost = g.get("cost_per_hr", float("inf"))
            if cost > max_cost:
                continue
        qualified.append(g)

    if not qualified:
        reasons = [f"min_vram_gb={min_vram_gb}"]
        if target:
            reasons.append(f"target='{target}'")
        if max_cost is not None:
            reasons.append(f"max_cost=${max_cost}/hr")
        raise RuntimeError(f"No qualified RunPod GPUs available satisfying constraints: {', '.join(reasons)}")

    return min(qualified, key=lambda g: g["cost_per_hr"])


def terminate_pod_confirmed(client: Any, pod_id: str) -> None:
    """Terminate a pod and require the client to confirm it; raise PodCleanupError otherwise."""
    try:
        confirmed = client.terminate_pod(pod_id)
    except PodCleanupError:
        raise
    except Exception as exc:
        raise PodCleanupError(pod_id, exc) from exc
    if not confirmed:
        raise PodCleanupError(pod_id, "the client did not confirm termination")


# ---------------------------------------------------------------------------
# Active Pod Cleanup Trap Registry
# ---------------------------------------------------------------------------

class _ActivePodRegistry:
    """Pods whose termination is not yet confirmed, cleaned up at exit and on SIGINT/SIGTERM.

    Signal handlers are installed when the first pod registers, chain to the handlers
    they replaced, and are restored once the last pod's termination is confirmed.
    """

    _SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self) -> None:
        self._active_pods: Dict[str, Any] = {}
        # Re-entrant: a signal handler runs on the main thread, possibly inside a locked block.
        self._lock = threading.RLock()
        self._atexit_installed = False
        self._previous_handlers: Dict[int, Any] = {}

    def register(self, pod_id: str, client: Any) -> None:
        with self._lock:
            self._active_pods[pod_id] = client
            if not self._atexit_installed:
                atexit.register(self.cleanup_all)
                self._atexit_installed = True
            if not self._previous_handlers:
                self._install_signal_handlers()

    def is_registered(self, pod_id: str) -> bool:
        with self._lock:
            return pod_id in self._active_pods

    def unregister(self, pod_id: str) -> None:
        with self._lock:
            self._active_pods.pop(pod_id, None)
            if not self._active_pods:
                self._restore_signal_handlers()

    def release(self, pod_id: str) -> None:
        """Terminate a registered pod and unregister it once termination is confirmed.

        A pod that is not registered (already released) is left alone. If termination
        cannot be confirmed the pod stays registered and PodCleanupError propagates.
        """
        with self._lock:
            client = self._active_pods.get(pod_id)
        if client is None:
            return
        terminate_pod_confirmed(client, pod_id)
        self.unregister(pod_id)

    def cleanup_all(self) -> List[str]:
        """Release every registered pod; return the ids whose termination failed."""
        with self._lock:
            pod_ids = list(self._active_pods)
        failures = []
        for pod_id in pod_ids:
            try:
                self.release(pod_id)
            except PodCleanupError as exc:
                failures.append(pod_id)
                sys.stderr.write(f"error: {exc}\n")
        return failures

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in self._SIGNALS:
            try:
                previous = signal.getsignal(sig)
                if previous == signal.SIG_IGN:
                    continue
                signal.signal(sig, self._on_signal)
                self._previous_handlers[sig] = previous
            except (ValueError, OSError):
                pass

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, previous in list(self._previous_handlers.items()):
            try:
                # Leave alone a handler someone else installed after ours.
                if signal.getsignal(sig) == self._on_signal:
                    signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        self._previous_handlers.clear()

    def _on_signal(self, signum: int, frame: Any) -> None:
        previous = self._previous_handlers.get(signum)
        self.cleanup_all()
        if callable(previous):
            previous(signum, frame)
        else:
            sys.exit(128 + signum)


_GLOBAL_POD_REGISTRY = _ActivePodRegistry()


# ---------------------------------------------------------------------------
# Ephemeral Context Manager
# ---------------------------------------------------------------------------

@contextmanager
def provision_ephemeral_pod(
    client: Any,
    gpu_type: str,
    image: str = DEFAULT_POD_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> Iterator[str]:
    """Create a pod and terminate it on normal exit, exception, or signal.

    The pod stays in the cleanup registry until its termination is confirmed; when it
    cannot be confirmed, PodCleanupError propagates from the context manager.

    Args:
        client: Client instance implementing create_pod and terminate_pod.
        gpu_type: GPU type identifier.
        image: Container image.
        timeout_seconds: Hard timeout in seconds (1..180).

    Yields:
        Provisioned pod identifier string.
    """
    valid_timeout = validate_timeout_guardrail(timeout_seconds)
    pod_id = client.create_pod(gpu_type=gpu_type, image=image, timeout_seconds=valid_timeout)
    _GLOBAL_POD_REGISTRY.register(pod_id, client)
    try:
        yield pod_id
    finally:
        _GLOBAL_POD_REGISTRY.release(pod_id)


# ---------------------------------------------------------------------------
# Live HTTP Client
# ---------------------------------------------------------------------------

class RunPodHttpClient:
    """Zero-dependency RunPod GraphQL API client using standard library urllib.

    ``urlopen`` and ``sleep`` are injectable so tests can supply a fake transport.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = RUNPOD_GRAPHQL_ENDPOINT,
        urlopen: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout: float = 30.0,
        terminate_attempts: int = 4,
        retry_backoff_seconds: float = 2.0,
    ):
        resolved_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not resolved_key:
            raise ValueError("RUNPOD_API_KEY is required for the RunPod API")
        self.api_key = resolved_key
        self.endpoint = endpoint
        if urlopen is None:
            import urllib.request
            urlopen = urllib.request.urlopen
        self._urlopen = urlopen
        self._sleep = sleep
        self.request_timeout = request_timeout
        self.terminate_attempts = max(1, terminate_attempts)
        self.retry_backoff_seconds = retry_backoff_seconds

    def _execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import urllib.error
        import urllib.request

        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        # The key travels only in the header: query strings end up in proxy and access logs.
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "HyperGAN-RunPod-Runner/1.0",
            },
            method="POST",
        )
        try:
            with self._urlopen(req, timeout=self.request_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            # A 4xx is a rejection; a 5xx may have been applied before the server failed.
            raise RunPodApiError(f"RunPod HTTP {exc.code}: {err_body}", definitive=400 <= exc.code < 500) from exc
        except (OSError, ValueError) as exc:
            raise RunPodApiError(f"RunPod request failed: {exc}", definitive=False) from exc
        if data.get("errors"):
            raise RunPodApiError(f"RunPod API error: {data['errors']}", definitive=True)
        return data.get("data") or {}

    def get_gpu_catalog(self) -> List[Dict[str, Any]]:
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                lowestPrice(input: { gpuCount: 1 }) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        data = self._execute(query)
        catalog = []
        for g in data.get("gpuTypes", []):
            lp = g.get("lowestPrice") or {}
            price = lp.get("uninterruptablePrice") or lp.get("minimumBidPrice")
            if price is not None:
                catalog.append({
                    "id": g["id"],
                    "displayName": g.get("displayName", g["id"]),
                    "vram_gb": g.get("memoryInGb", 0),
                    "cost_per_hr": float(price),
                    "available": True,
                })
        if not catalog:
            # Never choose a "cheapest" GPU from invented prices.
            raise RuntimeError("RunPod returned no priced GPU types; refusing to select a GPU")
        return catalog

    def find_pods_named(self, name: str) -> List[str]:
        query = """
        query Pods {
            myself {
                pods {
                    id
                    name
                }
            }
        }
        """
        data = self._execute(query)
        pods = (data.get("myself") or {}).get("pods") or []
        return [p["id"] for p in pods if p.get("name") == name and p.get("id")]

    def create_pod(self, gpu_type: str, image: str = DEFAULT_POD_IMAGE, timeout_seconds: int = DEFAULT_TIMEOUT) -> str:
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                desiredStatus
                imageName
            }
        }
        """
        # A unique name lets an ambiguous failure (e.g. a timeout) be traced to the pod it may have created.
        name = f"hypergan-qualify-{uuid.uuid4().hex[:12]}"
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                # No persistent volume: a stopped pod then holds no billed storage.
                "volumeInGb": 0,
                "containerDiskInGb": 20,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuTypeId": gpu_type,
                "name": name,
                "imageName": image,
                # Server-side backstop. Unverified against the live API, and it stops
                # rather than terminates; the runner's own termination is the real guard.
                "stopAfter": timeout_seconds,
            }
        }
        try:
            data = self._execute(mutation, variables)
        except RunPodApiError as exc:
            if exc.definitive:
                raise
            self._reap_orphans(name, exc)
            raise
        pod_info = data.get("podFindAndDeployOnDemand") or {}
        pod_id = pod_info.get("id")
        if not pod_id:
            raise RuntimeError(f"RunPod returned no pod ID on creation: {data}")
        return pod_id

    def _reap_orphans(self, name: str, cause: Exception) -> None:
        """After an ambiguous create failure, terminate any pod that was created anyway."""
        try:
            orphans = self.find_pods_named(name)
        except Exception as exc:
            raise PodCleanupError(f"named {name}", f"creation failed ambiguously ({cause}) and the pod list "
                                  f"could not be read ({exc})") from exc
        for pod_id in orphans:
            sys.stderr.write(f"warning: terminating pod {pod_id} created by a failed request ({cause})\n")
            self.terminate_pod(pod_id)

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod, retrying with backoff; raise PodCleanupError if it never succeeds."""
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        variables = {"input": {"podId": pod_id}}
        last_exc: Optional[Exception] = None
        for attempt in range(self.terminate_attempts):
            try:
                self._execute(mutation, variables)
                return True
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.terminate_attempts:
                    self._sleep(self.retry_backoff_seconds * (2 ** attempt))
        raise PodCleanupError(pod_id, f"{self.terminate_attempts} attempts failed, last: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Primary Runner: RunPodRunner
# ---------------------------------------------------------------------------

def _qualification_executor(client: Any) -> Callable[[str], Any]:
    """The client's execute_qualification; refuse before anything is created when it has none."""
    execute = getattr(client, "execute_qualification", None)
    if not callable(execute):
        # A pod that runs nothing only costs money.
        raise LiveModeUnavailable(
            f"{type(client).__name__} cannot run qualification on a pod (no execute_qualification); "
            "refusing to create one"
        )
    return execute


class RunPodRunner:
    """Ephemeral RunPod runner with strict cost guardrails and confirmed cleanup.

    ``dry_run=True`` without a client plans a run and never contacts RunPod. A client that
    implements ``execute_qualification(pod_id)`` drives a real pod lifecycle. Live mode
    without such a client raises LiveModeUnavailable.
    """

    def __init__(
        self,
        dry_run: bool = False,
        gpu_type: Optional[str] = None,
        max_cost: Optional[float] = None,
        timeout: int = DEFAULT_TIMEOUT,
        output: Optional[Union[str, Path]] = None,
        min_vram_gb: int = DEFAULT_MIN_VRAM_GB,
        client: Optional[Any] = None,
        image: str = DEFAULT_POD_IMAGE,
        verbose: bool = False,
        log_to_stderr: bool = False,
    ):
        self.timeout = validate_timeout_guardrail(timeout)
        if client is None and not dry_run:
            raise LiveModeUnavailable(LIVE_MODE_UNAVAILABLE_MESSAGE)
        self.dry_run = dry_run
        self.gpu_type = gpu_type
        self.max_cost = max_cost
        self.output = Path(output) if output else None
        self.min_vram_gb = min_vram_gb
        self.image = image
        self.verbose = verbose
        self.log_to_stderr = log_to_stderr
        self.lifecycle_history: List[str] = []
        self.last_output_path: Optional[Path] = None
        self._active_pod_id: Optional[str] = None
        # None in a plain dry run: nothing that could create a pod exists.
        self.client = client

    def _log(self, message: str) -> None:
        """Emit diagnostic message when verbose is enabled."""
        if self.verbose:
            stream = sys.stderr if self.log_to_stderr else sys.stdout
            stream.write(f"[runpod] {message}\n")
            stream.flush()

    def filter_and_select_gpu(self, catalog: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Select lowest cost GPU from catalog, enforcing max_cost guardrail."""
        if catalog is None:
            if self.client is not None and hasattr(self.client, "get_gpu_catalog"):
                catalog = self.client.get_gpu_catalog()
            else:
                catalog = DEFAULT_GPU_CATALOG

        chosen = select_cheapest_gpu(
            catalog,
            min_vram_gb=self.min_vram_gb,
            gpu_type=self.gpu_type,
        )

        if self.max_cost is not None and chosen.get("cost_per_hr", float("inf")) > self.max_cost:
            raise ValueError(
                f"Selected GPU {chosen['id']} (${chosen['cost_per_hr']}/hr) exceeds --max-cost ceiling (${self.max_cost}/hr)"
            )
        self._log(
            f"Selected GPU '{chosen['id']}' (${chosen.get('cost_per_hr', 0.0):.2f}/hr, "
            f"{chosen.get('vram_gb', 0)}GB VRAM)"
        )
        return chosen

    def select_gpu(self) -> Dict[str, Any]:
        """Convenience alias for filter_and_select_gpu."""
        return self.filter_and_select_gpu()

    @contextmanager
    def ephemeral_pod(self, gpu_type: str, timeout_seconds: Optional[int] = None) -> Iterator[str]:
        if self.client is None:
            raise LiveModeUnavailable("A dry run has no RunPod client and never creates a pod")
        timeout = validate_timeout_guardrail(timeout_seconds or self.timeout)
        try:
            with provision_ephemeral_pod(
                client=self.client,
                gpu_type=gpu_type,
                image=self.image,
                timeout_seconds=timeout,
            ) as pod_id:
                self._active_pod_id = pod_id
                self.lifecycle_history.extend(["created", "running"])
                yield pod_id
        finally:
            # A pod whose termination was not confirmed stays registered and visible here.
            if self._active_pod_id is not None and not _GLOBAL_POD_REGISTRY.is_registered(self._active_pod_id):
                self._active_pod_id = None
                self.lifecycle_history.append("terminated")

    def execute_with_lifecycle(self, client: Any, gpu_type: str) -> Dict[str, Any]:
        """Run qualification on a pod through ``client`` and return its validated receipt."""
        execute = _qualification_executor(client)
        previous_client, self.client = self.client, client
        try:
            start = time.monotonic()
            with self.ephemeral_pod(gpu_type=gpu_type) as pod_id:
                receipt = execute(pod_id)
            elapsed = time.monotonic() - start
        finally:
            self.client = previous_client
        if not isinstance(receipt, dict):
            raise ValueError(f"Remote qualification returned {type(receipt).__name__}, not a receipt object")
        # The remote receipt must stand on its own; nothing is filled in or defaulted.
        validate_qualification_receipt(receipt)
        if elapsed > self.timeout:
            raise RuntimeError(f"Qualification duration ({elapsed:.2f}s) exceeded hard {self.timeout}s limit")
        return receipt

    def _dry_run_receipt(self, gpu: Dict[str, Any], suite: Optional[str]) -> Dict[str, Any]:
        """Receipt for a planned run: every suite reports dry_run and no pod is created."""
        receipt = QualificationRunner(dry_run=True, suite_filter=suite, timeout=self.timeout).execute()
        receipt["environment"]["runpod"] = {
            "dry_run": True,
            "pod_created": False,
            "gpu_type": gpu["id"],
            "cost_per_hr": gpu.get("cost_per_hr"),
            "price_source": "built-in illustrative catalog" if self.client is None else "client catalog",
            "image": self.image,
            "timeout_seconds": self.timeout,
            "note": "dry run: no pod was created and no suite ran; environment describes the local host",
        }
        receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
        return receipt

    def run_qualification_suite(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        suite: Optional[str] = None,
        catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute ephemeral qualification suite and emit receipt JSON.

        Args:
            output_dir: Optional receipt file path, or a directory to hold qualification-receipt.json.
            suite: Optional suite filter.
            catalog: Optional catalog override for GPU selection.

        Returns:
            The complete, validated qualification receipt dictionary.
        """
        # Reject an unknown suite or an incapable client before any API call or pod creation.
        QualificationRunner(dry_run=True, suite_filter=suite)
        if self.client is not None:
            _qualification_executor(self.client)
        chosen_gpu = self.filter_and_select_gpu(catalog)
        gpu_id = chosen_gpu["id"]
        self._log(
            f"Starting ephemeral qualification (dry_run={self.client is None}, timeout={self.timeout}s, suite={suite or 'all'})"
        )
        if self.client is None:
            receipt = self._dry_run_receipt(chosen_gpu, suite)
        else:
            receipt = self.execute_with_lifecycle(self.client, gpu_id)

        validate_qualification_receipt(receipt)
        self._log(
            f"Qualification completed: status={receipt.get('status')} in {receipt.get('duration_seconds')}s"
        )

        dest = output_dir or self.output
        if dest:
            dest_path = Path(dest)
            if dest_path.is_dir() or not dest_path.suffix:
                dest_path = dest_path / "qualification-receipt.json"
            self.last_output_path = write_receipt(receipt, dest_path)

        return receipt


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entrypoint
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypergan.runpod_runner",
        description=(
            "Plan an ephemeral RunPod hardware qualification with strict cost and lifetime guardrails. "
            "Experimental: only --dry-run is available; live mode is refused until the runner can run "
            "`hypergan qualify` on the pod."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the run (GPU choice, guardrails) and write a dry_run receipt without contacting RunPod",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=None,
        help="optional target GPU type filter (e.g. 'RTX A4000', 'A5000')",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="maximum allowable instance cost in USD/hr",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"hard timeout in seconds (1..{MAX_ALLOWED_TIMEOUT}, default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="path to write emitted JSON receipt (default: a per-run file under the HyperGAN state directory)",
    )
    parser.add_argument(
        "--min-vram",
        type=int,
        default=DEFAULT_MIN_VRAM_GB,
        help=f"minimum required VRAM in GB (default: {DEFAULT_MIN_VRAM_GB})",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_POD_IMAGE,
        help=f"pod container image (default: {DEFAULT_POD_IMAGE})",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="qualification suite filter",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable verbose diagnostic logging and progress output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output receipt JSON to stdout",
    )
    parser.epilog = "The RunPod API key is read from the RUNPOD_API_KEY environment variable, never from argv."
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        runner = RunPodRunner(
            dry_run=args.dry_run,
            gpu_type=args.gpu_type,
            max_cost=args.max_cost,
            timeout=args.timeout,
            min_vram_gb=args.min_vram,
            image=args.image,
            verbose=args.verbose,
            log_to_stderr=args.json,
        )
        receipt = runner.run_qualification_suite(output_dir=args.output, suite=args.suite)
        target = runner.last_output_path or write_receipt(receipt, default_receipt_path(receipt["qualification_id"]))
        if args.json:
            print(json.dumps(receipt, indent=2))
        else:
            print(f"Qualification status: {receipt['status']} (duration: {receipt['duration_seconds']}s)")
            if args.verbose and "suites" in receipt:
                symbols = {"passed": "✓", "skipped": "⊘", "dry_run": "-"}
                for s in receipt.get("suites", []):
                    duration_str = f" ({s['duration_seconds']}s)" if "duration_seconds" in s else ""
                    print(f"  [{symbols.get(s.get('status'), '✗')}] {s['name']}: {s.get('status')}{duration_str}")
            print(f"Receipt written to: {target}")
        return 0 if receipt["status"] in ("passed", "dry_run") else 1
    except LiveModeUnavailable as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
