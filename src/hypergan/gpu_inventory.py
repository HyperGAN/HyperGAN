"""GPU Inventory Provider Abstraction for HyperGAN.

Supports nvidia-smi, amd-smi, and rocm-smi with normalized hardware and process records.
"""
from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WARNED_MISSING_TOOL = False


def reset_inventory_warning_state():
    """Reset the single warning log guard (primarily for unit tests)."""
    global _WARNED_MISSING_TOOL
    _WARNED_MISSING_TOOL = False


def get_proc_cmdline(pid: int) -> str:
    """Read command line string for a process from /proc/<pid>/cmdline."""
    cmd_path = Path(f"/proc/{pid}/cmdline")
    if cmd_path.exists():
        try:
            return cmd_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    return ""


def parse_memory_to_mib(val: Any, default_unit: str = "MB") -> float:
    """Parse various memory representations into float MiB/MB.

    Supports floats/ints, dicts with {'value': ..., 'unit': ...}, and
    unit-suffixed strings like '24.0 GB', '24000.0 MB', '104857600 B'.
    """
    if val is None:
        return 0.0

    if isinstance(val, dict):
        value = float(val.get("value", 0.0))
        unit = str(val.get("unit", default_unit)).strip().upper()
        return _convert_unit(value, unit)

    if isinstance(val, (int, float)):
        if default_unit.upper() == "B":
            return float(val) / (1024.0 * 1024.0)
        return float(val)

    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.upper() == "N/A":
            return 0.0
        match = re.match(r"^([0-9.]+)\s*([a-zA-Z]*)$", val_str)
        if match:
            num = float(match.group(1))
            unit = match.group(2).strip().upper() or default_unit.upper()
            return _convert_unit(num, unit)
        try:
            return float(val_str)
        except ValueError:
            return 0.0

    return 0.0


def _convert_unit(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit in ("GB", "GIB"):
        return value * 1024.0
    if unit in ("MB", "MIB", ""):
        return value
    if unit in ("KB", "KIB"):
        return value / 1024.0
    if unit in ("B", "BYTES"):
        return value / (1024.0 * 1024.0)
    return value


def parse_utilization(val: Any) -> float:
    """Parse utilization percentage (0.0 to 100.0) from float, dict, or string."""
    if val is None:
        return 0.0
    if isinstance(val, dict):
        return float(val.get("value", 0.0))
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().rstrip("%").strip()
        try:
            return float(val_str)
        except ValueError:
            return 0.0
    return 0.0


class BaseGpuInventoryProvider(ABC):
    """Abstract base class for GPU management tool inventory providers."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the underlying management tool executable exists."""
        pass

    @abstractmethod
    def get_hardware(self) -> list[dict]:
        """Return list of normalized hardware records."""
        pass

    @abstractmethod
    def get_processes(self) -> list[dict]:
        """Return list of normalized process records."""
        pass


class NvidiaSmiProvider(BaseGpuInventoryProvider):
    """GPU inventory provider using nvidia-smi."""

    @classmethod
    def is_available(cls) -> bool:
        return bool(shutil.which("nvidia-smi"))

    def get_hardware(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            return self.parse_hardware_output(res.stdout)
        except Exception:
            return []

    def parse_hardware_output(self, stdout: str) -> list[dict]:
        gpus = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total": float(parts[3]),
                    "memory_free": float(parts[4]),
                    "memory_used": float(parts[5]),
                    "utilization": float(parts[6]),
                })
        return gpus

    def get_processes(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            return self.parse_processes_output(res.stdout)
        except Exception:
            return []

    def parse_processes_output(self, stdout: str) -> list[dict]:
        procs = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                pid = int(parts[1])
                cmdline = get_proc_cmdline(pid)
                procs.append({
                    "gpu_uuid": parts[0],
                    "pid": pid,
                    "process_name": parts[2],
                    "used_memory": float(parts[3]),
                    "cmdline": cmdline,
                })
        return procs


class AmdSmiProvider(BaseGpuInventoryProvider):
    """Modern AMD GPU inventory provider using amd-smi."""

    @classmethod
    def is_available(cls) -> bool:
        return bool(shutil.which("amd-smi"))

    def get_hardware(self) -> list[dict]:
        if not self.is_available():
            return []
        static_out = ""
        metric_out = ""
        try:
            res = subprocess.run(
                ["amd-smi", "static", "--json"],
                capture_output=True, text=True, check=True
            )
            static_out = res.stdout
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["amd-smi", "metric", "--json"],
                capture_output=True, text=True, check=True
            )
            metric_out = res.stdout
        except Exception:
            pass

        return self.parse_hardware_output(static_out, metric_out)

    def parse_hardware_output(self, static_json_str: str, metric_json_str: str = "") -> list[dict]:
        static_data = []
        metric_data = []
        if static_json_str.strip():
            try:
                parsed = json.loads(static_json_str)
                static_data = parsed if isinstance(parsed, list) else list(parsed.values())
            except Exception:
                static_data = []

        if metric_json_str.strip():
            try:
                parsed = json.loads(metric_json_str)
                metric_data = parsed if isinstance(parsed, list) else list(parsed.values())
            except Exception:
                metric_data = []

        metrics_by_gpu: dict[int, dict] = {}
        for m in metric_data:
            if isinstance(m, dict) and "gpu" in m:
                try:
                    metrics_by_gpu[int(m["gpu"])] = m
                except (ValueError, TypeError):
                    pass

        devices = []
        # If static_data is present, drive by static_data; otherwise use metric_data
        source_list = static_data if static_data else metric_data

        for item in source_list:
            if not isinstance(item, dict):
                continue
            gpu_idx = int(item.get("gpu", len(devices)))
            m_item = metrics_by_gpu.get(gpu_idx, item if item in metric_data else {})

            board = item.get("board", {}) or {}
            asic = item.get("asic", {}) or {}
            name = board.get("product_name") or asic.get("market_name") or item.get("name") or "AMD GPU"

            raw_uuid = item.get("uuid")
            if not raw_uuid or raw_uuid == "N/A":
                raw_uuid = board.get("product_serial") or item.get("bdf") or f"gpu_{gpu_idx}"
            uuid = str(raw_uuid)

            vram = item.get("vram", {}) or {}
            mem_dict = m_item.get("memory", {}) or {}
            total_raw = mem_dict.get("vram_total") or vram.get("vram_size") or 0.0
            used_raw = mem_dict.get("vram_used") or 0.0

            memory_total = parse_memory_to_mib(total_raw, default_unit="MB")
            memory_used = parse_memory_to_mib(used_raw, default_unit="MB")
            memory_free = max(0.0, memory_total - memory_used)

            usage = m_item.get("usage", {}) or {}
            util_raw = usage.get("gfx_activity") or usage.get("gfx_activity_percent") or 0.0
            utilization = parse_utilization(util_raw)

            devices.append({
                "index": gpu_idx,
                "name": name,
                "uuid": uuid,
                "memory_total": memory_total,
                "memory_free": memory_free,
                "memory_used": memory_used,
                "utilization": utilization,
            })

        return devices

    def get_processes(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["amd-smi", "process", "--json"],
                capture_output=True, text=True, check=True
            )
            # Query hardware to map gpu index -> uuid
            hw = self.get_hardware()
            hw_map = {g["index"]: g["uuid"] for g in hw}
            return self.parse_processes_output(res.stdout, hw_map)
        except Exception:
            return []

    def parse_processes_output(self, proc_json_str: str, hw_map: Optional[dict[int, str]] = None) -> list[dict]:
        if not proc_json_str.strip():
            return []
        try:
            parsed = json.loads(proc_json_str)
        except Exception:
            return []

        proc_data = parsed if isinstance(parsed, list) else list(parsed.values())
        if hw_map is None:
            hw_map = {}

        procs = []
        for dev in proc_data:
            if not isinstance(dev, dict):
                continue
            gpu_idx = int(dev.get("gpu", 0))
            gpu_uuid = hw_map.get(gpu_idx, f"gpu_{gpu_idx}")

            p_list = dev.get("process_list", []) or []
            for p in p_list:
                if not isinstance(p, dict):
                    continue
                # Handle possible nested process_info
                p_info = p.get("process_info", p)
                pid = int(p_info.get("pid", 0))
                pname = p_info.get("name") or p_info.get("process_name") or ""
                mem_usage = p_info.get("memory_usage", {}) or {}
                vram_raw = mem_usage.get("vram_mem") or p_info.get("vram_used") or 0.0
                used_memory = parse_memory_to_mib(vram_raw, default_unit="MB")
                cmdline = get_proc_cmdline(pid) or pname

                procs.append({
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "process_name": pname,
                    "used_memory": used_memory,
                    "cmdline": cmdline,
                })

        return procs


class RocmSmiProvider(BaseGpuInventoryProvider):
    """Fallback AMD GPU inventory provider using legacy rocm-smi."""

    @classmethod
    def is_available(cls) -> bool:
        return bool(shutil.which("rocm-smi"))

    def get_hardware(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname", "--showuniqueid", "--showserial",
                 "--showbus", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, check=True
            )
            return self.parse_hardware_output(res.stdout)
        except Exception:
            return []

    def parse_hardware_output(self, json_str: str) -> list[dict]:
        if not json_str.strip():
            return []
        try:
            data = json.loads(json_str)
        except Exception:
            return []

        devices = []
        for key, card in sorted(data.items()):
            if not key.startswith("card") or not isinstance(card, dict):
                continue
            digits = re.sub(r"\D", "", key)
            index = int(digits) if digits else len(devices)

            name = (card.get("Card Series")
                    or card.get("Device Name")
                    or card.get("Card Model")
                    or "AMD Radeon").strip()

            uuid = card.get("Unique ID")
            if not uuid or uuid == "N/A":
                uuid = card.get("Serial Number")
            if not uuid or uuid == "N/A":
                uuid = card.get("PCI Bus") or f"gpu_{index}"
            uuid = str(uuid).strip()

            total_raw = card.get("VRAM Total Memory (B)") or card.get("VRAM Total Memory") or 0
            used_raw = card.get("VRAM Total Used Memory (B)") or card.get("VRAM Total Used Memory") or 0

            memory_total = parse_memory_to_mib(total_raw, default_unit="B")
            memory_used = parse_memory_to_mib(used_raw, default_unit="B")
            memory_free = max(0.0, memory_total - memory_used)

            util_raw = card.get("GPU use (%)") or card.get("GFX Activity") or 0.0
            utilization = parse_utilization(util_raw)

            devices.append({
                "index": index,
                "name": name,
                "uuid": uuid,
                "memory_total": memory_total,
                "memory_free": memory_free,
                "memory_used": memory_used,
                "utilization": utilization,
            })

        return devices

    def get_processes(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["rocm-smi", "--showpids", "details", "--json"],
                capture_output=True, text=True, check=True
            )
            hw = self.get_hardware()
            hw_map = {g["index"]: g["uuid"] for g in hw}
            return self.parse_processes_output(res.stdout, hw_map)
        except Exception:
            return []

    def parse_processes_output(self, json_str: str, hw_map: Optional[dict[int, str]] = None) -> list[dict]:
        if not json_str.strip():
            return []
        try:
            data = json.loads(json_str)
        except Exception:
            return []

        if hw_map is None:
            hw_map = {}

        procs = []

        # Format 1: system dict with PID<pid> entries from print2DArray
        if "system" in data and isinstance(data["system"], dict):
            for k, v in data["system"].items():
                if k.startswith("PID") and isinstance(v, str):
                    pid_str = k[3:]
                    try:
                        pid = int(pid_str)
                    except ValueError:
                        continue
                    parts = [p.strip() for p in v.split(",")]
                    pname = parts[0] if len(parts) > 0 else ""
                    try:
                        gpu_idx = int(parts[1]) if len(parts) > 1 else 0
                    except ValueError:
                        gpu_idx = 0
                    gpu_uuid = hw_map.get(gpu_idx, f"gpu_{gpu_idx}")
                    try:
                        vram_bytes = float(parts[2]) if len(parts) > 2 else 0.0
                    except ValueError:
                        vram_bytes = 0.0
                    used_memory = vram_bytes / (1024.0 * 1024.0)
                    cmdline = get_proc_cmdline(pid) or pname

                    procs.append({
                        "gpu_uuid": gpu_uuid,
                        "pid": pid,
                        "process_name": pname,
                        "used_memory": used_memory,
                        "cmdline": cmdline,
                    })

        # Format 2: "KFD Processes" dict
        if "KFD Processes" in data and isinstance(data["KFD Processes"], dict):
            for pid_str, pinfo in data["KFD Processes"].items():
                if not isinstance(pinfo, dict):
                    continue
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                pname = pinfo.get("Process Name", "")
                gpu_str = str(pinfo.get("GPU(s)", "0")).split(",")[0].strip()
                try:
                    gpu_idx = int(gpu_str)
                except ValueError:
                    gpu_idx = 0
                gpu_uuid = hw_map.get(gpu_idx, f"gpu_{gpu_idx}")
                vram_raw = pinfo.get("VRAM Used", "0 B")
                used_memory = parse_memory_to_mib(vram_raw, default_unit="B")
                cmdline = get_proc_cmdline(pid) or pname

                procs.append({
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "process_name": pname,
                    "used_memory": used_memory,
                    "cmdline": cmdline,
                })

        return procs


def get_gpu_provider() -> Optional[BaseGpuInventoryProvider]:
    """Return the first available GPU inventory provider in priority order.

    Order: NvidiaSmiProvider -> AmdSmiProvider -> RocmSmiProvider.
    If none is available, logs a single WARNING and returns None.
    """
    global _WARNED_MISSING_TOOL
    if NvidiaSmiProvider.is_available():
        return NvidiaSmiProvider()
    if AmdSmiProvider.is_available():
        return AmdSmiProvider()
    if RocmSmiProvider.is_available():
        return RocmSmiProvider()

    if not _WARNED_MISSING_TOOL:
        logger.warning("No GPU management tool (nvidia-smi, amd-smi, rocm-smi) found; GPU monitoring disabled.")
        _WARNED_MISSING_TOOL = True
    return None


def inspect_gpu_hardware() -> list[dict]:
    """Query GPU device inventory via the active provider without importing PyTorch."""
    provider = get_gpu_provider()
    if provider is None:
        return []
    return provider.get_hardware()


def inspect_gpu_processes() -> list[dict]:
    """Query active compute applications per GPU via the active provider."""
    provider = get_gpu_provider()
    if provider is None:
        return []
    return provider.get_processes()
