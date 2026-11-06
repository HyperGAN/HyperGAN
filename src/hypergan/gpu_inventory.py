"""GPU Inventory Provider Abstraction for HyperGAN.

Supports nvidia-smi, amd-smi, and rocm-smi with normalized hardware and process records.
"""
from abc import ABC, abstractmethod
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WARNED_MISSING_TOOL = False
# nvidia-smi/amd-smi/rocm-smi can hang on a wedged driver; the queue and status
# paths call these, so a hung tool reads as "no data" instead of blocking them.
SMI_TIMEOUT_SECONDS = 10


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


def _clean_id(val: Any) -> Optional[str]:
    """Return stripped string if valid non-empty identifier, else None."""
    if val is None or isinstance(val, (dict, list, tuple, set, bool)):
        return None
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNAVAILABLE", "UNDEFINED"):
        return None
    return s


def parse_memory_to_mib(val: Any, default_unit: str = "MB") -> float:
    """Parse various memory representations into float MiB/MB.

    Supports floats/ints, dicts with {'value': ..., 'unit': ...}, and
    unit-suffixed strings like '24.0 GB', '24000.0 MB', '104857600 B'.
    """
    if val is None or isinstance(val, bool):
        return 0.0

    if isinstance(val, dict):
        raw_val = val.get("value")
        if raw_val is None:
            return 0.0
        unit = str(val.get("unit") or default_unit).strip().upper()
        try:
            num = float(raw_val)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            return max(0.0, _convert_unit(num, unit))
        except (ValueError, TypeError, OverflowError):
            if isinstance(raw_val, str):
                return parse_memory_to_mib(raw_val, default_unit=unit)
            return 0.0

    if isinstance(val, (int, float)):
        try:
            num = float(val)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            if default_unit.upper() == "B":
                return max(0.0, num / (1024.0 * 1024.0))
            return max(0.0, num)
        except (ValueError, TypeError, OverflowError):
            return 0.0

    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.upper() in ("N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNAVAILABLE", "UNDEFINED"):
            return 0.0
        match = re.match(r"^([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*([a-zA-Z]*)$", val_str)
        if match:
            try:
                num = float(match.group(1))
                if math.isnan(num) or math.isinf(num):
                    return 0.0
                unit = match.group(2).strip().upper() or default_unit.upper()
                return max(0.0, _convert_unit(num, unit))
            except (ValueError, TypeError, OverflowError):
                return 0.0
        try:
            num = float(val_str)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            return max(0.0, num)
        except (ValueError, TypeError, OverflowError):
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
    if val is None or isinstance(val, bool):
        return 0.0
    if isinstance(val, dict):
        return parse_utilization(val.get("value", 0.0))
    if isinstance(val, (int, float)):
        try:
            num = float(val)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            return min(100.0, max(0.0, num))
        except (ValueError, TypeError, OverflowError):
            return 0.0
    if isinstance(val, str):
        val_str = val.strip().rstrip("%").strip()
        if not val_str or val_str.upper() in ("N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNAVAILABLE", "UNDEFINED"):
            return 0.0
        try:
            num = float(val_str)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            return min(100.0, max(0.0, num))
        except (ValueError, TypeError, OverflowError):
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
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
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
                try:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "uuid": parts[2],
                        "memory_total": float(parts[3]),
                        "memory_free": float(parts[4]),
                        "memory_used": float(parts[5]),
                        "utilization": float(parts[6]),
                    })
                except (ValueError, TypeError):
                    pass
        return gpus

    def get_processes(self) -> list[dict]:
        if not self.is_available():
            return []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
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
                try:
                    pid = int(parts[1])
                    if pid <= 0:
                        continue
                    cmdline = get_proc_cmdline(pid)
                    procs.append({
                        "gpu_uuid": parts[0],
                        "pid": pid,
                        "process_name": parts[2],
                        "used_memory": float(parts[3]),
                        "cmdline": cmdline,
                    })
                except (ValueError, TypeError):
                    pass
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
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
            )
            static_out = res.stdout
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["amd-smi", "metric", "--json"],
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
            )
            metric_out = res.stdout
        except Exception:
            pass

        try:
            return self.parse_hardware_output(static_out, metric_out)
        except Exception:
            return []

    def parse_hardware_output(self, static_json_str: str, metric_json_str: str = "") -> list[dict]:
        static_data = []
        metric_data = []
        if static_json_str.strip():
            try:
                parsed = json.loads(static_json_str)
                if isinstance(parsed, list):
                    static_data = parsed
                elif isinstance(parsed, dict):
                    static_data = list(parsed.values())
                else:
                    static_data = []
            except Exception:
                static_data = []

        if metric_json_str.strip():
            try:
                parsed = json.loads(metric_json_str)
                if isinstance(parsed, list):
                    metric_data = parsed
                elif isinstance(parsed, dict):
                    metric_data = list(parsed.values())
                else:
                    metric_data = []
            except Exception:
                metric_data = []

        metrics_by_gpu: dict[int, dict] = {}
        for m in metric_data:
            if isinstance(m, dict) and "gpu" in m:
                try:
                    metrics_by_gpu[int(float(m["gpu"]))] = m
                except (ValueError, TypeError, OverflowError):
                    pass

        devices = []
        seen_uuids: set[str] = set()
        source_list = static_data if static_data else metric_data

        for item in source_list:
            if not isinstance(item, dict):
                continue
            raw_gpu = item.get("gpu")
            try:
                gpu_idx = int(float(raw_gpu)) if raw_gpu is not None else len(devices)
            except (ValueError, TypeError, OverflowError):
                gpu_idx = len(devices)

            m_item = metrics_by_gpu.get(gpu_idx, item if item in metric_data else {})

            board_raw = item.get("board")
            board = board_raw if isinstance(board_raw, dict) else {}
            asic_raw = item.get("asic")
            asic = asic_raw if isinstance(asic_raw, dict) else {}
            vram_raw = item.get("vram")
            vram = vram_raw if isinstance(vram_raw, dict) else {}

            name = "AMD GPU"
            for cand in (board.get("product_name"), asic.get("market_name"), item.get("name")):
                cand_clean = _clean_id(cand)
                if cand_clean:
                    name = cand_clean
                    break

            raw_uuid = _clean_id(item.get("uuid"))
            if raw_uuid is None:
                raw_uuid = _clean_id(board.get("product_serial"))
            if raw_uuid is None:
                raw_uuid = _clean_id(item.get("bdf"))
            if raw_uuid is None:
                raw_uuid = f"gpu_{gpu_idx}"
            uuid = str(raw_uuid)
            if uuid in seen_uuids:
                bdf_cand = _clean_id(item.get("bdf"))
                if bdf_cand and bdf_cand not in seen_uuids:
                    uuid = bdf_cand
                else:
                    uuid = f"{uuid}_{gpu_idx}"
            seen_uuids.add(uuid)

            mem_dict_raw = m_item.get("memory") if isinstance(m_item, dict) else {}
            mem_dict = mem_dict_raw if isinstance(mem_dict_raw, dict) else {}

            total_raw = mem_dict.get("vram_total") or vram.get("vram_size") or 0.0
            used_raw = mem_dict.get("vram_used") or 0.0

            memory_total = parse_memory_to_mib(total_raw, default_unit="MB")
            memory_used = parse_memory_to_mib(used_raw, default_unit="MB")
            memory_free = max(0.0, memory_total - memory_used)

            usage_raw = m_item.get("usage") if isinstance(m_item, dict) else {}
            usage = usage_raw if isinstance(usage_raw, dict) else {}
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
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
            )
            hw = self.get_hardware()
            hw_map = {g["index"]: g["uuid"] for g in hw if isinstance(g, dict) and "index" in g and "uuid" in g}
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

        if isinstance(parsed, list):
            proc_data = parsed
        elif isinstance(parsed, dict):
            proc_data = list(parsed.values())
        else:
            return []

        if hw_map is None:
            hw_map = {}

        procs = []
        for dev in proc_data:
            if not isinstance(dev, dict):
                continue
            raw_gpu = dev.get("gpu")
            try:
                gpu_idx = int(float(raw_gpu)) if raw_gpu is not None else 0
            except (ValueError, TypeError, OverflowError):
                gpu_idx = 0
            gpu_uuid = hw_map.get(gpu_idx, f"gpu_{gpu_idx}")

            p_list = dev.get("process_list", []) or []
            if not isinstance(p_list, list):
                continue
            for p in p_list:
                if not isinstance(p, dict):
                    continue
                p_info_raw = p.get("process_info")
                p_info = p_info_raw if isinstance(p_info_raw, dict) else p

                raw_pid = p_info.get("pid")
                try:
                    if raw_pid is None:
                        continue
                    pid = int(raw_pid)
                    if pid <= 0:
                        continue
                except (ValueError, TypeError, OverflowError):
                    continue

                pname = str(p_info.get("name") or p_info.get("process_name") or "")
                mem_usage_raw = p_info.get("memory_usage")
                mem_usage = mem_usage_raw if isinstance(mem_usage_raw, dict) else {}
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
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
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

        if not isinstance(data, dict):
            return []

        devices = []
        seen_uuids: set[str] = set()
        for key, card in sorted(data.items()):
            if not key.startswith("card") or not isinstance(card, dict):
                continue
            digits = re.sub(r"\D", "", key)
            index = int(digits) if digits else len(devices)

            raw_name = (card.get("Card Series")
                        or card.get("Device Name")
                        or card.get("Card Model")
                        or "AMD Radeon")
            name = str(raw_name).strip() if raw_name is not None else "AMD Radeon"
            if not name:
                name = "AMD Radeon"

            uuid = _clean_id(card.get("Unique ID"))
            if uuid is None:
                uuid = _clean_id(card.get("Serial Number"))
            if uuid is None:
                uuid = _clean_id(card.get("PCI Bus"))
            if uuid is None:
                uuid = f"gpu_{index}"
            if uuid in seen_uuids:
                pci_cand = _clean_id(card.get("PCI Bus"))
                if pci_cand and pci_cand not in seen_uuids:
                    uuid = pci_cand
                else:
                    uuid = f"{uuid}_{index}"
            seen_uuids.add(uuid)

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
                capture_output=True, text=True, check=True, timeout=SMI_TIMEOUT_SECONDS
            )
            hw = self.get_hardware()
            hw_map = {g["index"]: g["uuid"] for g in hw if isinstance(g, dict) and "index" in g and "uuid" in g}
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

        if not isinstance(data, dict):
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
                        if pid <= 0:
                            continue
                    except (ValueError, TypeError, OverflowError):
                        continue
                    parts = [p.strip() for p in v.split(",")]
                    pname = parts[0] if len(parts) > 0 else ""
                    try:
                        gpu_idx = int(float(parts[1])) if len(parts) > 1 else 0
                    except (ValueError, TypeError, OverflowError):
                        gpu_idx = 0
                    gpu_uuid = hw_map.get(gpu_idx, f"gpu_{gpu_idx}")
                    vram_raw = parts[2] if len(parts) > 2 else "0 B"
                    used_memory = parse_memory_to_mib(vram_raw, default_unit="B")
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
                    if pid <= 0:
                        continue
                except (ValueError, TypeError, OverflowError):
                    continue
                pname = str(pinfo.get("Process Name", ""))
                gpu_str = str(pinfo.get("GPU(s)", "0")).split(",")[0].strip()
                try:
                    gpu_idx = int(float(gpu_str))
                except (ValueError, TypeError, OverflowError):
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
    try:
        provider = get_gpu_provider()
        if provider is None:
            return []
        return provider.get_hardware()
    except Exception:
        return []


def inspect_gpu_processes() -> list[dict]:
    """Query active compute applications per GPU via the active provider."""
    try:
        provider = get_gpu_provider()
        if provider is None:
            return []
        return provider.get_processes()
    except Exception:
        return []
