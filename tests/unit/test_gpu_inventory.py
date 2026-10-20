"""Unit tests for GPU Inventory Provider Abstraction.

Fixtures are explicitly labeled 'from docs, verify on hardware' per specification.
"""
import json
import logging
from unittest.mock import MagicMock, patch
import pytest

from hypergan.gpu_inventory import (
    NvidiaSmiProvider,
    AmdSmiProvider,
    RocmSmiProvider,
    get_gpu_provider,
    inspect_gpu_hardware,
    inspect_gpu_processes,
    reset_inventory_warning_state,
    parse_memory_to_mib,
    parse_utilization,
)
from hypergan.queue import (
    inspect_gpu_hardware as queue_inspect_hardware,
    inspect_gpu_processes as queue_inspect_processes,
    check_device_availability,
)


# ==============================================================================
# FIXTURES: Derived from official AMD documentation ("from docs, verify on hardware")
# ==============================================================================

# Fixture 1: amd-smi static --json ("from docs, verify on hardware")
AMD_SMI_STATIC_FIXTURE = """[
  {
    "gpu": 0,
    "bdf": "0000:23:00.0",
    "uuid": "ffff73bf-0000-1000-80ff-ffffffffffff",
    "asic": {
      "market_name": "AMD Instinct MI300X",
      "vendor_id": "0x1002",
      "device_id": "0x74a1",
      "rev_id": "0x00"
    },
    "board": {
      "model_number": "PCB068560-0020",
      "product_serial": "9XYZ1234ABCD",
      "fru_id": "113-AMDG302010B14",
      "product_name": "Instinct MI300X",
      "manufacturer_name": "AMD"
    },
    "vram": {
      "vram_type": "HBM3",
      "vram_vendor": "SK Hynix",
      "vram_size": 196608,
      "vram_bit_width": 8192
    },
    "driver": {
      "driver_version": "6.8.0"
    }
  },
  {
    "gpu": 1,
    "bdf": "0000:41:00.0",
    "uuid": "ffff73bf-0000-1000-80ff-eeeeeeeeeeee",
    "asic": {
      "market_name": "AMD Instinct MI300X",
      "vendor_id": "0x1002",
      "device_id": "0x74a1",
      "rev_id": "0x00"
    },
    "board": {
      "model_number": "PCB068560-0020",
      "product_serial": "9XYZ5678EFGH",
      "fru_id": "113-AMDG302010B14",
      "product_name": "Instinct MI300X",
      "manufacturer_name": "AMD"
    },
    "vram": {
      "vram_type": "HBM3",
      "vram_vendor": "SK Hynix",
      "vram_size": 196608,
      "vram_bit_width": 8192
    },
    "driver": {
      "driver_version": "6.8.0"
    }
  }
]"""

# Fixture 2: amd-smi metric --json ("from docs, verify on hardware")
AMD_SMI_METRIC_FIXTURE = """[
  {
    "gpu": 0,
    "usage": {
      "gfx_activity": {
        "value": 45.0,
        "unit": "%"
      },
      "umc_activity": {
        "value": 12.0,
        "unit": "%"
      },
      "mm_activity": {
        "value": 3.0,
        "unit": "%"
      }
    },
    "memory": {
      "vram_total": {
        "value": 196608.0,
        "unit": "MB"
      },
      "vram_used": {
        "value": 24576.0,
        "unit": "MB"
      }
    }
  },
  {
    "gpu": 1,
    "usage": {
      "gfx_activity": {
        "value": 0.0,
        "unit": "%"
      },
      "umc_activity": {
        "value": 0.0,
        "unit": "%"
      },
      "mm_activity": {
        "value": 0.0,
        "unit": "%"
      }
    },
    "memory": {
      "vram_total": {
        "value": 196608.0,
        "unit": "MB"
      },
      "vram_used": {
        "value": 512.0,
        "unit": "MB"
      }
    }
  }
]"""

# Fixture 3: amd-smi process --json ("from docs, verify on hardware")
AMD_SMI_PROCESS_FIXTURE = """[
  {
    "gpu": 0,
    "process_list": [
      {
        "name": "python3",
        "pid": 207590,
        "memory_usage": {
          "gtt_mem": "2.0 MB",
          "cpu_mem": "202.0 MB",
          "vram_mem": "24000.0 MB",
          "mem_usage": "24204.0 MB"
        },
        "usage": {
          "gfx": "120000 ns",
          "enc": "0 ns"
        },
        "cu_occupancy": 48
      }
    ]
  },
  {
    "gpu": 1,
    "process_list": []
  }
]"""

# Fixture 4: rocm-smi inventory JSON ("from docs, verify on hardware")
ROCM_SMI_INVENTORY_FIXTURE = """{
  "card0": {
    "Device Name": "Navi 31 [Radeon RX 7900 XTX]",
    "Device ID": "0x744c",
    "Device Rev": "0xc8",
    "Subsystem ID": "0x0c01",
    "GUID": "45678",
    "Card Series": "AMD Radeon RX 7900 XTX",
    "Card Model": "0x744c",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Unique ID": "0x1234567890abcdef",
    "Serial Number": "109-D70401-00_02",
    "PCI Bus": "0000:03:00.0",
    "GPU use (%)": "35",
    "GFX Activity": "35",
    "VRAM Total Memory (B)": "25753026560",
    "VRAM Total Used Memory (B)": "5368709120"
  },
  "card1": {
    "Device Name": "Navi 31 [Radeon RX 7900 XTX]",
    "Device ID": "0x744c",
    "Device Rev": "0xc8",
    "Subsystem ID": "0x0c01",
    "GUID": "45679",
    "Card Series": "AMD Radeon RX 7900 XTX",
    "Card Model": "0x744c",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Unique ID": "N/A",
    "Serial Number": "N/A",
    "PCI Bus": "0000:04:00.0",
    "GPU use (%)": "0",
    "GFX Activity": "0",
    "VRAM Total Memory (B)": "25753026560",
    "VRAM Total Used Memory (B)": "268435456"
  },
  "system": {
    "Driver version": "6.8.0"
  }
}"""

# Fixture 5: rocm-smi process JSON ("from docs, verify on hardware")
ROCM_SMI_PROCESS_FIXTURE = """{
  "system": {
    "Driver version": "6.8.0",
    "PID207590": "python3, 0, 5368709120, 0, 15"
  }
}"""

# Fixture 6: rocm-smi empty process JSON ("from docs, verify on hardware")
ROCM_SMI_PROCESS_EMPTY_FIXTURE = """{
  "system": {
    "Driver version": "6.8.0"
  }
}"""


# ==============================================================================
# Helper Unit Tests: Parsing Logic
# ==============================================================================

def test_parse_memory_to_mib():
    assert parse_memory_to_mib(1024) == 1024.0
    assert parse_memory_to_mib(1024.5) == 1024.5
    assert parse_memory_to_mib("2048 MB") == 2048.0
    assert parse_memory_to_mib("24.0 GB") == 24.0 * 1024.0
    assert parse_memory_to_mib("1048576 B") == 1.0
    assert parse_memory_to_mib("512 KB") == 0.5
    assert parse_memory_to_mib({"value": 196608, "unit": "MB"}) == 196608.0
    assert parse_memory_to_mib({"value": 24, "unit": "GB"}) == 24 * 1024.0
    assert parse_memory_to_mib("N/A") == 0.0
    assert parse_memory_to_mib(None) == 0.0


def test_parse_utilization():
    assert parse_utilization(45.0) == 45.0
    assert parse_utilization("35") == 35.0
    assert parse_utilization("42.5%") == 42.5
    assert parse_utilization({"value": 80.0, "unit": "%"}) == 80.0
    assert parse_utilization(None) == 0.0


# ==============================================================================
# Unit Tests: NvidiaSmiProvider
# ==============================================================================

def test_nvidia_smi_provider_hardware_parsing():
    provider = NvidiaSmiProvider()
    csv_output = (
        "0, NVIDIA RTX A6000, GPU-12345678-abcd-ef01-2345-6789abcdef01, 49152, 45000, 4152, 12\n"
        "1, NVIDIA RTX A6000, GPU-87654321-dcba-10fe-5432-10fedcba9876, 49152, 48000, 1152, 0\n"
    )
    gpus = provider.parse_hardware_output(csv_output)
    assert len(gpus) == 2
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "NVIDIA RTX A6000"
    assert gpus[0]["uuid"] == "GPU-12345678-abcd-ef01-2345-6789abcdef01"
    assert gpus[0]["memory_total"] == 49152.0
    assert gpus[0]["memory_free"] == 45000.0
    assert gpus[0]["memory_used"] == 4152.0
    assert gpus[0]["utilization"] == 12.0

    assert gpus[1]["index"] == 1
    assert gpus[1]["utilization"] == 0.0


def test_nvidia_smi_provider_processes_parsing(monkeypatch):
    provider = NvidiaSmiProvider()
    csv_output = "GPU-12345678-abcd-ef01-2345-6789abcdef01, 12345, python3, 4000\n"
    with patch("hypergan.gpu_inventory.get_proc_cmdline", return_value="python3 -m hypergan train"):
        procs = provider.parse_processes_output(csv_output)
    assert len(procs) == 1
    assert procs[0]["gpu_uuid"] == "GPU-12345678-abcd-ef01-2345-6789abcdef01"
    assert procs[0]["pid"] == 12345
    assert procs[0]["process_name"] == "python3"
    assert procs[0]["used_memory"] == 4000.0
    assert procs[0]["cmdline"] == "python3 -m hypergan train"


# ==============================================================================
# Unit Tests: AmdSmiProvider (with fixtures "from docs, verify on hardware")
# ==============================================================================

def test_amd_smi_provider_hardware_parsing():
    provider = AmdSmiProvider()
    gpus = provider.parse_hardware_output(AMD_SMI_STATIC_FIXTURE, AMD_SMI_METRIC_FIXTURE)
    assert len(gpus) == 2

    # GPU 0
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "Instinct MI300X"
    assert gpus[0]["uuid"] == "ffff73bf-0000-1000-80ff-ffffffffffff"
    assert gpus[0]["memory_total"] == 196608.0
    assert gpus[0]["memory_used"] == 24576.0
    assert gpus[0]["memory_free"] == 196608.0 - 24576.0
    assert gpus[0]["utilization"] == 45.0

    # GPU 1
    assert gpus[1]["index"] == 1
    assert gpus[1]["name"] == "Instinct MI300X"
    assert gpus[1]["uuid"] == "ffff73bf-0000-1000-80ff-eeeeeeeeeeee"
    assert gpus[1]["memory_total"] == 196608.0
    assert gpus[1]["memory_used"] == 512.0
    assert gpus[1]["memory_free"] == 196608.0 - 512.0
    assert gpus[1]["utilization"] == 0.0


def test_amd_smi_provider_processes_parsing():
    provider = AmdSmiProvider()
    hw_map = {0: "ffff73bf-0000-1000-80ff-ffffffffffff", 1: "ffff73bf-0000-1000-80ff-eeeeeeeeeeee"}
    with patch("hypergan.gpu_inventory.get_proc_cmdline", return_value="python3 -m hypergan train exp.toml"):
        procs = provider.parse_processes_output(AMD_SMI_PROCESS_FIXTURE, hw_map)

    assert len(procs) == 1
    assert procs[0]["gpu_uuid"] == "ffff73bf-0000-1000-80ff-ffffffffffff"
    assert procs[0]["pid"] == 207590
    assert procs[0]["process_name"] == "python3"
    assert procs[0]["used_memory"] == 24000.0
    assert procs[0]["cmdline"] == "python3 -m hypergan train exp.toml"


def test_amd_smi_uuid_fallback():
    provider = AmdSmiProvider()
    static_no_uuid = """[
      {
        "gpu": 0,
        "bdf": "0000:03:00.0",
        "uuid": "N/A",
        "asic": {"market_name": "AMD Radeon RX 7900 XTX"},
        "board": {"product_serial": "SERIAL1234"},
        "vram": {"vram_size": 24576}
      }
    ]"""
    gpus = provider.parse_hardware_output(static_no_uuid, "")
    assert len(gpus) == 1
    assert gpus[0]["uuid"] == "SERIAL1234"


# ==============================================================================
# Unit Tests: RocmSmiProvider (with fixtures "from docs, verify on hardware")
# ==============================================================================

def test_rocm_smi_provider_hardware_parsing():
    provider = RocmSmiProvider()
    gpus = provider.parse_hardware_output(ROCM_SMI_INVENTORY_FIXTURE)
    assert len(gpus) == 2

    # Card 0
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "AMD Radeon RX 7900 XTX"
    assert gpus[0]["uuid"] == "0x1234567890abcdef"
    # 25753026560 bytes = 24560.0 MiB
    assert round(gpus[0]["memory_total"], 1) == 24560.0
    # 5368709120 bytes = 5120.0 MiB
    assert round(gpus[0]["memory_used"], 1) == 5120.0
    assert round(gpus[0]["memory_free"], 1) == 19440.0
    assert gpus[0]["utilization"] == 35.0

    # Card 1 (Unique ID is N/A, Serial is N/A -> falls back to PCI Bus)
    assert gpus[1]["index"] == 1
    assert gpus[1]["uuid"] == "0000:04:00.0"
    assert round(gpus[1]["memory_used"], 1) == 256.0
    assert gpus[1]["utilization"] == 0.0


def test_rocm_smi_provider_processes_parsing():
    provider = RocmSmiProvider()
    hw_map = {0: "0x1234567890abcdef", 1: "0000:04:00.0"}
    with patch("hypergan.gpu_inventory.get_proc_cmdline", return_value="python3 -m hypergan queue run"):
        procs = provider.parse_processes_output(ROCM_SMI_PROCESS_FIXTURE, hw_map)

    assert len(procs) == 1
    assert procs[0]["gpu_uuid"] == "0x1234567890abcdef"
    assert procs[0]["pid"] == 207590
    assert procs[0]["process_name"] == "python3"
    assert round(procs[0]["used_memory"], 1) == 5120.0
    assert procs[0]["cmdline"] == "python3 -m hypergan queue run"


def test_rocm_smi_provider_processes_empty():
    provider = RocmSmiProvider()
    procs = provider.parse_processes_output(ROCM_SMI_PROCESS_EMPTY_FIXTURE)
    assert procs == []


def test_rocm_smi_kfd_processes_format():
    provider = RocmSmiProvider()
    kfd_fixture = """{
      "KFD Processes": {
        "1075718": {
          "Process Name": "hypergan_worker",
          "GPU(s)": "0",
          "VRAM Used": "104857600 B"
        }
      }
    }"""
    hw_map = {0: "gpu_0"}
    procs = provider.parse_processes_output(kfd_fixture, hw_map)
    assert len(procs) == 1
    assert procs[0]["pid"] == 1075718
    assert procs[0]["process_name"] == "hypergan_worker"
    assert round(procs[0]["used_memory"], 1) == 100.0


# ==============================================================================
# Unit Tests: Provider Selection & Single Warning on Missing Tools
# ==============================================================================

def test_missing_tools_logs_single_warning(caplog):
    reset_inventory_warning_state()
    with patch.object(NvidiaSmiProvider, "is_available", return_value=False), \
         patch.object(AmdSmiProvider, "is_available", return_value=False), \
         patch.object(RocmSmiProvider, "is_available", return_value=False):

        with caplog.at_level(logging.WARNING):
            provider1 = get_gpu_provider()
            hw1 = inspect_gpu_hardware()
            procs1 = inspect_gpu_processes()

        assert provider1 is None
        assert hw1 == []
        assert procs1 == []

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "No GPU management tool" in r.message]
        assert len(warning_records) == 1
        assert "No GPU management tool (nvidia-smi, amd-smi, rocm-smi) found; GPU monitoring disabled." in warning_records[0].message

        # Subsequent call does NOT log warning again
        caplog.clear()
        provider2 = get_gpu_provider()
        assert provider2 is None
        warning_records_2 = [r for r in caplog.records if r.levelno == logging.WARNING and "No GPU management tool" in r.message]
        assert len(warning_records_2) == 0


def test_provider_selection_precedence():
    # Nvidia present -> Nvidia selected
    with patch.object(NvidiaSmiProvider, "is_available", return_value=True), \
         patch.object(AmdSmiProvider, "is_available", return_value=True), \
         patch.object(RocmSmiProvider, "is_available", return_value=True):
        p = get_gpu_provider()
        assert isinstance(p, NvidiaSmiProvider)

    # Nvidia absent, Amd present -> Amd selected
    with patch.object(NvidiaSmiProvider, "is_available", return_value=False), \
         patch.object(AmdSmiProvider, "is_available", return_value=True), \
         patch.object(RocmSmiProvider, "is_available", return_value=True):
        p = get_gpu_provider()
        assert isinstance(p, AmdSmiProvider)

    # Nvidia & Amd absent, Rocm present -> Rocm selected
    with patch.object(NvidiaSmiProvider, "is_available", return_value=False), \
         patch.object(AmdSmiProvider, "is_available", return_value=False), \
         patch.object(RocmSmiProvider, "is_available", return_value=True):
        p = get_gpu_provider()
        assert isinstance(p, RocmSmiProvider)


# ==============================================================================
# Unit Tests: Queue & Inspection Integration
# ==============================================================================

def test_queue_delegates_to_provider():
    mock_hw = [{"index": 0, "name": "AMD MI300X", "uuid": "uuid-1", "memory_total": 196608.0, "memory_free": 190000.0, "memory_used": 6608.0, "utilization": 10.0}]
    mock_procs = [{"gpu_uuid": "uuid-1", "pid": 9999, "process_name": "python", "used_memory": 500.0, "cmdline": "python train"}]

    with patch("hypergan.gpu_inventory.inspect_gpu_hardware", return_value=mock_hw), \
         patch("hypergan.gpu_inventory.inspect_gpu_processes", return_value=mock_procs):
        assert queue_inspect_hardware() == mock_hw
        assert queue_inspect_processes() == mock_procs


def test_queue_check_device_availability_with_amd_records():
    mock_hw = [
        {"index": 0, "name": "AMD MI300X", "uuid": "amd-uuid-0", "memory_total": 196608.0, "memory_free": 190000.0, "memory_used": 6608.0, "utilization": 10.0},
        {"index": 1, "name": "AMD MI300X", "uuid": "amd-uuid-1", "memory_total": 196608.0, "memory_free": 500.0, "memory_used": 196108.0, "utilization": 99.0},
    ]
    mock_procs = [
        {"gpu_uuid": "amd-uuid-1", "pid": 1111, "process_name": "hypergan", "used_memory": 190000.0, "cmdline": "hypergan train cifar.toml"},
    ]

    with patch("hypergan.queue.inspect_gpu_hardware", return_value=mock_hw), \
         patch("hypergan.queue.inspect_gpu_processes", return_value=mock_procs):

        # Device 0 is free
        avail, reason = check_device_availability("0")
        assert avail is True
        assert reason is None

        # Device 1 has insufficient free memory and is busy with hypergan
        avail, reason = check_device_availability("1")
        assert avail is False
        assert "free memory" in reason or "busy" in reason


# ==============================================================================
# Robustness & Adversarial Edge Cases (Reviewer M2-2 & Challenger M2-1 Verification)
# ==============================================================================

def test_parse_memory_adversarial_and_none():
    assert parse_memory_to_mib(None) == 0.0
    assert parse_memory_to_mib({}) == 0.0
    assert parse_memory_to_mib({"value": None}) == 0.0
    assert parse_memory_to_mib({"value": "N/A"}) == 0.0
    assert parse_memory_to_mib({"value": "invalid"}) == 0.0
    assert parse_memory_to_mib("1.2.3 MB") == 0.0
    assert parse_memory_to_mib("0 B") == 0.0
    assert parse_memory_to_mib("invalid") == 0.0
    assert parse_memory_to_mib("") == 0.0


def test_parse_utilization_adversarial_and_none():
    assert parse_utilization(None) == 0.0
    assert parse_utilization({}) == 0.0
    assert parse_utilization({"value": None}) == 0.0
    assert parse_utilization({"value": "N/A"}) == 0.0
    assert parse_utilization({"value": "45.0%"}) == 45.0
    assert parse_utilization("invalid") == 0.0


def test_amd_smi_uuid_fallback_multi_tier():
    provider = AmdSmiProvider()
    # Tier 1: uuid present
    s1 = """[{"gpu": 0, "uuid": "real-uuid", "board": {"product_serial": "serial"}, "bdf": "0000:01:00.0"}]"""
    assert provider.parse_hardware_output(s1)[0]["uuid"] == "real-uuid"

    # Tier 2: uuid is N/A, product_serial present
    s2 = """[{"gpu": 0, "uuid": "N/A", "board": {"product_serial": "serial-123"}, "bdf": "0000:01:00.0"}]"""
    assert provider.parse_hardware_output(s2)[0]["uuid"] == "serial-123"

    # Tier 3: uuid is N/A, product_serial is N/A, bdf present
    s3 = """[{"gpu": 0, "uuid": "N/A", "board": {"product_serial": "N/A"}, "bdf": "0000:03:00.0"}]"""
    assert provider.parse_hardware_output(s3)[0]["uuid"] == "0000:03:00.0"

    # Tier 4: uuid, serial, and bdf are all N/A
    s4 = """[{"gpu": 0, "uuid": "N/A", "board": {"product_serial": "N/A"}, "bdf": "N/A"}]"""
    assert provider.parse_hardware_output(s4)[0]["uuid"] == "gpu_0"


def test_amd_smi_hardware_malformed_entries():
    provider = AmdSmiProvider()
    # Null or non-integer gpu index
    res_null = provider.parse_hardware_output('[{"gpu": null}]')
    assert len(res_null) == 1
    assert res_null[0]["index"] == 0

    res_str = provider.parse_hardware_output('[{"gpu": "not_an_int"}]')
    assert len(res_str) == 1
    assert res_str[0]["index"] == 0

    # Non-dict board or asic
    res_board = provider.parse_hardware_output('[{"gpu": 0, "board": "N/A", "asic": null}]')
    assert len(res_board) == 1
    assert res_board[0]["name"] == "AMD GPU"

    # Metrics with null or N/A
    metric_json = """[{"gpu": 0, "memory": {"vram_total": {"value": null, "unit": "MB"}, "vram_used": {"value": "N/A", "unit": "MB"}}, "usage": {"gfx_activity": {"value": null}}}]"""
    res_m = provider.parse_hardware_output("", metric_json)
    assert len(res_m) == 1
    assert res_m[0]["memory_total"] == 0.0
    assert res_m[0]["memory_used"] == 0.0
    assert res_m[0]["utilization"] == 0.0


def test_amd_smi_processes_malformed_entries():
    provider = AmdSmiProvider()
    # Non-dict JSON input
    assert provider.parse_processes_output("null") == []
    assert provider.parse_processes_output("123") == []

    # Non-integer / null PIDs and invalid gpu index
    proc_json = """[{"gpu": "invalid", "process_list": [{"name": "p_na", "pid": "N/A"}, {"name": "p_none", "pid": null}, {"name": "p_ok", "pid": 12345}]}]"""
    procs = provider.parse_processes_output(proc_json)
    assert len(procs) == 1
    assert procs[0]["pid"] == 12345
    assert procs[0]["gpu_uuid"] == "gpu_0"


def test_rocm_smi_malformed_json_and_primitives():
    provider = RocmSmiProvider()
    assert provider.parse_hardware_output("[]") == []
    assert provider.parse_hardware_output("123") == []
    assert provider.parse_hardware_output("null") == []

    assert provider.parse_processes_output("123") == []
    assert provider.parse_processes_output("null") == []
    assert provider.parse_processes_output("true") == []

    # Non-integer PIDs
    sys_json = '{"system": {"PIDinvalid": "python3, 0, 1000", "PID1234": "python3, 0, 1000"}}'
    res_sys = provider.parse_processes_output(sys_json)
    assert len(res_sys) == 1
    assert res_sys[0]["pid"] == 1234

    kfd_json = '{"KFD Processes": {"not_a_pid": {"Process Name": "p", "GPU(s)": "0"}, "5678": {"Process Name": "p", "GPU(s)": "0"}}}'
    res_kfd = provider.parse_processes_output(kfd_json)
    assert len(res_kfd) == 1
    assert res_kfd[0]["pid"] == 5678


def test_amd_smi_get_hardware_exception_containment():
    provider = AmdSmiProvider()
    from unittest.mock import MagicMock, patch
    with patch("subprocess.run") as mock_run:
        mock_res = MagicMock()
        mock_res.stdout = '[{"gpu": null}]'
        mock_run.return_value = mock_res
        with patch.object(AmdSmiProvider, "is_available", return_value=True):
            res = provider.get_hardware()
            assert isinstance(res, list)
            assert len(res) == 1


def test_amd_smi_uuid_fallback_prevents_na_collision():
    """Verify AmdSmiProvider avoids 'N/A' UUID collision when product_serial is 'N/A'."""
    provider = AmdSmiProvider()
    static_fixture = """[
      {
        "gpu": 0,
        "bdf": "0000:03:00.0",
        "uuid": "N/A",
        "board": {"product_serial": "N/A"}
      },
      {
        "gpu": 1,
        "bdf": "0000:04:00.0",
        "uuid": "None",
        "board": {"product_serial": "N/A"}
      },
      {
        "gpu": 2,
        "bdf": null,
        "uuid": "",
        "board": {"product_serial": ""}
      }
    ]"""
    gpus = provider.parse_hardware_output(static_fixture, "")
    assert len(gpus) == 3
    assert gpus[0]["uuid"] == "0000:03:00.0"
    assert gpus[1]["uuid"] == "0000:04:00.0"
    assert gpus[2]["uuid"] == "gpu_2"
    uuids = [g["uuid"] for g in gpus]
    assert len(set(uuids)) == 3
    assert "N/A" not in uuids


def test_rocm_smi_uuid_fallback_prevents_na_collision():
    """Verify RocmSmiProvider avoids collision when identifiers are sentinels."""
    provider = RocmSmiProvider()
    rocm_json = """{
      "card0": {"Unique ID": "N/A", "Serial Number": "N/A", "PCI Bus": "N/A"},
      "card1": {"Unique ID": "N/A", "Serial Number": "N/A", "PCI Bus": "N/A"}
    }"""
    devs = provider.parse_hardware_output(rocm_json)
    assert len(devs) == 2
    assert devs[0]["uuid"] == "gpu_0"
    assert devs[1]["uuid"] == "gpu_1"
    assert devs[0]["uuid"] != devs[1]["uuid"]


def test_amd_smi_processes_special_characters():
    """Verify AmdSmiProvider handles unicode characters, null bytes, and emojis."""
    provider = AmdSmiProvider()
    proc_json = """[
      {
        "gpu": 0,
        "process_list": [
          {"name": "worker\\u0000hidden 🔥", "pid": 54321}
        ]
      }
    ]"""
    res = provider.parse_processes_output(proc_json)
    assert len(res) == 1
    assert res[0]["pid"] == 54321
    assert "🔥" in res[0]["process_name"]


def test_parse_utilization_clamping_and_extremes():
    """Verify utilization parsing strictly clamps to [0.0, 100.0] and handles extremes."""
    assert parse_utilization(150.0) == 100.0
    assert parse_utilization(-20.0) == 0.0
    assert parse_utilization("125%") == 100.0
    assert parse_utilization("-5%") == 0.0
    assert parse_utilization({"value": 200.0}) == 100.0
    assert parse_utilization({"value": -10.0}) == 0.0
    assert parse_utilization(float("nan")) == 0.0
    assert parse_utilization(float("inf")) == 0.0
    assert parse_utilization(True) == 0.0
    assert parse_utilization(False) == 0.0


def test_inspect_gpu_hardware_and_processes_exception_shielding():
    """Verify top-level inspect_gpu_hardware and inspect_gpu_processes never raise exceptions."""
    with patch("hypergan.gpu_inventory.get_gpu_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.get_hardware.side_effect = RuntimeError("Fatal hardware query crash")
        mock_provider.get_processes.side_effect = RuntimeError("Fatal process query crash")
        mock_get_provider.return_value = mock_provider

        assert inspect_gpu_hardware() == []
        assert inspect_gpu_processes() == []


def test_amd_and_rocm_smi_overflow_error_handling():
    """Verify AmdSmiProvider and RocmSmiProvider handle infinite/overflowing numeric values gracefully."""
    amd = AmdSmiProvider()
    rocm = RocmSmiProvider()

    # Hardware: metric gpu index is 1e1000
    hw_res_metric = amd.parse_hardware_output("", '[{"gpu": 1e1000}]')
    assert isinstance(hw_res_metric, list)

    # Hardware: static gpu index is 1e1000
    hw_res_static = amd.parse_hardware_output('[{"gpu": 1e1000}]', "")
    assert isinstance(hw_res_static, list)

    # Processes: process gpu index is 1e1000
    proc_res_gpu = amd.parse_processes_output('[{"gpu": 1e1000, "process_list": []}]')
    assert isinstance(proc_res_gpu, list)

    # Processes: pid is 1e1000
    proc_res_pid = amd.parse_processes_output('[{"gpu": 0, "process_list": [{"pid": 1e1000}]}]')
    assert isinstance(proc_res_pid, list)

    # RocmSmiProvider: system process format with 1e1000 gpu index
    rocm_sys = rocm.parse_processes_output('{"system": {"PID123": "python, 1e1000, 100"}}')
    assert isinstance(rocm_sys, list)

    # RocmSmiProvider: KFD process format with 1e1000 gpu index
    rocm_kfd = rocm.parse_processes_output('{"KFD Processes": {"123": {"GPU(s)": "1e1000"}}}')
    assert isinstance(rocm_kfd, list)


def test_rocm_smi_non_string_card_series():
    """Verify RocmSmiProvider safely handles non-string Card Series / Model fields without AttributeError."""
    rocm = RocmSmiProvider()

    # Card Series as integer
    devs_int = rocm.parse_hardware_output('{"card0": {"Card Series": 7900}}')
    assert len(devs_int) == 1
    assert devs_int[0]["name"] in ("7900", "AMD Radeon")

    # Card Series as dict
    devs_dict = rocm.parse_hardware_output('{"card0": {"Card Series": {"series": "7900"}}}')
    assert len(devs_dict) == 1
    assert devs_dict[0]["index"] == 0

    # Card Series as list
    devs_list = rocm.parse_hardware_output('{"card0": {"Card Series": ["Radeon", "7900"]}}')
    assert len(devs_list) == 1

    # Empty / None / whitespace series
    devs_empty = rocm.parse_hardware_output('{"card0": {"Card Series": "   "}}')
    assert len(devs_empty) == 1
    assert devs_empty[0]["name"] == "AMD Radeon"


def test_parse_memory_to_mib_dict_nan_inf_negative_sanitization():
    """Verify parse_memory_to_mib sanitizes NaN, Inf, and negative values in dict representations."""
    assert parse_memory_to_mib({"value": float("nan")}) == 0.0
    assert parse_memory_to_mib({"value": "nan"}) == 0.0
    assert parse_memory_to_mib({"value": float("inf")}) == 0.0
    assert parse_memory_to_mib({"value": "inf"}) == 0.0
    assert parse_memory_to_mib({"value": -100.0}) == 0.0
    assert parse_memory_to_mib({"value": -50, "unit": "GB"}) == 0.0
    assert parse_memory_to_mib({"value": 16.0, "unit": "GB"}) == 16384.0


def test_inspect_gpu_hardware_and_processes_get_provider_exception_containment():
    """Verify inspect_gpu_hardware and inspect_gpu_processes catch exceptions from get_gpu_provider."""
    with patch("hypergan.gpu_inventory.get_gpu_provider", side_effect=OSError("Permission denied")):
        assert inspect_gpu_hardware() == []
        assert inspect_gpu_processes() == []

    with patch("hypergan.gpu_inventory.get_gpu_provider", side_effect=RuntimeError("Unexpected driver crash")):
        assert inspect_gpu_hardware() == []
        assert inspect_gpu_processes() == []


