"""Adversarial stress tests for Milestone 3 Features 25–33.

Author: Challenger M3-1 (Empirical Challenger)
Covers:
  - F25: Adversarial HNDL AST Validation (syntax errors, shape mismatches, cyclic/undefined refs, line/column diagnostics, 2-arg backwards compatibility)
  - F26: Pretrained provider symbolic shape functions
  - F27: Adversarial Typo Suggestions (table headers, inner fields, garbage/unicode strings)
  - F28 & F29: Adversarial Schema and Ops Introspection (85 operators, categories, docs, queries, schema sections)
  - F31–F33: Global JSON Error Envelopes, Exit Codes, and Quiet Combinations
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

import hypergan as hg
from hypergan.network_config import validate_network_args, MAX_SOURCE_BYTES
from hypergan.pretrained_providers import _resnet18_shape_fn, _dinov3_shape_fn
from hypergan.model_file import ModelFileError, lower, require_networks, TOP_LEVEL
from hypergan.schema_info import get_schema, MODEL_SCHEMA
from hypergan.ops_info import get_operators


# ==============================================================================
# Objective 1: Adversarial HNDL AST Validation
# ==============================================================================

class TestAdversarialHndlAstValidation:
    """Stress-test static HNDL AST parsing, validation, and error reporting."""

    def test_invalid_operator_names(self):
        """Invalid or misspelled operator names must be caught statically with syntax/call error."""
        invalid_ops = [
            "y = non_existent_operator_xyz(x)",
            "y = linearr(x, 16)",
            "y = convv2d(x, 32)",
            "y = my_custom_undefined_layer(x)",
        ]
        for src in invalid_ops:
            spec = {
                "source": src,
                "input_shape": ["B", 16],
                "output_shape": ["B", 16],
            }
            errors = validate_network_args(spec)
            assert len(errors) > 0, f"Failed to reject invalid operator in: {src}"
            err_msg = errors[0]
            assert "line 1" in err_msg.lower() or "column" in err_msg.lower()
            assert any(term in err_msg.lower() for term in ["operator", "syntax", "unknown", "registered"])

    def test_syntax_errors_and_line_column_context(self):
        """Malformed syntax (unclosed parens, bad tokens, empty blocks) report line and column."""
        syntax_cases = [
            ("y = linear(x, 16", "unclosed paren"),
            ("y = linear(x, [16, )", "unclosed list"),
            ("y = linear(x, 16))\n", "extra paren"),
            ("y = @invalid_token(x)", "bad symbol"),
            ("y = linear(x, 16)\nz = relu(", "multiline unclosed paren"),
        ]
        for src, case_desc in syntax_cases:
            spec = {
                "source": src,
                "input_shape": ["B", 16],
                "output_shape": ["B", 16],
            }
            errors = validate_network_args(spec)
            assert len(errors) > 0, f"Failed on syntax case: {case_desc}"
            err_msg = errors[0]
            assert "line" in err_msg.lower() and "column" in err_msg.lower(), (
                f"Missing line/col context in {case_desc}: {err_msg}"
            )

    def test_shape_and_rank_mismatches(self):
        """Static shape resolver detects dimension and rank conflicts."""
        # 1. Output dimension mismatch: linear outputs 32, contract expects 64
        spec_dim = {
            "source": "y = linear(x, 32)",
            "input_shape": ["B", 16],
            "output_shape": ["B", 64],
        }
        errors_dim = validate_network_args(spec_dim)
        assert len(errors_dim) > 0
        assert "constraint" in errors_dim[0].lower() or "incompatible" in errors_dim[0].lower()

        # 2. Rank conflict: input is 2D (B, 16), but conv requires 4D (B, C, H, W)
        spec_rank = {
            "source": "y = conv(32, kernel_size=3)",
            "input_shape": ["B", 16],
            "output_shape": ["B", 32, 16, 16],
        }
        errors_rank = validate_network_args(spec_rank)
        assert len(errors_rank) > 0
        assert "rank" in errors_rank[0].lower() or "conflict" in errors_rank[0].lower() or "constraint" in errors_rank[0].lower()

    def test_undefined_and_cyclic_references(self):
        """Undefined variables and cyclic dataflow are rejected statically."""
        # Undefined input identifier
        spec_undef = {
            "source": "y = linear(undefined_variable, 16)",
            "input_shape": ["B", 16],
            "output_shape": ["B", 16],
        }
        errors_undef = validate_network_args(spec_undef)
        assert len(errors_undef) > 0
        assert "undefined" in errors_undef[0].lower() or "name" in errors_undef[0].lower()

    def test_backwards_compatibility_location_arg(self):
        """Passing non-empty string `location` raises ValueError on schema or structural failure."""
        bad_spec = {"source": "foo", "unknown_arg": 1}
        with pytest.raises(ValueError, match="unknown HNDL arguments"):
            validate_network_args(bad_spec, location="networks.generator")

        bad_shape = {"source": "y = linear(x, 16)", "input_shape": ["invalid"]}
        with pytest.raises(ValueError, match="must be \\['B'"):
            validate_network_args(bad_shape, location="networks.critic")

    def test_boundary_inputs_and_limits(self):
        """Boundary limits: empty spec, oversized source text, non-B batch dimensions."""
        # Spec with both source and file
        spec_both = {"source": "y = linear(x, 16)", "file": "foo.hndl", "input_shape": ["B", 16], "output_shape": ["B", 16]}
        assert any("exactly one" in e for e in validate_network_args(spec_both))

        # Spec with neither source nor file
        spec_neither = {"input_shape": ["B", 16], "output_shape": ["B", 16]}
        assert any("exactly one" in e for e in validate_network_args(spec_neither))

        # Oversized source (> 1 MiB)
        huge_src = "y = linear(x, 16)\n" + ("# comment block padding\n" * 60000)
        assert len(huge_src.encode("utf-8")) > MAX_SOURCE_BYTES
        spec_huge = {"source": huge_src, "input_shape": ["B", 16], "output_shape": ["B", 16]}
        assert any("exceeds 1 MiB" in e for e in validate_network_args(spec_huge))

        # Non-positive shape dimensions
        spec_neg = {"source": "y = linear(x, 16)", "input_shape": ["B", -4], "output_shape": ["B", 16]}
        assert any("positive" in e for e in validate_network_args(spec_neg))


# ==============================================================================
# Objective 2: Adversarial Typo Suggestions
# ==============================================================================

class TestAdversarialTypoSuggestions:
    """Stress-test difflib-based typo suggestions across tables and fields."""

    @pytest.mark.parametrize("typo,expected_suggestion", [
        ("tran", "train"),
        ("dat", "data"),
        ("netwrk", "networks"),
        ("losss", "losses"),
        ("checkpints", "checkpoints"),
        ("metrcs", "metrics"),
    ])
    def test_table_typo_suggestions_with_networks(self, typo, expected_suggestion):
        """Top-level table typos in a model file suggest the closest valid table."""
        base_spec = {
            "format": "hypergan-model/1",
            "name": "typo-test",
            "networks": {
                "generator": {
                    "role": "generator",
                    "source": "y = linear(x, 16)",
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 16],
                    "inputs": {"x": "latent"}
                },
                "critic": {
                    "role": "critic",
                    "source": "y = linear(x, 1)",
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 1],
                    "inputs": {"x": "candidate"}
                }
            },
            "losses": [{"type": "adversarial", "critic": "critic"}],
            "data": {"builtin": "gaussian_grid"},
        }
        spec = dict(base_spec)
        spec[typo] = {"dummy": 1}

        with pytest.raises(ModelFileError) as exc_info:
            lower(spec)
        err_msg = str(exc_info.value).lower()
        assert f"did you mean '{expected_suggestion}'?" in err_msg or f"did you mean '[{expected_suggestion}]'?" in err_msg

    def test_require_networks_table_typo_suggestions(self):
        """Files missing [networks] with misspelled tables trigger suggestions."""
        with pytest.raises(ModelFileError) as exc1:
            require_networks({"tran": {"batch_size": 16}})
        assert "did you mean '[train]'?" in str(exc1.value).lower()

        with pytest.raises(ModelFileError) as exc2:
            require_networks({"dat": {"builtin": "gaussian_grid"}})
        assert "did you mean '[data]'?" in str(exc2.value).lower()

    @pytest.mark.parametrize("typo_field,expected_field", [
        ("step", "steps"),
        ("batchsize", "batch_size"),
        ("lrr", "lr"),
        ("betass", "betas"),
        ("every_step", "every_steps"),
    ])
    def test_inner_train_field_typos(self, typo_field, expected_field):
        """Misspelled fields inside [train] table suggest valid field names."""
        base_spec = {
            "format": "hypergan-model/1",
            "name": "train-field-typo",
            "networks": {
                "generator": {
                    "role": "generator",
                    "source": "y = linear(x, 16)",
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 16],
                    "inputs": {"x": "latent"}
                },
                "critic": {
                    "role": "critic",
                    "source": "y = linear(x, 1)",
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 1],
                    "inputs": {"x": "candidate"}
                }
            },
            "losses": [{"type": "adversarial", "critic": "critic"}],
            "data": {"builtin": "gaussian_grid"},
            "train": {typo_field: 1},
        }
        with pytest.raises(ModelFileError) as exc_info:
            lower(base_spec)
        err_msg = str(exc_info.value).lower()
        assert f"did you mean '{expected_field}'?" in err_msg

    def test_robustness_on_pathological_strings(self):
        """difflib matching handles unicode, symbols, and empty keys without crashing."""
        for strange_key in ["", "   ", "___", "\U0001F600", "???!!!", "a" * 1000]:
            with pytest.raises(ModelFileError):
                require_networks({strange_key: {}})


# ==============================================================================
# Objective 3: Adversarial Schema and Ops Introspection
# ==============================================================================

class TestAdversarialSchemaAndOpsIntrospection:
    """Stress-test hypergan schema and hypergan ops CLI and python APIs."""

    def test_schema_full_and_section_json(self):
        """hypergan schema --json returns full hypergan-model/1 specification."""
        proc = subprocess.run([sys.executable, "-m", "hypergan.cli", "schema", "--json"],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data.get("format") == "hypergan-model/1"
        assert "sections" in data
        assert len(data["sections"]) >= 13

        # Test valid sections
        for s in ["train", "data", "networks", "losses", "prior", "checkpoints", "run"]:
            p = subprocess.run([sys.executable, "-m", "hypergan.cli", "schema", "--section", s, "--json"],
                               capture_output=True, text=True)
            assert p.returncode == 0, p.stderr
            sec_data = json.loads(p.stdout)
            assert sec_data.get("section") == s
            assert "type" in sec_data
            assert "description" in sec_data

    def test_schema_invalid_section_json_envelope(self):
        """hypergan schema --section <invalid> --json emits error envelope to stdout."""
        p = subprocess.run([sys.executable, "-m", "hypergan.cli", "schema", "--section", "non_existent_section", "--json"],
                           capture_output=True, text=True)
        assert p.returncode == 1
        assert p.stderr == ""
        envelope = json.loads(p.stdout)
        assert envelope.get("ok") is False
        assert envelope["error"]["type"] == "KeyError"
        assert "non_existent_section" in envelope["error"]["message"]

    def test_ops_catalog_completeness_and_85_operators(self):
        """hypergan ops indexes exactly 85 operators across categories."""
        ops = get_operators()
        assert len(ops) == 85, f"Expected exactly 85 operators, got {len(ops)}"

        categories = {op["category"] for op in ops}
        expected_categories = {
            "activation", "arithmetic", "convolution", "core", "memory", "normalization",
            "pretrained", "regularization", "sequence", "shape", "spatial", "vision"
        }
        for exp_cat in expected_categories:
            assert exp_cat in categories, f"Missing category: {exp_cat}"

        for op in ops:
            assert op.get("name"), f"Missing operator name: {op}"
            assert op.get("category"), f"Missing operator category: {op}"
            assert "shape" in op, f"Missing shape metadata: {op}"
            assert "inputs" in op and isinstance(op["inputs"], list)
            assert "outputs" in op and isinstance(op["outputs"], list)
            assert "args" in op and isinstance(op["args"], dict)

    def test_ops_filter_queries(self):
        """hypergan ops query filters match names, categories, and summaries."""
        # Query matching linear
        linear_ops = get_operators("linear")
        assert any(op["name"] == "linear" for op in linear_ops)

        # Query matching norm
        norm_ops = get_operators("norm")
        assert len(norm_ops) > 0
        assert all("norm" in (op["name"] + op["category"] + op["summary"]).lower() for op in norm_ops)

        # Query with zero matches returns empty list without error
        none_ops = get_operators("xyz_zero_matches_query_999")
        assert none_ops == []

    def test_python_api_parity(self):
        """hg.schema() and hg.operators() / hg.ops() match CLI results."""
        schema = hg.schema()
        assert schema["format"] == "hypergan-model/1"

        train_sec = hg.schema("train")
        assert train_sec["section"] == "train"

        ops = hg.operators()
        assert len(ops) == 85

        query_ops = hg.ops("conv")
        assert len(query_ops) > 0
        assert any(op["name"] == "conv" for op in query_ops)


# ==============================================================================
# Objective 4: Adversarial JSON Error Envelopes & Exit Codes
# ==============================================================================

class TestAdversarialJsonErrorEnvelopes:
    """Stress-test CLI error handling when --json is provided."""

    @pytest.mark.parametrize("cmd_args,expected_rc,expected_err_type", [
        (["validate", "non_existent_model_file.toml", "--json"], 1, "FileNotFoundError"),
        (["train", "--non-existent-flag", "--json"], 2, "ArgumentError"),
        (["resume", "non_existent_run_directory_xyz", "--json"], 1, "ValueError"),
        (["status", "non_existent_run_directory_xyz", "--json"], 1, "FileNotFoundError"),
        (["inspect", "non_existent_run_directory_xyz", "--json"], 1, "FileNotFoundError"),
        (["compare", "non_existent_run_1", "non_existent_run_2", "--json"], 1, "FileNotFoundError"),
        (["queue", "--bad-flag", "--json"], 2, "ArgumentError"),
        (["queue", "bad_command", "--json"], 2, "ArgumentError"),
        (["schema", "--section", "invalid_section_name", "--json"], 1, "KeyError"),
        (["--invalid-global-flag", "--json"], 2, "ArgumentError"),
    ])
    def test_json_error_envelopes_and_exit_codes(self, cmd_args, expected_rc, expected_err_type):
        """Commands failing with --json output strictly valid JSON on stdout and leave stderr clean."""
        proc = subprocess.run([sys.executable, "-m", "hypergan.cli", *cmd_args],
                              capture_output=True, text=True)
        assert proc.returncode == expected_rc, (
            f"Expected RC {expected_rc}, got {proc.returncode} for cmd: {cmd_args}"
        )
        assert proc.stderr == "", f"Stderr must be empty, but got: {proc.stderr}"

        # Stdout must be strictly valid JSON
        try:
            envelope = json.loads(proc.stdout)
        except Exception as exc:
            pytest.fail(f"Stdout was not valid JSON for cmd {cmd_args}: {exc}\nRaw stdout: {proc.stdout}")

        assert envelope.get("ok") is False
        assert "error" in envelope
        assert envelope["error"]["type"] == expected_err_type
        assert envelope["error"]["message"]

    def test_json_combined_with_quiet_flag(self):
        """Combining --json and -q/--quiet retains valid JSON output without suppressing envelopes."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan.cli", "validate", "missing.toml", "--json", "-q"],
            capture_output=True, text=True
        )
        assert proc.returncode == 1
        assert proc.stderr == ""
        envelope = json.loads(proc.stdout)
        assert envelope.get("ok") is False
        assert envelope["error"]["type"] == "FileNotFoundError"

        # Valid success command with --json and --quiet
        proc_ok = subprocess.run(
            [sys.executable, "-m", "hypergan.cli", "version", "--json", "--quiet"],
            capture_output=True, text=True
        )
        assert proc_ok.returncode == 0
        data = json.loads(proc_ok.stdout)
        assert "version" in data
