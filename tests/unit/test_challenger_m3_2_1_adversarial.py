"""Adversarial stress-testing suite for select_metrics and diversity/ratio edge cases.

Author: Challenger M3-2-1 (Empirical Challenger)
Focus:
  - NaN / +inf / -inf in diversity/ratio (must raise ValueError)
  - String / non-numeric / bool / complex / list / None in diversity/ratio (must raise ValueError)
  - values with diversity_ratio only (must normalize to diversity/ratio and emit float)
  - values with both diversity_ratio and diversity/ratio
  - values with neither diversity_ratio nor diversity/ratio (must omit without raising or creating unavailable status)
  - config['metrics']['disable'] = ['diversity/ratio'] (must not emit diversity/ratio even when present in row)
  - interaction with progress scalars and step cadence
"""
import math
import pytest

from hypergan.config import resolve_config
from hypergan.metrics import metric_catalog, select_metrics


@pytest.fixture
def standard_setup():
    config = resolve_config({"metrics": {"every_steps": 1}})
    catalog = metric_catalog(config)
    return config, catalog


class TestSelectMetricsAdversarialDiversityRatio:
    """Adversarial stress tests for diversity/ratio in select_metrics."""

    @pytest.mark.parametrize("bad_val", [
        float("nan"),
        float("inf"),
        -float("inf"),
    ])
    def test_nonfinite_diversity_ratio_raises_value_error(self, standard_setup, bad_val):
        """values with NaN or inf in diversity/ratio must raise ValueError."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": bad_val}
        with pytest.raises(ValueError, match="Nonfinite or invalid metric value: diversity/ratio"):
            select_metrics(config, catalog, row, step=1, step_seconds=0.01)

    @pytest.mark.parametrize("bad_val", [
        float("nan"),
        float("inf"),
        -float("inf"),
    ])
    def test_nonfinite_aliased_diversity_ratio_raises_value_error(self, standard_setup, bad_val):
        """values with NaN or inf in diversity_ratio (alias) must also raise ValueError."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity_ratio": bad_val}
        with pytest.raises(ValueError, match="Nonfinite or invalid metric value: diversity/ratio"):
            select_metrics(config, catalog, row, step=1, step_seconds=0.01)

    @pytest.mark.parametrize("invalid_val", [
        "0.5",
        "invalid",
        "",
        None,
        True,
        False,
        [0.5],
        {"ratio": 0.5},
        complex(1, 2),
    ])
    def test_non_numeric_diversity_ratio_raises_value_error(self, standard_setup, invalid_val):
        """values with string, None, boolean, or complex diversity/ratio must raise ValueError."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": invalid_val}
        with pytest.raises(ValueError, match="Nonfinite or invalid metric value: diversity/ratio"):
            select_metrics(config, catalog, row, step=1, step_seconds=0.01)

    @pytest.mark.parametrize("invalid_val", [
        "0.5",
        "invalid",
        None,
        True,
        False,
        [0.5],
    ])
    def test_non_numeric_aliased_diversity_ratio_raises_value_error(self, standard_setup, invalid_val):
        """values with string or non-numeric diversity_ratio must also raise ValueError."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity_ratio": invalid_val}
        with pytest.raises(ValueError, match="Nonfinite or invalid metric value: diversity/ratio"):
            select_metrics(config, catalog, row, step=1, step_seconds=0.01)

    def test_diversity_ratio_alias_normalization(self, standard_setup):
        """values with diversity_ratio only must normalize to diversity/ratio."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity_ratio": 0.35}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01)
        assert "diversity/ratio" in metrics
        assert metrics["diversity/ratio"] == 0.35
        assert isinstance(metrics["diversity/ratio"], float)
        assert kind == "sampled"

    def test_diversity_ratio_integer_coercion(self, standard_setup):
        """values with integer diversity/ratio (e.g. 0 or 1) must be coerced to float."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": 0}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01)
        assert "diversity/ratio" in metrics
        assert metrics["diversity/ratio"] == 0.0
        assert type(metrics["diversity/ratio"]) is float

    def test_diversity_ratio_both_keys_present(self, standard_setup):
        """When both primary and alias are present, primary value is preserved."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": 0.42, "diversity_ratio": 0.99}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01)
        assert metrics["diversity/ratio"] == 0.42

    def test_missing_diversity_ratio_omitted_without_error_or_unavailable(self, standard_setup):
        """values with neither diversity/ratio nor diversity_ratio must omit without raising or creating unavailable status."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01)
        assert "diversity/ratio" not in metrics
        assert "diversity/ratio" not in statuses
        assert not any(name.startswith("diversity/") for name in metrics.keys() | statuses.keys())

    def test_disabled_metric_does_not_emit(self):
        """When config['metrics']['disable'] = ['diversity/ratio'], diversity/ratio must not be emitted."""
        config = resolve_config({
            "metrics": {
                "every_steps": 1,
                "disable": ["diversity/ratio"],
            }
        })
        catalog = metric_catalog(config)
        assert "diversity/ratio" not in catalog["metrics"]

        # Even if row provides diversity/ratio, it must not be emitted
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": 0.77, "diversity_ratio": 0.77}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01)
        assert "diversity/ratio" not in metrics
        assert "diversity/ratio" not in statuses

        # And if disabled, bad values in row must not trigger ValueError because the metric is disabled in catalog
        row_bad = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": float("nan")}
        metrics, statuses, kind = select_metrics(config, catalog, row_bad, step=1, step_seconds=0.01)
        assert "diversity/ratio" not in metrics

    def test_cadence_skipping_behavior(self):
        """Step not on cadence returns empty dicts and kind 'cadence'."""
        config = resolve_config({"metrics": {"every_steps": 5}})
        catalog = metric_catalog(config)
        row = {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": 0.42}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=2, step_seconds=0.01)
        assert metrics == {}
        assert statuses == {}
        assert kind == "cadence"

    def test_progress_scalars_inclusion(self, standard_setup):
        """progress dict can also supply or accompany diversity/ratio."""
        config, catalog = standard_setup
        row = {"d_loss": 0.5, "g_loss": 0.5}
        progress = {"diversity_ratio": 0.65}
        metrics, statuses, kind = select_metrics(config, catalog, row, step=1, step_seconds=0.01, progress=progress)
        assert metrics["diversity/ratio"] == 0.65
