"""Provides tests for module calc_metric."""

import numpy as np

from openairclim.core import calc_metric


class TestGetMetricsDict:
    """Tests function get_metrics_dict(config, t_zero, horizon, resp_dict)."""

    def test_simple(self):
        """Simple case with only one species and time_metrics subset of time_range."""
        config = {"time": {"range": [2000, 2020, 1]}}
        t_zero = 2000
        horizon = 10
        resp_dict = {"spec": np.arange(0, 2.0, 0.1)}
        expected_output = {"spec": np.arange(0, 1.0, 0.1)}
        computed_output = calc_metric.get_metrics_dict(
            config, t_zero, horizon, resp_dict
        )
        np.testing.assert_equal(expected_output, computed_output)


class TestCalcAegwp:
    """Tests function calc_aegwp(config, t_zero, horizon, rf_dict)."""

    def test_simple(self):
        """Single species case, checks that AGWP is weighted by efficacy."""
        t_zero = 2000
        horizon = 10
        config = {
            "time": {"range": [t_zero, t_zero + horizon, 1]},
            "temperature": {"spec": {"efficacy": 2.0}},
        }
        rf_dict = {"spec": np.full(horizon, 0.5)}
        expected_output = {"spec": 10.0, "total": 10.0}
        computed_output = calc_metric.calc_aegwp(config, t_zero, horizon, rf_dict)
        np.testing.assert_equal(expected_output, computed_output)

    def test_diff_agwp_aegwp(self):
        """Checks that AGWP and AEGWP differ only by efficacy."""
        t_zero = 2000
        horizon = 10
        efficacy = 2.0
        config = {
            "time": {"range": [t_zero, t_zero + horizon, 1]},
            "temperature": {"spec": {"efficacy": efficacy}},
        }
        rf_dict = {"spec": np.full(horizon, 0.5)}
        agwp_results = calc_metric.calc_agwp(config, t_zero, horizon, rf_dict)["spec"]
        aegwp_results = calc_metric.calc_aegwp(config, t_zero, horizon, rf_dict)["spec"]
        np.testing.assert_allclose(agwp_results * efficacy, aegwp_results)

    def test_multiple_species(self):
        """Multiple species case, checks per-species efficacy weighting and total."""
        t_zero = 2000
        horizon = 5
        config = {
            "time": {"range": [t_zero, t_zero + horizon, 1]},
            "temperature": {
                "spec1": {"efficacy": 1.0},
                "spec2": {"efficacy": 2.0},
            },
        }
        rf_dict = {
            "spec1": np.full(horizon, 1.0),
            "spec2": np.full(horizon, 0.5),
        }
        expected_output = {"spec1": 5.0, "spec2": 5.0, "total": 10.0}
        computed_output = calc_metric.calc_aegwp(config, t_zero, horizon, rf_dict)
        np.testing.assert_equal(expected_output, computed_output)
