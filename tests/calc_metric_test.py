"""
Provides tests for module calc_metric
"""

import numpy as np
import openairclim as oac


def test_get_metrics_dict_simple():
    """Simple case with only one species and time_metrics subset of time_range"""
    config = {"time": {"range": [2000, 2020, 1]}}
    t_zero = 2000
    horizon = 10
    resp_dict = {"spec": np.arange(0, 2.0, 0.1)}
    expected_output = {"spec": np.arange(0, 1.0, 0.1)}
    computed_output = oac.get_metrics_dict(config, t_zero, horizon, resp_dict)
    np.testing.assert_equal(expected_output, computed_output)
