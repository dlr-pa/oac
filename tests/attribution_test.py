"""
Provides tests for module attribution
"""

from typing import Literal
import numpy as np
import pytest
import openairclim as oac


def _func_factory(
    mode: Literal["constant", "linear", "affine"] = "linear",
    *,
    scale: float = 1.0,
    value: float = 1.0,
    offset: float = 0.0,
):
    """
    Factory for simple test functions:
    - constant: f(x) = value
    - linear:   f(x) = scale * x
    - affine:   f(x) = scale * x + offset
    """

    def func(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if mode == "constant":
            return {
                k: np.full_like(v, fill_value=value, dtype=float)
                for k, v in data.items()
            }
        if mode == "linear":
            return {k: scale * v for k, v in data.items()}
        if mode == "affine":
            return {k: scale * v + offset for k, v in data.items()}

        raise ValueError(f"Unknown mode: {mode!r}")

    return func


class TestApplyAttribution:
    """Tests function apply_attribution(func, diff_func, method, species,
    sub_dict, full_dict)."""

    def test_invalid_method(self):
        """Test that an invalid attribution raises an error."""
        with pytest.raises(ValueError):
            oac.apply_attribution(None, None, "invalid", "", {}, {})

    @pytest.mark.parametrize(
        "att_func",
        [
            oac.residual_attribution,
            oac.marginal_attribution,
            oac.proportional_attribution,
            oac.differential_attribution,
        ],
    )
    def test_output_shape(self, att_func):
        """Check that the output has the same shape as the input for all
        attribution methods."""
        species = "CO2"
        sub_vals = np.linspace(0.0, 10.0, 11)
        sub_dict = {species: sub_vals}
        full_dict = {species: sub_vals.copy()}
        func = _func_factory(mode="constant", value=1.0)

        result = att_func(func, sub_dict, full_dict, species)

        assert species in result
        assert result[species].shape == sub_vals.shape

    @pytest.mark.parametrize(
        "att_func",
        [
            oac.residual_attribution,
            oac.marginal_attribution,
            oac.proportional_attribution,
            oac.differential_attribution,
        ],
    )
    def test_zero_subdict(self, att_func):
        """If the sub_dict contribution is zero, the attribution should also
        be zero for all methods."""
        species = "CO2"
        sub_vals = np.zeros(11)
        sub_dict = {species: sub_vals}
        full_dict = {species: np.ones(sub_vals.shape)}
        func = _func_factory(mode="constant", value=3.14)

        result = att_func(func, sub_dict, full_dict, species)

        np.testing.assert_allclose(result[species], 0.0)


class TestResidualAttribution:
    """Tests function residual_attribution(func, sub_dict, full_dict, species,
    **kwargs)."""

    def test_equal_dicts(self):
        """If sub_dict == full_dict, then the residual attribution should be
        func(full) - func(0) = func(full)."""
        species = "CO2"
        sub_vals = np.array([1.0, 2.0, 3.0, 4.0])
        sub_dict = {species: sub_vals}
        full_dict = {species: sub_vals.copy()}
        func = _func_factory(mode="linear", scale=2.0)

        result = oac.residual_attribution(func, sub_dict, full_dict, species)

        expected = func(full_dict)[species]
        np.testing.assert_allclose(result[species], expected, atol=1e-12)

    def test_linear_func_with_bg(self):
        """For a linear function and full = background + sub, the result should
        be scale * sub_vals."""
        species = "CO2"
        background = np.array([5.0, 5.0, 5.0])
        sub_vals = np.array([1.0, 2.0, 3.0])
        full_vals = background + sub_vals
        sub_dict = {species: sub_vals}
        full_dict = {species: full_vals}

        scale = 4.0
        func = _func_factory(mode="linear", scale=scale)

        result = oac.residual_attribution(
            func=func,
            sub_dict=sub_dict,
            full_dict=full_dict,
            species=species,
        )

        expected = scale * sub_vals
        np.testing.assert_allclose(result[species], expected, atol=1e-12)

    def test_affine_func_with_bg(self):
        """For an affine function and background, the offset should cancel in
        the residual."""
        species = "CO2"
        background = np.array([2.0, 2.0, 2.0])
        sub_vals = np.array([0.0, 1.0, 2.0])
        full_vals = background + sub_vals

        sub_dict = {species: sub_vals}
        full_dict = {species: full_vals}

        scale = 3.0
        offset = 10.0
        func = _func_factory(mode="affine", scale=scale, offset=offset)

        result = oac.residual_attribution(
            func=func,
            sub_dict=sub_dict,
            full_dict=full_dict,
            species=species,
        )

        expected = scale * sub_vals
        np.testing.assert_allclose(result[species], expected, atol=1e-12)


class TestMarginalAttribution:
    """Tests function marginal_attribution(diff_func, sub_dict, full_dict,
    species, **kwargs)."""

    @pytest.mark.parametrize("deriv_val", [0.5, 1.0, 2.0])
    def test_linear_time_series(self, deriv_val):
        """For a linear time series x[t] = t and a constant derivative,
        np.gradient(x) should approximate the derivative and the attribution
        should equal deriv_val * t."""
        species = "CO2"
        n = 15
        sub_vals = np.arange(1, n, dtype=float)
        sub_dict = {species: sub_vals}
        full_dict = {species: sub_vals.copy()}
        diff_func = _func_factory(mode="constant", value=deriv_val)

        result = oac.marginal_attribution(diff_func, sub_dict, full_dict, species)

        expected = deriv_val * sub_vals
        np.testing.assert_allclose(result[species], expected, atol=1e-12)


class TestProportionalAttribution:
    """Tests function proportional_attribution(func, sub_dict, full_dict,
    species, **kwargs)."""

    def test_equal_dicts(self):
        """If sub_dict == full_dict, proportion is 1 everywhere and attribution
        should equal func(full_dict)."""
        species = "CO2"
        sub_vals = np.array([1.0, 2.0, 3.0, 4.0])
        sub_dict = {species: sub_vals}
        full_dict = {species: sub_vals.copy()}
        func = _func_factory(mode="linear", scale=2.0, offset=5.0)

        result = oac.proportional_attribution(func, sub_dict, full_dict, species)

        expected = func(full_dict)[species]
        np.testing.assert_allclose(result[species], expected, atol=1e-12)

    def test_even_split(self):
        """If full is the sum of two equal components and func is linear,
        proportional attribution to each component should be half of
        func(full)."""
        species = "CO2"
        comp = np.array([1.0, 2.0, 3.0])
        full_vals = comp + comp
        sub_dict = {species: comp}
        full_dict = {species: full_vals}
        func = _func_factory(mode="linear", scale=2.0)

        result = oac.proportional_attribution(func, sub_dict, full_dict, species)

        expected = 0.5 * func(full_dict)[species]
        np.testing.assert_allclose(result[species], expected, atol=1e-12)


class TestDifferentialAttribution:
    """Tests function differential_attribution(diff_func, sub_dict, full_dict,
    species, **kwargs)."""

    @pytest.mark.parametrize("deriv_val", [0.5, 1.0, 2.0])
    def test_linear_time_series(self, deriv_val):
        """For a linear time series x[t] = t and a constant derivative,
        np.gradient(x) should approximate the derivative and the attribution
        should equal deriv_val * t."""
        species = "CO2"
        n = 15
        sub_vals = np.arange(1, n, dtype=float)
        sub_dict = {species: sub_vals}
        full_dict = {species: sub_vals.copy()}
        diff_func = _func_factory(mode="constant", value=deriv_val)

        result = oac.differential_attribution(diff_func, sub_dict, full_dict, species)

        expected = deriv_val * sub_vals
        np.testing.assert_allclose(result[species], expected, atol=1e-12)
