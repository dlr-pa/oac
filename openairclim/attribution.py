"""
Provides attribution functionality for non-linear species.
"""

from typing import Protocol
import numpy as np


# DEFINE TYPE STYLES
class AttributionFunc(Protocol):
    """Defines type protocol for attribution function inputs."""
    def __call__(
        self,
        data: dict[str, np.ndarray],
        /,
        **kwargs
    ) -> dict[str, np.ndarray]:
        ...


def apply_attribution(
    func: AttributionFunc,
    method: str,
    species: str,
    sub_dict: dict[str, np.ndarray],
    full_dict: dict[str, np.ndarray],
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Applies attribution methodology using the function `func` for `sub_dict`.

    Args:
        func (AttributionFunc): A callable that maps a dict of species arrays to
            another such dict. The first argument must be a
            dict[str, np.ndarray].
        method (str): Attribution method. Choice of: "none", "residual", 
            "marginal", "proportional", "differential".
        species (str): Name of the species to consider (e.g. 'CO2').
        sub_dict (dict[str, np.ndarray]): Dict of species arrays representing a
            part of the full dict (e.g. one aircraft identifier).
        full_dict (dict[str, np.ndarray]): Dict of species arrays representing
            all sub-dicts. **Must include `sub_dict`!**
        **kwargs: Additional keyword arguments to pass to `func`.

    Raises:
        ValueError: If an invalid method is given.

    Returns:
        dict[str, np.ndarray]: `func` value attributable to `sub_dict`
    """

    if method == "none":
        return func(sub_dict, **kwargs)

    if method == "residual":
        return residual_attribution(
            func, sub_dict, full_dict, species, **kwargs
        )
    if method == "marginal":
        return marginal_attribution(
            func, sub_dict, full_dict, species, **kwargs
        )
    if method == "proportional":
        return proportional_attribution(
            func, sub_dict, full_dict, species, **kwargs
        )
    if method == "differential":
        return differential_attribution(
            func, sub_dict, full_dict, species, **kwargs
        )

    raise ValueError(f"Invalid attribution method for species {species}")


def residual_attribution(
    func: AttributionFunc,
    sub_dict: dict[str, np.ndarray],
    full_dict: dict[str, np.ndarray],
    species: str,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Calculates the `func` value for species `species` attributable to `sub_dict`
    using a residual methodology. The dict `full_dict`, representing all other
    sources (i.e. background plus other aircraft identifiers) must include
    `sub_dict`.

    Args:
        func (AttributionFunc): A callable that maps a dict of species arrays to
            another such dict. The first argument must be a
            dict[str, np.ndarray].
        sub_dict (dict[str, np.ndarray]): Dict of species arrays representing a
            part of the full dict (e.g. one aircraft identifier).
        full_dict (dict[str, np.ndarray]): Dict of species arrays representing
            all sub-dicts. **Must include `sub_dict`!**
        species (str): Name of the species to consider (e.g. 'CO2').
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        dict[str, np.ndarray]: `func` value attributable to `sub_dict`
    """

    # calculate difference between full and sub dicts
    diff_dict = {species: full_dict[species] - sub_dict[species]}

    # do attribution
    full_res_arr = func(full_dict, **kwargs)[species]
    diff_res_arr = func(diff_dict, **kwargs)[species]
    return {species: full_res_arr - diff_res_arr}


def marginal_attribution(
    func: AttributionFunc,
    sub_dict: dict[str, np.ndarray],
    full_dict: dict[str, np.ndarray],
    species: str,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Calculates the `func` value for species `species` attributable to `sub_dict`
    using a marginal methodology. The dict `full_dict`, representing all other
    sources (i.e. background plus other aircraft identifiers) must include
    `sub_dict`.

    Args:
        func (AttributionFunc): A callable that maps a dict of species arrays to
            another such dict. The first argument must be a
            dict[str, np.ndarray].
        sub_dict (dict[str, np.ndarray]): Dict of species arrays representing a
            part of the full dict (e.g. one aircraft identifier).
        full_dict (dict[str, np.ndarray]): Dict of species arrays representing
            all sub-dicts. **Must include `sub_dict`!**
        species (str): Name of the species to consider (e.g. 'CO2').
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        dict[str, np.ndarray]: `func` value attributable to `sub_dict`
    """

    # initialise
    full_inp_arr = full_dict[species]
    full_res_arr = func(full_dict, **kwargs)[species]

    # calculate gradient
    idx = np.argsort(full_inp_arr)
    deriv_sorted = np.gradient(full_res_arr[idx], full_inp_arr[idx], edge_order=2)
    deriv = np.empty_like(deriv_sorted)
    deriv[idx] = deriv_sorted

    return {species: deriv * sub_dict[species]}


def proportional_attribution(
    func: AttributionFunc,
    sub_dict: dict[str, np.ndarray],
    full_dict: dict[str, np.ndarray],
    species: str,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Calculates the `func` value for species `species` attributable to `sub_dict`
    using a proportional methodology. The dict `full_dict`, representing all other
    sources (i.e. background plus other aircraft identifiers) must include
    `sub_dict`.

    Args:
        func (AttributionFunc): A callable that maps a dict of species arrays to
            another such dict. The first argument must be a
            dict[str, np.ndarray].
        sub_dict (dict[str, np.ndarray]): Dict of species arrays representing a
            part of the full dict (e.g. one aircraft identifier).
        full_dict (dict[str, np.ndarray]): Dict of species arrays representing
            all sub-dicts. **Must include `sub_dict`!**
        species (str): Name of the species to consider (e.g. 'CO2').
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        dict[str, np.ndarray]: `func` value attributable to `sub_dict`
    """

    # do attribution
    full_res_arr = func(full_dict, **kwargs)[species]
    prop = sub_dict[species] / full_dict[species]
    return {species: prop * full_res_arr}


def differential_attribution(
    func: AttributionFunc,
    sub_dict: dict[str, np.ndarray],
    full_dict: dict[str, np.ndarray],
    species: str,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Calculates the `func` value for species `species` attributable to `sub_dict`
    using a differential methodology. The dict `full_dict`, representing all other
    sources (i.e. background plus other aircraft identifiers) must include
    `sub_dict`.

    Args:
        func (AttributionFunc): A callable that maps a dict of species arrays to
            another such dict. The first argument must be a
            dict[str, np.ndarray].
        sub_dict (dict[str, np.ndarray]): Dict of species arrays representing a
            part of the full dict (e.g. one aircraft identifier).
        full_dict (dict[str, np.ndarray]): Dict of species arrays representing
            all sub-dicts. **Must include `sub_dict`!**
        species (str): Name of the species to consider (e.g. 'CO2').
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        dict[str, np.ndarray]: `func` value attributable to `sub_dict`
    """

    # initialise
    full_inp_arr = full_dict[species]
    full_res_arr = func(full_dict, **kwargs)[species]

    # calculate gradient (with respect to input)
    idx = np.argsort(full_inp_arr)
    deriv_sorted = np.gradient(full_res_arr[idx], full_inp_arr[idx], edge_order=2)
    deriv = np.empty_like(deriv_sorted)
    deriv[idx] = deriv_sorted

    # do attribution
    # lower triangular cumulative sum of products
    sub_vals = sub_dict[species]
    sub_grad = np.gradient(sub_vals)  # derivative (with respect to time)
    p = deriv * sub_grad
    res_arr = np.cumsum(p, axis=0)
    res_arr = np.roll(res_arr, 1)
    res_arr[0] = 0.0

    return {species: res_arr}
