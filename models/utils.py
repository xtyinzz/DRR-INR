"""
Utility functions for selecting the appropriate forward pass function based on the model type.
"""
from typing import Callable, Type
from torch.nn import Module

from .baselines.explorable_inr import INR_FG
from .baselines.fainr.attention_MoE_cloverleaf import KVMemoryModel
from .baselines.kplane import KPlaneField
from .modules import fwd_hdinr, fwd_inrsurrogate
from .srns import HDINRBase


def get_fwd_fn(model: Module) -> Callable:
    """
    Retrieves the appropriate forward function based on the model's class.

    This function is used to dynamically select a forward pass implementation
    that is compatible with a given model architecture.

    Args:
        model: The neural network model instance.

    Returns:
        The corresponding forward function for the given model.

    Raises:
        NotImplementedError: If the model type is not supported.
    """
    BASELINES: tuple[Type[Module], ...] = (
        INR_FG,
        KVMemoryModel,
        KPlaneField,
    )

    if isinstance(model, HDINRBase):
        return fwd_hdinr
    if isinstance(model, BASELINES):
        return fwd_inrsurrogate

    raise NotImplementedError(f"No forward function found for model type: {type(model).__name__}")


def get_fwd_fn_test(model: Module) -> tuple[Callable, Callable]:
    """
    Retrieves the forward functions for training and testing.

    In this setup, the same forward function is used for both training and testing.
    This function provides a consistent interface for obtaining them.

    Args:
        model: The neural network model instance.

    Returns:
        A tuple containing the forward function for training and testing, respectively.
    """
    fwd_fn = get_fwd_fn(model)
    return fwd_fn, fwd_fn