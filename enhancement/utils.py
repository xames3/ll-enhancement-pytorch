"""\
Project Utilities
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, November 05 2023
Last updated on: Monday, November 06 2023
"""

from __future__ import annotations

import abc
import datetime as dt
import logging
import os.path as p
import typing as t
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from . import _globals as global_settings

__all__ = [
    "BaseEnhancementCNN",
    "DimensionMismatchError",
    "InputNotATensorError",
    "_ModuleOrTensor",
    "_T",
    "config",
    "logger",
    "validate",
]

_ModuleOrTensor = torch.nn.Module | torch.Tensor
_Str = str | None
_Tensor = torch.Tensor
_T = t.TypeVar("_T", bound=t.Callable[..., t.Any])


class _ExecutionError(Exception):
    """Base class for all errors raised by the project."""

    _description: str

    def __init__(self, description: _Str = None) -> None:
        """Initialize exception with error description."""
        if description:
            self._description = description
        logger.error(self._description)
        super().__init__(self._description)


class InputNotATensorError(_ExecutionError):
    """Error to be raised when the input type is not a tensor."""


class DimensionMismatchError(_ExecutionError):
    """Error to be raised when the input dimensions don't match."""


class _Config:
    """Global configuration class."""

    def __init__(self) -> None:
        """Initialize from settings but only for ALL_CAPS settings."""
        for setting in dir(global_settings):
            if setting.isupper():
                setattr(self, setting, getattr(global_settings, setting))

    def _logger(
        self,
        name: _Str = None,
        level: int | None = None,
        fmt: _Str = None,
        handlers: list[logging.Handler] | None = None,
        capture_warnings: bool = True,
    ) -> logging.Logger:
        """Create a logger.

        This method initializes a logger with default configurations for
        the logging system. By default, the logger streams the output
        and logs the output to a logfile.

        .. code-block:: python

            >>> config = _Config()
            >>> logger = config._logger()
            >>> logger.info("...")

        :param name: Name for the logger, defaults to ``None``.
        :param level: Minimum logging level of the event, defaults
                      to ``None``.
        :param fmt: Format for the log message, defaults to ``None``.
        :param handlers: List of various logging handlers to use,
                         defaults to ``None``.
        :param capture_warnings: Boolean option to whether capture the
                                 warnings while logging, defaults to
                                 ``True``.
        :returns: Configured logger instance.
        """
        if name is None:
            name = f"enhancement.{self.MODEL_MODE}"
        logger = logging.getLogger(name)
        if level is None:
            level = (
                logging.DEBUG if self.MODEL_MODE == "train" else logging.INFO
            )
        logger.setLevel(level)
        if handlers is None:
            filename = (
                dt.datetime.now()
                .strftime(self.ISO8601)
                .translate(str.maketrans({"-": "_", ":": "_", ".": "_"}))
                + ".log"
            )
            filename = p.join(self.PROJECT_PATHS["../logs"], filename)
            handlers = [logging.StreamHandler(), logging.FileHandler(filename)]
        if fmt is None:
            fmt = self.LOG_FORMAT
            datefmt = self.ISO8601
        for handler in handlers:
            logger.addHandler(handler)
            handler.setFormatter(logging.Formatter(fmt, datefmt))
        logging.captureWarnings(capture_warnings)
        return logger


class _DeviceManager:
    """Handle device management for model and tensor allocations.

    The ``_DeviceManager`` class is designed to abstract away the
    details of device management within the context of Neural Network
    training and inference. It allows users for easy switching between
    different computational devices, such as CPUs and GPUs, and ensures
    that both the model and the data are on the same device to prevent
    runtime errors.

    The class chooses an optimum device depending on the system it is
    running upon, reducing the hassle of choosing the right device type.

    If CUDA or MPS is requested but not available,
    the ``_DeviceManager`` will fall back to using the CPU. This ensures
    that the code can run even on systems without a GPU.

    .. code-block:: python

        >>> import torch.nn as nn
        >>> class CustomCNN(nn.Module):
        ...     def __init__(self, input_channels):
        ...         super().__init__()
        ...         self.device_manager = _DeviceManager()
        ...         self.to(self.device_manager.device)
        ...         ...
        ...         ...
        ...     def forward(self, x):
        ...         x = self.device_manager.to(x)[0]
        ...         ...

    .. seealso::
        [1] PyTorch's device specifications: https://shorturl.at/jlpBU
    """

    def __init__(self) -> None:
        """Initialize ``_DeviceManager`` with suitable device type."""
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def to(self, *args: t.Sequence[_ModuleOrTensor]) -> list[_ModuleOrTensor]:
        """Move the model or tensors to the specified device.

        Moves the given model or tensors to the device managed by this
        instance. This ensures that the computation for both the model
        and the data takes place on the same device, which is essential
        for the efficiency and correctness of the operations.

        :param args: A sequence of PyTorch modules or tensors to move to
                     the device.
        """
        return [arg.to(self._device) for arg in args]


def validate(func: _T) -> _T:
    """Decorator for validating the inputs to a Convolutional Neural
    Network (CNN).

    This decorator wraps a CNN's ``forward`` method to perform checks on
    the input tensor, ensuring that it is a 4-dimensional object and has
    the correct number of channels as expected by the model.

    .. code-block:: python

        >>> import torch.nn as nn
        >>> class CustomCNN(nn.Module):
        >>>     def __init__(self):
        ...         super().__init__()
        ...         ...
        ...         ...
        ...     @validate
        ...     def forward(self, x):
        ...         pass

    :param: The ``forward`` method to be wrapped.
    :returns: The wrapped ``forward`` method with input validation.
    """

    @wraps(func)
    def forward(
        self,
        x: _Tensor,
        y: _Tensor | None = None,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Abstracted ``forward`` method."""
        if not isinstance(x, _Tensor):
            raise InputNotATensorError(
                "Function expected input of type ``torch.Tensor``, got "
                f"``{type(x).__name__}`` instead"
            )
        if x.dim() != 4:
            raise DimensionMismatchError(
                "Function expected input with 4-dimensions (batch, channel, "
                f"height and width), but got {x.dim()} dimensions instead"
            )
        return func(self, x, y, *args, **kwargs)

    return forward


class _MetaBaseEnhancementCNN(abc.ABCMeta):
    """A metaclass for injecting layers into a Convolutional Neural
    Network (CNN) class based on class annotations.

    This metaclass enables the declaration of neural network layers
    within class annotations, promoting a declarative style of model
    definition. It automatically adds these annotated layers as class
    attributes during class creation, allowing for a more modular and
    organized structure of the neural network definition. It recognizes
    both individual layers and sequences of layers facilitating a
    declarative and modular structure for defining neural network models.
    """

    def __new__(
        mcls: type[_MetaBaseEnhancementCNN],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> _MetaBaseEnhancementCNN:
        """Create new layers based on class annotations."""
        cls = super().__new__(mcls, name, bases, namespace)
        for name, obj in namespace.items():
            if isinstance(obj, dict):
                if "layer" in obj and isinstance(obj["layer"], nn.Module):
                    setattr(cls, name, obj["layer"])
                elif "sequence" in obj:
                    layers = [
                        layer()
                        if isinstance(layer, type)
                        and issubclass(layer, nn.Module)
                        else layer
                        for layer in obj["sequence"]
                    ]
                    setattr(cls, name, nn.Sequential(*layers))
        return cls


class BaseEnhancementCNN(nn.Module, metaclass=_MetaBaseEnhancementCNN):
    """An abstract base class for CNN models utilizing a metaclass for
    layer injection.

    This class serves as a blueprint for CNN architectures, ensuring
    that all derived classes implement the necessary forward pass and
    training methods. It uses a metaclass to inject layers defined in
    class annotations into the class itself, facilitating a clear and
    declarative way of composing neural network models.

    The class adheres to the ``SOLID`` design principles by providing a
    stable base for CNN models that can be extended but not modified,
    ensuring that instances of subclasses can substitute for the base
    class and maintaining a single responsibility for each class.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize ``BaseEnhancementCNN`` model."""
        super().__init__(*args, **kwargs)
        self.device = _DeviceManager().device
        logger.info(f"Using device: {self.device.upper()}...")
        self.to(self.device)

    @abc.abstractmethod
    def forward(
        self,
        x: _Tensor,
        y: _Tensor | None = None,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Defines the forward pass of the CNN.

        Must be implemented by all subclasses to define the sequence of
        operations that the input tensor should go through to produce
        the model's output. It is the core function that maps inputs to
        predictions.

        :param x: A 4-dimensional tensor representing the input batch.
        :param y: Optional tensor depending on the model requirement.
        :returns: The output of the model after the forward pass.
        """
        raise NotImplementedError


config = _Config()
logger = config._logger()
