"""Fully-connected classifier network."""
from typing import List

from torch import Tensor
from torch.nn import Sequential, Module, Linear, ReLU, BatchNorm1d, Dropout

# pylint: disable=too-many-instance-attributes, too-many-arguments


class LinearBlock(Module):
    """Simple `Linear` block with optional extras.

    A block (optionally) combining `BatchNorm1d`, `Dropout`, `Linear` and
    `ReLU`.

    Parameters
    ----------
    input_size : int
        The number of input features to the block.
    output_size : int
        The number of output features the block should yield.
    batchnorm : bool
        Bool determining if a `BatchNorm1d` is applied to the input.
    dropout : float
        The level of dropout to apply to the input (after batchnorm).
    relu : bool
        Determines if a `ReLU` activation is applied to the output of the
        linear layer.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        batchnorm: bool = False,
        dropout: float = 0.0,
        relu: bool = False,
    ):
        """Set up `LinearBlock`."""
        super().__init__()
        self._check_args(input_size, output_size, batchnorm, dropout, relu)
        self.fwd = Sequential()
        self._add_components(input_size, output_size, batchnorm, dropout, relu)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the network.

        Parameters
        ----------
        batch : Tensor
            An input batch.

        Returns
        -------
        Tensor
            The output of the model.

        """
        return self.fwd(batch)

    def _add_components(
        self,
        input_size: int,
        output_size: int,
        batchnorm: bool,
        dropout: float,
        relu: bool,
    ):
        """Add the components to `self.block` to make up the forward pass."""
        if batchnorm is True:
            self.fwd.add_module("BatchNorm", BatchNorm1d(input_size))

        if dropout != 0.0:
            self.fwd.add_module("Dropout", Dropout(dropout))

        self.fwd.add_module(
            "Linear",
            Linear(input_size, output_size),
        )

        if relu is True:
            self.fwd.add_module("ReLU", ReLU())

    @staticmethod
    def _check_args(
        input_size: int,
        output_size: int,
        batchnorm: bool,
        dropout: float,
        relu: bool,
    ):
        _check_layer_size(input_size)
        _check_layer_size(output_size)
        _check_bool_arg(batchnorm)
        _check_dropout_arg(dropout)
        _check_bool_arg(relu)


class ClassifierFC(Module):
    """Fully connected classifier model.

    Parameters
    ----------
    layer_sizes : List[int]
        List of layer sizes. For example, passing [10, 5, 2] gives a model
        with 10 input features, one hidden layer of size 5, and an output layer
        of size 2.
    input_batchnorm : bool
        Should we apply a batchnorm directly to the classifier's input?
    hidden_batchnorms : bool
        Should we include batchnorms in the hidden layers?
    input_dropout : float
        Dropout probability to apply to the classifier's input. Must be on
        [0, 1].
    hidden_dropout : float
        Dropout probability to use in the hidden layers. Must be on [0, 1].

    """

    def __init__(
        self,
        layer_sizes: List[int],
        input_batchnorm: bool = False,
        hidden_batchnorm: bool = False,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        """Construct the fully connected classifier."""
        super().__init__()
        self.fwd = self._create_layers(
            layer_sizes,
            input_batchnorm,
            input_dropout,
            hidden_batchnorm,
            hidden_dropout,
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Feed `batch` forward.

        Parameters
        ----------
        batch : Tensor
            Mini-batch of inputs.

        Returns
        -------
        Tensor
            Model output.

        """
        return self.fwd(batch)

    @staticmethod
    def _create_layers(
        sizes: List[int],
        input_batchnorm: bool,
        input_dropout: float,
        hidden_batchnorm: bool,
        hidden_dropout: float,
    ) -> Sequential:
        """Create a `Sequential` of layers making upn the forward pass."""
        if not isinstance(sizes, list):
            msg = f"Layer sizes should be list. Got {type(sizes)}."
            raise TypeError(msg)

        if (num := len(sizes)) < 2:
            msg = f"Must specify at least two layer sizes. Got {num}."
            raise RuntimeError(msg)

        fwd = Sequential()

        for idx, (ins, outs) in enumerate(zip(sizes[:-1], sizes[1:])):

            if idx == 0:
                batchnorm = input_batchnorm
                dropout = input_dropout
            else:
                batchnorm = hidden_batchnorm
                dropout = hidden_dropout

            include_relu = idx != (len(sizes) - 2)

            block = LinearBlock(
                ins,
                outs,
                batchnorm=batchnorm,
                dropout=dropout,
                relu=include_relu,
            )
            fwd.add_module(f"Block{idx+1}", block)

        return fwd


def _check_layer_size(size: int) -> None:
    """Process any argument detailing the size of a layer.

    Parameters
    ----------
    size : int
        The size (or number of features) of a linear layer.

    Raises
    ------
    TypeError
        If size is not an integer.
    ValueError
        If size does not exceed 0.

    """
    if not isinstance(size, int):
        msg = f"Layer input and output sizes should be int. Got {type(size)}"
        raise TypeError(msg)
    if not size > 0:
        msg = f"Layer input and output sizes be >= 1. Got {size}"
        raise ValueError(msg)


def _check_bool_arg(bool_arg: bool) -> None:
    """Process argument controling batchnorm layers.

    Parameters
    ----------
    bool_arg : bool
        Argument controlling whether a respective batchnorm layer should be
        included or not.

    Raises
    ------
    TypeError
        If `bnorm` is not boolean.


    """
    if not isinstance(bool_arg, bool):
        raise TypeError(f"Expected bool arg, got {type(bool_arg)}.")


def _check_dropout_arg(dropout: float) -> None:
    """Process argument determing level of dropout.

    Parameters
    ----------
    dropout : float
        Level of dropout to apply.


    Raises
    ------
    TypeError
        If `dropout` is not a float.
    ValueError
        If `not 0.0 <= dropout <= 1`.

    """
    if not isinstance(dropout, float):
        raise TypeError(f"Dropout arg should be float. Got {type(dropout)}.")

    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"Droput arg should be on [0, 1]. Got {dropout}.")
