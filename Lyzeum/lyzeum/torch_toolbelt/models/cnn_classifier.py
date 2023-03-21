"""Module of cutomised pytorch networks."""
from typing import Tuple, List, Optional, Union


import torch
from torch import nn
import torchvision as vision  # type: ignore
from . import ClassifierFC

# pylint: disable=too-many-arguments


class ClassifierCNN(nn.Module):
    """CNN Classifier model comrpised of `encoder`, `pool` and `classifier`.

    Parameters
    ----------
    encoder : str
        The choice of encoder you wish to use. See `_architectures` for
        options.
    num_classes : int
        The size of the prediction vector the classifier should produce.
    pretrained : bool, optional
        Determines if we use Pytorch's pretrained version of the encoder.
    pool : str, optional
        The kind of pooling layer you wish to use. The options are `avg`,
        `max` or `concat`. `concat` is simply a layer which concatenates both
        the average and max pooling operations (See `AdaptiveConcatPool2d`).
    clf_layer_sizes : List[int], optional
        List of integer sizes for the hidden layers in the classifier. `None`
        means there will be a single linear layer to map between the features
        and the output.
    clf_input_batchnorm : bool, optional
        Determines whether we apply a one-dimensional batchnorm to the
        input of the classifier.
    clf_hidden_batchnorm : bool, optional
        Determines if we include one-dimensional batchnorms between the hidden
        layers of the classifier.
    clf_input_dropout : float, optional
        The dropout probability to apply to the input of the classifier.
    clf_hidden_drouput : float, optional
        The dropout probability to apply at the hidden layers of the
        classifier.

    """

    def __init__(
        self,
        encoder: str,
        num_classes: int,
        pretrained: bool = True,
        pool_style: str = "avg",
        clf_hidden_sizes: Optional[List[int]] = None,
        clf_input_batchnorm: bool = False,
        clf_hidden_batchnorm: bool = False,
        clf_input_dropout: float = 0.0,
        clf_hidden_dropout: float = 0.0,
    ):
        """Construct ClassifierCNN."""
        super().__init__()
        encoder_feats, self.encoder = _get_encoder(encoder, pretrained)
        self.pool = _get_2d_pool(pool_style, encoder)

        _num_feats = self._num_encoder_feats(encoder_feats, pool_style)
        _clf_layer_sizes = self._classifier_layer_sizes(
            _num_feats,
            clf_hidden_sizes,
            num_classes,
        )

        self.classifier = ClassifierFC(
            _clf_layer_sizes,
            input_batchnorm=clf_input_batchnorm,
            hidden_batchnorm=clf_hidden_batchnorm,
            input_dropout=clf_input_dropout,
            hidden_dropout=clf_hidden_dropout,
        )

    @staticmethod
    def _classifier_layer_sizes(
        num_features: int,
        hidden_sizes: Union[List[int], None],
        num_classes: int,
    ) -> List[int]:
        """List the sizes of the linear layer sizes to go in classifier.

        Parameters
        ----------
        num_features : int
            The number of features the encoder will produce (after pool).
        hidden_sizes : List[int] or None
            The user-request hidden layers to add to the classifier.
        num_classes : int
            The number of output classes.

        Returns
        -------
        List[int]
            The layer sizes in the classifier.

        """
        if not isinstance(hidden_sizes, (list, type(None))):
            msg = f"hidden_sizes should be a list. Got {type(hidden_sizes)}"
            raise TypeError(msg)
        return [num_features] + (hidden_sizes or []) + [num_classes]

    @staticmethod
    def _num_encoder_feats(encoder_feats: int, pool_style: str) -> int:
        """Return the number of features in the encoder.

        Parameters
        ----------
        encoder_feats : int
            The number of features the backbone model's encoder would
            naturally yield.
        pool_style : str
            The type of pool the user requests.

        Returns
        -------
        int
            The number of features to be returned by the pooling layer.

        """
        return (2 * encoder_feats) if pool_style == "concat" else encoder_feats

    def forward(
        self,
        batch: torch.Tensor,
        frozen_encoder: bool = False,
    ) -> torch.Tensor:
        """Pass `batch` through the network.

        Parameters
        ----------
        batch : torch.Tensor
            Tensor holding the batch of input images.
        frozen_encoder : bool, optional
            Bool determining whether the encoder is frozen or not.

        """
        if frozen_encoder is True:
            self.encoder.eval()
            with torch.no_grad():
                encoder_out = self.encoder(batch)
        else:
            self.encoder.train()
            encoder_out = self.encoder(batch)
        features = self.pool(encoder_out)
        return self.classifier(features)


class AdaptiveConcatPool2d(nn.Module):
    """AdaptiveAvgPool2d and AdaptiveMaxPool2d.

    Parameters
    ----------
    output_size : tuple of int
        Output Dimensions of the adaptive pooling layers.

    """

    def __init__(self, output_size: Tuple[int]):
        """Construct ConcatPool."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, batch: torch.Tensor):
        """Apply and concatenate the two pooling operations.

        Parameters
        ----------
        batch : torch.Tensor
            A mini-batch of inputs.

        """
        return torch.cat([self.avg_pool(batch), self.max_pool(batch)], dim=1)


_architectures = {
    "resnet18": vision.models.resnet18,
    "resnet34": vision.models.resnet34,
    "resnet50": vision.models.resnet50,
    "vgg11": vision.models.vgg11,
    "vgg13": vision.models.vgg13,
    "vgg16": vision.models.vgg16,
    "vgg11_bn": vision.models.vgg11_bn,
    "vgg13_bn": vision.models.vgg13_bn,
    "vgg16_bn": vision.models.vgg16_bn,
}


_pools = {
    "avg": nn.AdaptiveAvgPool2d,
    "max": nn.AdaptiveMaxPool2d,
    "concat": AdaptiveConcatPool2d,
}


def _check_encoder_option(option: str):
    """Check the type and value of the encoder option.

    Parameters
    ----------
    option : str
        The string denoting the encoder choice.

    Raises
    ------
    TypeError
        If `option` is not a str.
    ValueError
        If `option` is not in `_architectures`.

    """
    if not isinstance(option, str):
        raise TypeError(f"Encoder option should be a str. Got {type(option)}.")
    if not option in _architectures:
        msg = f"Encoder option {option} is unsupported. Please choose from "
        msg += f"{list(_architectures.keys())}."
        raise ValueError(msg)


def _check_pretrained_option(pretrained: bool):
    """Check the pretrained option for the encoder.

    Parameters
    ----------
    pretrained : bool
        The pretrained option for the encoder.

    Raises
    ------
    TypeError
        If `pretrained` is not a bool.

    """
    if not isinstance(pretrained, bool):
        msg = f"pretrained option should be bool. Got {type(pretrained)}."
        raise TypeError(msg)


def _get_encoder(option: str, pretrained: bool) -> Tuple[int, nn.Sequential]:
    """Return an encoder based on `option`.

    Parameters
    ----------
    option : str
        Encoder choice.
    pretrained : bool
        Should we use pytorch's pretrained version?

    Returns
    -------
    num_feats : nn.Sequential
        The dimensionality of the feature space.
    encoder : nn.Sequential
        Encoder of choice.

    """
    _check_encoder_option(option)
    _check_pretrained_option(pretrained)

    model = _architectures[option](pretrained=pretrained).to("cpu")

    if "resnet" in option:
        encoder_list = list(model.children())[:-2]
        num_feats = model.fc.in_features
    if "vgg" in option:
        encoder_list = list(model.features.children())
        num_feats = model.classifier[0].in_features

    encoder = nn.Sequential(*encoder_list)
    return num_feats, encoder


def _get_2d_pool(pool: str, encoder_style: str) -> nn.Sequential:
    """Return a 2D pooling layer.

    Parameters
    ----------
    pool : str
        Choose from: "avg", "max" or "concat". The former two return average
        or max pooling layers, and the latter yeilds a layer which
        concatenates both the average and maximum pools.
    encoder_style : str
        The style of encoder to copy the output size from.

    Returns
    -------
    nn.Sequential
        The pooling layer with an optional flattening operation.

    """
    if not pool in _pools:
        msg = f"Pool option should be one of {list(_pools.keys())}. "
        msg += f"Got '{pool}'"
        raise ValueError(msg)
    output_size = _architectures[encoder_style]().avgpool.output_size
    return nn.Sequential(_pools[pool](output_size), nn.Flatten())
