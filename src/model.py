"""U-Net model backed by segmentation-models-pytorch."""

from torch import Tensor, nn
from segmentation_models_pytorch import Unet


class MyModel(nn.Module):
    """U-Net for grayscale X-rays and binary mask segmentation.

    The model produces raw logits use ``BCEWithLogitsLoss`` for training and
    apply ``sigmoid`` to its output when probabilities are needed.
    """

    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
    ) -> None:
        
        super().__init__()
        self.unet = Unet(
            encoder_name    = encoder_name,
            encoder_weights = None,
            in_channels     = in_channels,
            classes         = out_channels,
            activation      = None,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)
