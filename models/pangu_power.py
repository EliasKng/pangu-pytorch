import sys
import os
from typing import List, Tuple, Optional, Union
from era5_data import utils_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.layers import (
    PatchRecoveryPowerAllWithClippedReLU,
    PowerConv,
)
from era5_data.config import cfg
from models.pangu_model import PanguModel


class PanguPowerPatchRecovery(PanguModel):
    """Replaces the patch recovery layer of pangu with a new convolution that aims to predict power"""

    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[torch.device] = None,
    ) -> None:
        super(PanguPowerPatchRecovery, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        # Delete the pangu output layer
        del self._output_layer

        # Replace the output layer with new PatchRecovery
        self._output_power_layer = PatchRecoveryPowerAllWithClippedReLU(dims[-2])

    def forward(
        self,
        input: torch.Tensor,
        input_surface: torch.Tensor,
        statistics: torch.Tensor,
        maps: torch.Tensor,
        const_h: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone architecture"""
        # Embed the input fields into patches
        # input:(B, N, Z, H, W) ([1, 5, 13, 721, 1440])input_surface(B,N,H,W)([1, 4, 721, 1440])
        # x = checkpoint.checkpoint(self._input_layer, input, input_surface)
        x = self._input_layer(
            input, input_surface, statistics, maps, const_h
        )  # ([1, 521280, 192]) [B, spatial, C]

        # Encoder, composed of two layers
        # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper

        x = self.layers[0](x, 8, 181, 360)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 360, 181) to (8, 180, 91)
        x = self.downsample(x, 8, 181, 360)

        x = self.layers[1](x, 8, 91, 180)
        # Decoder, composed of two layers
        # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
        x = self.layers[2](x, 8, 91, 180)

        # Upsample from (8, 180, 91) to (8, 360, 181)
        x = self.upsample(x)

        # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
        x = self.layers[3](x, 8, 181, 360)  # ([1, 521280, 192])

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=-1)

        # Recover the output fields from patches
        output = self._output_power_layer(x, 8, 181, 360)

        return output

    def load_pangu_state_dict(self, device: torch.device) -> None:
        """Get the prepared state dict of the pretrained pangu weights"""
        checkpoint = torch.load(
            cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
        )
        pretrained_dict = checkpoint["model"]
        model_dict = self.state_dict()

        # Update the model's state_dict
        model_dict.update(pretrained_dict)

        # Remove keys from model_dict that contain _output_layer, since those are not present in this model
        keys_to_remove = [k for k in model_dict.keys() if "_output_layer" in k]
        for k in keys_to_remove:
            del model_dict[k]

        self.load_state_dict(model_dict)


class PanguPowerConv(PanguModel):
    """Adds convolutional layers to the output of pangu to use pangus output to predict power"""

    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[Union[torch.device, int]] = None,
    ) -> None:
        super(PanguPowerConv, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        self._conv_power_layers = PowerConv()

        # Re-Init weights
        super(PanguPowerConv, self).apply(self._init_weights)

    def forward(self, input, input_surface, statistics, maps, const_h, aux_constants):
        """Backbone architecture"""
        # Embed the input fields into patches
        # input:(B, N, Z, H, W) ([1, 5, 13, 721, 1440])input_surface(B,N,H,W)([1, 4, 721, 1440])
        # x = checkpoint.checkpoint(self._input_layer, input, input_surface)
        x = self._input_layer(
            input, input_surface, statistics, maps, const_h
        )  # ([1, 521280, 192]) [B, spatial, C]

        # Encoder, composed of two layers
        # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper

        x = self.layers[0](x, 8, 181, 360)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 360, 181) to (8, 180, 91)
        x = self.downsample(x, 8, 181, 360)

        x = self.layers[1](x, 8, 91, 180)
        # Decoder, composed of two layers
        # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
        x = self.layers[2](x, 8, 91, 180)

        # Upsample from (8, 180, 91) to (8, 360, 181)
        x = self.upsample(x)

        # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
        x = self.layers[3](x, 8, 181, 360)  # ([1, 521280, 192])

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=-1)

        # Recover the output fields from patches
        # output, output_surface = checkpoint.checkpoint(self._output_layer, x, 8, 181, 360)
        output_upper, output_surface = self._output_layer(x, 8, 181, 360)

        output_upper, output_surface = utils_data.normBackData(
            output_upper, output_surface, aux_constants["weather_statistics_last"]
        )

        output_power = self._conv_power_layers(output_upper, output_surface)

        # Return output_surface for visualization purposes only
        return output_power

    def load_pangu_state_dict(self, device: torch.device) -> None:
        checkpoint = torch.load(
            cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
        )

        checkpoint_powerconvdirect = torch.load(
            "/home/hk-project-test-mlperf/om1434/masterarbeit/wind_fusion/pangu_pytorch/result/PowerConvDirectTrain/24/models/best_checkpoint.pth",
            map_location=device,
            weights_only=False,
        )

        # print("Checkpoint keys: ", checkpoint['model'].keys())
        # print("Checkpoint PowerConvDirect keys: ", checkpoint_powerconvdirect['model'].keys())

        # Filter keys from checkpoint_powerconvdirect that contain "_conv_power_layers"
        conv_power_layers_keys = {
            k: v
            for k, v in checkpoint_powerconvdirect["model"].items()
            if "_conv_power_layers" in k
        }

        # Add the conv_power_layers_keys to the checkpoint model
        checkpoint["model"].update(conv_power_layers_keys)

        self.load_state_dict(checkpoint["model"], strict=False)


def main():
    pppr = PanguPowerPatchRecovery()
    pppr.load_pangu_state_dict(torch.device("cpu"))


if __name__ == "__main__":
    model = PanguPowerPatchRecovery()
    model.load_pangu_state_dict(torch.device("cpu"))
