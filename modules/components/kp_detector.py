import torch
from typing import Dict, Any
import torch.nn as nn
import torch.nn.functional as F
from .utils import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid


class KPD(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion: int = 64, num_kp: int=10, num_channels: int=3, max_features: int=512,
                 num_blocks: int=3, temperature: float=0.1, estimate_jacobian: bool=False, scale_factor: float=1,
                 single_jacobian_map: bool=False, pad: int=0,training: bool=False):
        super(KPD, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)
        self.training = training

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        if self.training:
            noise = torch.empty_like(value).uniform_(-0.05, 0.05)
            value += noise
        kp = {'value': value} 
        return kp

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            # if self.training:
            #     noise = torch.empty_like(jacobian).uniform_(-0.5, 0.5)
            #     jacobian += noise
            out['jacobian'] = jacobian 
        return out

    

if __name__ == "__main__":
    img = torch.randn((1,3,256,256))
    kp_detector = KPD(estimate_jacobian=True)
    kps = kp_detector(img)
    print(kps['value'].shape, kps['kp_bits'],kps['jacobian'].shape,kps['j_bits'])