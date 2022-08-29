import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any
from .utils import ResBlock2d, SameBlock2d, UpBlock2d,DownBlock2d,AntiAliasInterpolation2d
from .motion_predictor import MotionNetwork

class GeneratorDAC(nn.Module):
    """
    Re-Named from the 'OcclussionAwareGenerator' used in the First Order Model by Siarohin et al.
    --> Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    --> This is the generator used for Konuko et al. 2021 ICASSP paper "ULTRA-LOW BITRATE VIDEO CONFERENCING USING DEEP IMAGE ANIMATION"
     which included an adaptive refresh algorithm for quality scalability.
    """
    def __init__(self, num_channels: int = 3, num_kp: int = 10, block_expansion: int = 64, 
                        max_features: int = 1024, num_down_blocks: int = 2,num_bottleneck_blocks: int = 3,
                        estimate_occlusion_map: bool = False, dense_motion_params: Dict[str, Any] = None, 
                        estimate_jacobian: bool = False) -> None:
        super(GeneratorDAC, self).__init__()
        if dense_motion_params is not None:
            self.dense_motion_network = MotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,**dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))         

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = block_expansion * (2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
    
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        

    def deform_input(self, inp: torch.Tensor, deformation: torch.Tensor) -> torch.Tensor:
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)


    def forward(self, source_image: torch.Tensor, kp_source: Dict[str, torch.Tensor]=None,
                      kp_driving: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:       
        # Encoding (downsampling) part      
        out = self.first(source_image)        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            output_dict['flow'] = dense_motion['deformation']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None

            out = self.deform_input(out, dense_motion['deformation'])

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear', align_corners=True)
                out = out * occlusion_map
            output_dict["deformed"] = self.deform_input(source_image, dense_motion['deformation'])
        # Decoding part
        out = self.bottleneck(out) #input the weighted average 
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)
        output_dict["prediction"] = out
        return output_dict


class GeneratorHDAC(GeneratorDAC):
    """
    An extension of the GeneratorDAC to include a base layer video stream with scalable coding quality.
    This is our main contribution for the Konuko et al. 2022 ICIP Paper "A HYBRID DEEP ANIMATION CODEC FOR LOW-BITRATE VIDEO CONFERENCING"
    --> The base layer introduces 3 main benefits to the animation codec.
        1) Scene semantic segmentation information  which improves the background recontruction.
        2) Adds objects missing from the source frame but visible in the target.
        3) Consistent quality scalability without flickering in the reconstructed video
    """
    def __init__(self, num_channels: int=3, num_kp : int =10, block_expansion: int =64, 
                        max_features=1024, num_down_blocks: int = 2, num_bottleneck_blocks: int=3,
                        estimate_occlusion_map: bool=False, dense_motion_params: Dict[str, Any]=None, 
                        estimate_jacobian: bool=False, scale_factor: float=0.25) -> None:
        super(GeneratorHDAC, self).__init__(num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map, dense_motion_params, estimate_jacobian)
        
        in_features = block_expansion * (2 ** num_down_blocks)
        self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.base =  SameBlock2d(num_channels, in_features, kernel_size=(7, 7), padding=(3, 3))     
        
        main_bottleneck = []
        res_bottleneck = []
        
        for i in range(num_bottleneck_blocks):
            main_bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
            res_bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
    
        self.main_bottleneck = nn.ModuleList(main_bottleneck)
        self.res_bottleneck = nn.ModuleList(res_bottleneck)

    def forward(self, source_image: torch.Tensor,base_layer: torch.Tensor, 
                    kp_source: Dict[str, torch.Tensor]=None, 
                    kp_driving: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:       
        # Encoding (downsampling) part      
        base_out = self.down(base_layer)
        base_out = self.base(base_out)

        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            output_dict['flow'] = dense_motion['deformation']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
                
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear', align_corners=True)
                out = out * occlusion_map
            output_dict["deformed"] = self.deform_input(source_image, deformation)

        #block-wise fusion at the generator bottleneck
        for idx, layer in enumerate(self.main_bottleneck):
            out = layer(out)
            base_out = self.res_bottleneck[idx](base_out)
            out += base_out 
            
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)
        output_dict["prediction"] = out
        return output_dict


