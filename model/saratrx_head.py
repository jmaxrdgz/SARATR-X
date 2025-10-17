from typing_extensions import List

import torch
# from mmpretrain.mmpretrain.registry import MODELS
# from mmpretrain.mmpretrain.models.heads import MAEPretrainHead

# from .mgf import MGF

# @MODELS.register_module()
# class SARATRXHead(MAEPretrainHead):
#     """SARATR-X Head 
    
#     A PyTorch implement of: `SARATR-X: Toward Building A Foundation Model
#     for SAR Target Recognition <https://arxiv.org/pdf/2405.09365>`_.

#     Args:
#         loss (dict): Config of loss.
#         norm_pix_loss (bool): Whether or not normalize target.
#             Defaults to False.
#         patch_size (int): Patch size. Defaults to 16.
#         in_channels (int): Number of input channels. Defaults to 3.
#         mgf_kensizes (List[int]): List of kernel sizes for MGF.
#             Defaults to [9, 13, 17]
#     """

#     def __init__(self,
#                  loss: dict,
#                  norm_pix: bool = False,
#                  patch_size: int = 16,
#                  in_channels: int = 3,
#                  mgf_kensizes: List[int] = [9, 13, 17]) -> None:
#         super().__init__(
#             loss=loss,
#             norm_pix=norm_pix,
#             patch_size=patch_size,
#             in_channels=in_channels)
        
#         self.sarfeature = MGF(mgf_kensizes)
    
#     def construct_target(self, target: torch.Tensor) -> torch.Tensor:
#         """Construct the reconstruction target.

#         In addition to splitting images into tokens, this module will also
#         normalize the image according to ``norm_pix``.

#         Args:
#             target (torch.Tensor): Image with the shape of B x C x H x W

#         Returns:
#             torch.Tensor: Tokenized images with the shape of B x L x C
#         """
#         target = self.sarfeature(target)
#         target = self.patchify(target)

#         if self.norm_pix:
#             # normalize the target image
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6)**.5

#         return target
