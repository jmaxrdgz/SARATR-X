import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MGF(nn.Module):
    """Multi-scale Gradient Features for SAR images.
    
    Computes gradient features at multiple scales to suppress speckle noise
    and extract target edges at different scales.
    
    Args:
        kensizes (list[int]): List of kernel sizes for multi-scale computation.
            Default: [9, 13, 17]
    """
    
    def __init__(self, kensizes=[9, 13, 17]):
        super(MGF, self).__init__()
        self.kensizes = kensizes
        
        # Create gradient kernels for each scale
        for i, kensize in enumerate(kensizes):
            self._create_and_register_kernels(kensize, scale_idx=i)
    
    def _create_and_register_kernels(self, kensize, scale_idx):
        """Create and register convolution kernels for a given scale."""
        r = kensize
        
        def _create_gauss_kernel(r):
            M_13 = np.concatenate([np.ones([r+1, 2*r+1]), np.zeros([r, 2*r+1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2*r+1]), np.ones([r+1, 2*r+1])], axis=0)
            M_11 = np.concatenate([np.ones([2*r+1, r+1]), np.zeros([2*r+1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2*r+1, r]), np.ones([2*r+1, r+1])], axis=1)
            return (torch.from_numpy(M_13).float(), 
                    torch.from_numpy(M_23).float(), 
                    torch.from_numpy(M_11).float(), 
                    torch.from_numpy(M_21).float())
        
        M13, M23, M11, M21 = _create_gauss_kernel(r)
        
        weight_x1 = M11.view(1, 1, r*2+1, r*2+1)
        weight_x2 = M21.view(1, 1, r*2+1, r*2+1)
        weight_y1 = M13.view(1, 1, r*2+1, r*2+1)
        weight_y2 = M23.view(1, 1, r*2+1, r*2+1)
        
        # Register buffers with scale-specific names
        self.register_buffer(f"weight_x1_scale{scale_idx}", weight_x1)
        self.register_buffer(f"weight_x2_scale{scale_idx}", weight_x2)
        self.register_buffer(f"weight_y1_scale{scale_idx}", weight_y1)
        self.register_buffer(f"weight_y2_scale{scale_idx}", weight_y2)
    
    @torch.no_grad()
    def forward(self, x):
        """Compute multi-scale gradient features.
        
        Args:
            x (Tensor): Input SAR image with shape [B, C, H, W]
            
        Returns:
            Tensor: Concatenated multi-scale gradient features [B, C*len(kensizes), H, W]
        """
        gradient_features = []
        
        for scale_idx, kensize in enumerate(self.kensizes):
            # Get kernels for this scale
            weight_x1 = getattr(self, f"weight_x1_scale{scale_idx}")
            weight_x2 = getattr(self, f"weight_x2_scale{scale_idx}")
            weight_y1 = getattr(self, f"weight_y1_scale{scale_idx}")
            weight_y2 = getattr(self, f"weight_y2_scale{scale_idx}")
            
            # Compute gradient for this scale
            gf = self._compute_single_scale(x, kensize, weight_x1, weight_x2, 
                                           weight_y1, weight_y2)
            gradient_features.append(gf)
        
        # Concatenate all scales
        mgf = torch.cat(gradient_features, dim=1)
        return mgf
    
    def _compute_single_scale(self, x, kensize, weight_x1, weight_x2, 
                             weight_y1, weight_y2):
        """Compute gradient feature for a single scale."""
        k = kensize
        
        # Pad input
        x_padded = F.pad(x, pad=(k, k, k, k), mode="reflect") + 1e-2
        
        # Compute ratio-based gradients in x and y directions
        gx_1 = F.conv2d(x_padded, weight_x1, bias=None, stride=1, padding=0, groups=1)
        gx_2 = F.conv2d(x_padded, weight_x2, bias=None, stride=1, padding=0, groups=1)
        gy_1 = F.conv2d(x_padded, weight_y1, bias=None, stride=1, padding=0, groups=1)
        gy_2 = F.conv2d(x_padded, weight_y2, bias=None, stride=1, padding=0, groups=1)
        
        # Compute log ratio gradients
        gx = torch.log(gx_1 / (gx_2 + 1e-8))
        gy = torch.log(gy_1 / (gy_2 + 1e-8))
        
        # Compute gradient magnitude
        norm = torch.sqrt(gx**2 + gy**2)
        
        return norm