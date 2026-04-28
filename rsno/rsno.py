import torch
import torch.nn as nn
import pandas as pd

from . import utils
from . import interp



class RSNO(nn.Module):
    def __init__(self, prior_data: pd.DataFrame, no: nn.Module,
                 refinement=True,
                 *args, **kwargs):
        '''
        Args:
            prior_data: pd.DataFrame, contains prior data.
                wavelength: pd.Series[N_t], wavelength in ascending order, unit: um.
                value: pd.Series[N_t], extraterrestrial irradiance.
            no: nn.Module, neural operator, 
                 should take in [B_t, d_m, H, W, C_t] and [B_t, d_smarts, C_t] as input, and output [B_t, 1, H, W, C_t].
        '''
        super().__init__()
        prior_data['value'] = prior_data['value']
        self.prior = utils.KernelReg1D(prior_data['wavelength'], prior_data['value'])
        
        self.interp_module = interp.ACP()
        self.no = no

        self.refinement = refinement
        
    def forward(self, inp):
        '''
        Forward process, not that "_t" means that the variable is arbitrary.
        Args:
            inp: tuple, (rgb, coord, srf), hyperspectral image and target coordinates.
                rgb: torch.Tensor[B_t, M, H, W], input RGB image.
                coord: torch.Tensor[B_t, C_t], target coordinates.
                srf: torch.Tensor[B_t, M, C_t], spectral response function.
        Returns:
            hsi: torch.Tensor[B_t, C_t, H, W], output hyperspectral image.
        '''
        rgb, coord, srf = inp
        B_t, M, H, W = rgb.shape

        # Get ART prior
        #prior = self.scale * self.prior(coord)  # [B_t, C_t]
        prior = self.prior(coord)  # [B_t, C_t]


        # Get input RGB image map
        rgb_interp = self.interp_module(rgb, srf, prior.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))  # [B_t, C_t, H, W]
        rgb_interp_map = rgb_interp.permute(0, 2, 3, 1).unsqueeze(-4)  # [B_t, 1, H, W, C_t]

        # Predict hsi
        inp_rgb = rgb_interp_map  # [B_t, 1, H, W, C_t]
        inp_coord = coord.unsqueeze(-2)  # [B_t, 1, C_t]
        inp_no = (inp_rgb, inp_coord)

        out_no = self.no(inp_no)  # [B_t, 1, H, W, C_t]
        
        # Residual connection
        out_no = out_no.squeeze(-4).permute(0, 3, 1, 2)  # [B_t, C_t, H, W]
        hsi = rgb_interp + out_no  # [B_t, C_t, H, W]

        # Refinement
        if self.refinement:
            hsi = self.interp_module(rgb, srf, hsi)  # [B_t, C_t, H, W]
        else:
            pass

        return hsi
