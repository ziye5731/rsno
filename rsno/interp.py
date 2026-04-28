import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

'''
interpolation module
Inputs:
    x: [B_t, M_t, H_t, W_t]
    srf: [B_t, M_t, C_t]
    *args
Outputs:
    y: [B_t, C_t, H_t, W_t]
'''


class ACP(nn.Module):
    def __init__(self):
        '''
        Interpolation module based on SAM-minimizing guidance.
        '''
        super().__init__()

    def forward(self, x, srf, z, *args, **kwargs):
        '''
        Inputs:
            x: [B, M, H, W] - LR observation
            srf: [B, M, C] - spectral response function
            z: [B, C, H, W] - prior guidance
        Outputs:
            y: [B, C, H, W] - reconstructed HR image
        '''
        B, M, H, W = x.shape
        C = z.shape[1]
        N = H * W

        x_flat = x.view(B, M, N)  # [B, M, N]
        z_flat = z.view(B, C, N)  # [B, C, N]

        # Compute SS^T and its inverse
        SS_T = torch.einsum('bmc,bnc->bmn', srf, srf)  # [B, M, M]
        SS_T_inv = torch.linalg.pinv(SS_T)  # [B, M, M]

        # Compute S^T (SS^T)^-1 = A, shape [B, C, M]
        A = torch.einsum('bcm,bmn->bcn', srf.transpose(-1, -2), SS_T_inv)

        # Compute projection matrix part: P = I - A @ S
        AS = torch.einsum('bcm,bmn->bcn', A, srf)  # [B, C, C]
        I = torch.eye(C, device=x.device).expand(B, C, C)  # [B, C, C]
        P = I - AS  # [B, C, C]

        # Compute λ = (X^T (SS^T)^-1 X) / (X^T (SS^T)^-1 S Z)
        x_SS_inv = torch.einsum('bmn,bnd->bmd', SS_T_inv, x_flat)  # [B, M, N]
        num = torch.sum(x_flat * x_SS_inv, dim=1)  # [B, N]

        Sz = torch.einsum('bmc,bcn->bmn', srf, z_flat)  # [B, M, N]
        Sz_proj = torch.einsum('bmn,bnd->bmd', SS_T_inv, Sz)  # [B, M, N]
        denom = torch.sum(x_flat * Sz_proj, dim=1) + 1e-8  # [B, N]

        lamb = (num / denom).clamp(min=0).unsqueeze(1)  # [B, 1, N]
        z_sam = lamb * z_flat  # [B, C, N]

        # Final result: Y = A @ x + P @ z_sam
        Ax = torch.einsum('bcm,bmn->bcn', A, x_flat)  # [B, C, N]
        Pz = torch.einsum('bcp,bpn->bcn', P, z_sam)  # [B, C, N]

        y = Ax + Pz
        return y.view(B, C, H, W)
