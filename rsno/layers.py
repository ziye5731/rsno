import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


'''
Iterative update layers.
Non-conditional:
Inputs:
    x: [B_t, width, H, W, C_t]
Outputs:
    x: [B_t, width, H, W, C_t]

Conditional:
Inputs:
    inp: (x, z)
Outputs:
    x
'''
    


class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """

    def __init__(
        self,
        in_codim,out_codim,
        modes,
        Normalize=False,
        Non_Lin=True,
    ):
        super(OperatorBlock_3D, self).__init__()
        self.conv = utils.SAConv_Uno(
            in_codim, out_codim, modes
        )
        '''self.conv = utils.SpectralConv3d_Uno(
            in_codim=in_codim,
            out_codim=out_codim,
            modes=modes
        )'''
        
        print(f'modes: {modes}')

        self.w = utils.pointwise_op_3D(in_codim, out_codim)
        #self.w = utils.hybrid_op_3D(in_codim, out_codim)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.norm = nn.LayerNorm(in_codim)

    def forward(self, x, dim1, dim2, dim3):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if self.normalize:
            x = self.norm(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x1_out = self.conv(x, dim1, dim2, dim3)
        x2_out = self.w(x, dim1, dim2, dim3)
        x_out = x1_out + x2_out

        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
