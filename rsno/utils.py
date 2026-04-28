import torch
import torch.nn as nn
import torch.nn.functional as F



class KernelReg1D(nn.Module):
    def __init__(self, x_data, y_data, bandwidth=0.005, learnable_bandwidth=False):
        super().__init__()
        self.x_data = nn.Parameter(torch.tensor(x_data, dtype=torch.float32), requires_grad=False)  # shape [M]
        self.y_data = nn.Parameter(torch.tensor(y_data, dtype=torch.float32), requires_grad=False)  # shape [M]

        self.learnable_bandwidth = learnable_bandwidth
        self.log_bw = nn.Parameter(
            torch.log(torch.tensor(bandwidth)), requires_grad=learnable_bandwidth
        )

    def forward(self, x):
        # x: [B, N]
        B, N = x.shape
        M = self.x_data.shape[0]

        # x: [B, N, 1], x_data: [1, 1, M]
        x_expanded = x.unsqueeze(-1)               # [B, N, 1]
        x_data_expanded = self.x_data.view(1, 1, M)  # [1, 1, M]
        dist2 = (x_expanded - x_data_expanded) ** 2  # [B, N, M]
        bw = torch.exp(self.log_bw)
        weights = torch.exp(-dist2 / (2 * bw**2))  # [B, N, M]
        weights = weights / weights.sum(dim=-1, keepdim=True)  # [B, N, M]

        # y_data: [1, 1, M]
        y_data_expanded = self.y_data.view(1, 1, M)
        y = torch.sum(weights * y_data_expanded, dim=-1)  # [B, N]
        return y



class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x



class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class SAConv_Uno(nn.Module):
    def __init__(
        self, in_codim, out_codim, modes
    ):
        super(SAConv_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        
        self.modes = modes

        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights = nn.Parameter(
            self.scale
            * torch.randn(
                in_codim,
                out_codim,
                self.modes,
                dtype=torch.cfloat,
            )
        )


    # Complex multiplication
    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioz->boxyz", input, weights)

    def forward(self, x, dim1, dim2, dim3):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """

        batchsize = x.shape[0]
        H, W, C_t = x.shape[-3], x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="forward")

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            H,
            W,
            C_t // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :, :, : self.modes] = self.compl_mul3d(
            x_ft[:, :, :, :, : self.modes], self.weights
        )


        # Return to physical space
        x = torch.fft.irfftn(
            out_ft, s=(dim1, dim2, dim3), norm="forward"
        )
        return x
