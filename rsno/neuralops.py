import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers


'''
Neural operators for SSR models
inputs:
    inp: (r, s):
        r: [B_t, d_r, H, W, C_t]
        s: [B_t, d_s, C_t]
outputs:
    v: [B_t, d_out, H, W, C_t]
'''



class CatUNO(nn.Module):
    def __init__(self, d_r, d_s, d_out, width, layers_c, layers_t, modes=16, *args, **kwargs):
        '''
        Neural operator which concat r and s as input.
        Args:
            d_r: int, number of channels of r.
            d_s: int, number of channels of s.
            d_out: int, number of channels of output.
            width: int, number of hidden channels.
            layers_c: int, number of layers for contraction.
            layers_t: int, number of layers for transformation.
            modes: int, maximum number of Fourier modes to keep.
        '''
        super().__init__()
        self.d_r = d_r
        self.d_s = d_s
        self.d_out = d_out
        self.width = width
        self.layers_c = layers_c
        self.layers_t = layers_t
        self.num_layers = 2*layers_c + layers_t
        self.modes = modes


        self.p0 = nn.Conv3d(d_r + d_s, self.width//2, 1, 1, 0)
        self.p1 = nn.Conv3d(self.width//2, self.width, 1, 1, 0)


        G_contract = []
        scales = [1]
        for i in range(layers_c):
            previous_scale = 2**i
            next_scale = 2**(i+1)
            G_contract.append(
                layers.OperatorBlock_3D(
                    in_codim=previous_scale*self.width,
                    out_codim=next_scale*self.width,
                    modes=max(modes//previous_scale, 2),
                    Normalize=False,
                    Non_Lin=True,
                )
            )
            scales.append(next_scale)
        self.G_contract = nn.ModuleList(G_contract)
        self.scales = scales  # [1, 2, 4, ..., 2**layers_c]
        
        G_transform = []
        deepest_scale = 2**layers_c
        for i in range(layers_t):
            G_transform.append(
                layers.OperatorBlock_3D(
                    in_codim=deepest_scale*self.width,
                    out_codim=deepest_scale*self.width,
                    modes=max(modes//deepest_scale, 2),
                    Normalize=False,
                    Non_Lin=True,
                )
            )
        self.G_transform = nn.ModuleList(G_transform)

        G_expand = []
        for i in range(layers_c):
            previous_scale = min(2**(layers_c - i + 1), deepest_scale)
            next_scale = 2**(layers_c - i - 1)
            G_expand.append(
                layers.OperatorBlock_3D(
                    in_codim=previous_scale*self.width,
                    out_codim=next_scale*self.width,
                    modes=max(modes//next_scale, 2),
                    Normalize=False,
                    Non_Lin=True if i < layers_c - 1 else False,  # No activation for the last layer
                )
            )
        self.G_expand = nn.ModuleList(G_expand)

        self.q0 = nn.Conv3d(width, 2*width, 1, 1, 0)
        self.q1 = nn.Conv3d(2*width, d_out, 1, 1, 0)


        # init self.no
        for m in self.p0.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.p1.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.q0.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.q1.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def forward(self, inp):
        '''
        Args:
            inp: (r, s):
                r: [B_t, d_r, H, W, C_t]
                s: [B_t, d_s, C_t]
        Returns:
            v: [B_t, d_out, H, W, C_t]
        '''
        r, s = inp

        # Concat
        s = s.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, r.shape[2], r.shape[3], -1)  # [B_t, d_s, H, W, C_t]
        x = torch.cat([r, s], dim=1)  # [B_t, d_r + d_s, H, W, C_t]

        # Lift
        x = self.p0(x)  # [B_t, width//2, H, W, C_t]
        x = F.gelu(x)
        x0 = self.p1(x)  # [B_t, width, H, W, C_t]


        # Iterative updates
        H, W, C_t = x0.shape[-3], x0.shape[-2], x0.shape[-1]
        ## Contract
        x_c = [x0]
        for i in range(self.layers_c):
            x_c.append(
                self.G_contract[i](x_c[-1], H//self.scales[i+1], W//self.scales[i+1], C_t//self.scales[i+1])  # [B_t, (2**(i+1))*width, H//2**(i+1), W//2**(i+1), C_t]
            )
            

        ## Transform
        x_t = x_c[-1]
        for i in range(self.layers_t):
            x_t = self.G_transform[i](x_t, H//self.scales[-1], W//self.scales[-1], C_t//self.scales[-1]) + x_t
            # [B_t, (2**(i+1))*width, H//2**(i+1), W//2**(i+1)]

        ## Expand
        x_e = self.G_expand[0](x_t, H//self.scales[-2], W//self.scales[-2], C_t//self.scales[-2])  # [B_t, (2**(layers_c-1))*width, H//2**(layers_c-1), W//2**(layers_c-1), C_t]
        for i in range(1, self.layers_c):
            # x_e: [B_t, (2**(layers_c-i))*width, H//2**(layers_c-i), W//2**(layers_c-i), C_t]
            x_e_in = torch.cat([x_e, x_c[-(i+1)]], dim=-4)
            # [B_t, 2**(layers_c-i+1)*width, H//2**(layers_c-i), W//2**(layers_c-i), C_t]

            x_e = self.G_expand[i](x_e_in, H//self.scales[-(i+2)], W//self.scales[-(i+2)], C_t//self.scales[-(i+2)])
            # [B_t, (2**(layers_c-i-1))*width, H//2**(layers_c-i-1), W//2**(layers_c-i-1), C_t]

        # Project
        ## x_e: [B_t, width, H, W, C_t]
        y = self.q0(x_e+x0)
        y = F.gelu(y)
        y = self.q1(y)
        
        return y  # [B_t, d_out, H, W, C_t]
    