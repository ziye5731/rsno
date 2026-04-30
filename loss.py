import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_SAM_Loss(nn.Module):
    def __init__(self, lambda_sam=0.1, eps=1e-8):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps

    def forward(self, pred, hsi):
        B, C, H, W = hsi.shape
        pred = pred.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        gt   = hsi.view(B, C, -1).permute(0, 2, 1)

        # L1
        l1_loss = torch.mean(torch.abs(pred - gt))

        # SAM
        dot = torch.sum(pred * gt, dim=-1)
        norm_pred = torch.norm(pred, dim=-1)
        norm_gt   = torch.norm(gt, dim=-1)
        cos = dot / (norm_pred * norm_gt + self.eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        sam_loss = torch.acos(cos).mean()

        return l1_loss + self.lambda_sam * sam_loss
    

class L1_SAM_R_Loss(nn.Module):
    def __init__(self, lambda_sam=0.1, alpha=0.5, eps=1e-8):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps
        self.alpha = alpha

    def forward(self, pred, hsi, srf):
        B, C, H, W = hsi.shape
        pred = pred.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        gt   = hsi.view(B, C, -1).permute(0, 2, 1)

        # L1
        l1_loss = torch.mean(torch.abs(pred - gt))

        # SAM
        dot = torch.sum(pred * gt, dim=-1)
        norm_pred = torch.norm(pred, dim=-1)
        norm_gt   = torch.norm(gt, dim=-1)
        cos = dot / (norm_pred * norm_gt + self.eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        sam_loss = torch.acos(cos).mean()

        # Reconstruction
        rgb_gt = torch.matmul(srf, hsi.view(B, C, -1)).view(-1, 3, H, W)  # [B, 3, H, W]
        rgb_pred = torch.matmul(srf, pred.reshape(B, C, -1)).view(-1, 3, H, W)  # [B, 3, H, W]
        reconstruction_loss = F.l1_loss(rgb_pred, rgb_gt)


        return l1_loss + self.lambda_sam * sam_loss + self.alpha * reconstruction_loss