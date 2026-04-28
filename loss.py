import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_Loss(torch.nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    


class Gaussian_SRFP_Loss(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super(Gaussian_SRFP_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        '''
        Args:
            output: [B, M, C]
            target: [B, M, C]
        Return:
            loss: scalar
        '''
        B, M, C = output.size()
        Fitting_loss = torch.nn.MSELoss()(output, target)
        # collapse loss of output, to avoid it to be a constant
        Collapse_loss = -1*(output * torch.log2(output)).sum()/(B*M)
        return Fitting_loss + self.alpha*Collapse_loss


class RegressionandReconstruction(nn.Module):
    '''
    Loss function for PISSR.
    Args:
        reconstruction: float, reconstruction loss weight.
    '''
    def __init__(self, reconstruction=0.5):
        super().__init__()
        self.reconstruction = reconstruction

    def forward(self, pred, gt):
        '''
        Args:
            pred: tensor, predicted HSI, shape [B, C, H, W]
            gt: tuple, (hsi, srf), where
                hsi: tensor, ground truth HSI, shape [B, C, H, W]
                srf: tensor, spectral response function, shape [B, 3, C]
        Returns:
            loss: tensor, L1 loss + reconstruction loss
        '''
        hsi, srf = gt  # [B, C, H, W], [B, 3, C]
        b, c, h, w = hsi.shape

        # Calculate L1 loss
        l1_loss = F.l1_loss(pred, hsi)

        # Calculate reconstruction loss
        rgb_gt = torch.matmul(srf, hsi.view(b, c, -1)).view(-1, 3, h, w)  # [B, 3, H, W]
        rgb_pred = torch.matmul(srf, pred.view(b, c, -1)).view(-1, 3, h, w)  # [B, 3, H, W]
        reconstruction_loss = F.mse_loss(rgb_pred, rgb_gt)

        loss = l1_loss + self.reconstruction*reconstruction_loss
        
        return loss



class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        '''
        Args:
            pred: tensor, predicted HSI, shape [B, C, H, W]
            gt: tuple, (hsi, srf), where
                hsi: tensor, ground truth HSI, shape [B, C, H, W]
                srf: tensor, spectral response function, shape [B, 3, C]
        Returns:
            loss: tensor, L1 loss + reconstruction loss
        '''
        hsi, srf = gt  # [B, C, H, W], [B, 3, C]
        b, c, h, w = hsi.shape

        # Calculate L1 loss
        l1_loss = F.l1_loss(pred, hsi)
        
        return l1_loss





class MRAE(nn.Module):
    '''
    Mean Relative Absolute Error, based on NeSR.
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, hsi):
        '''
        Args:
            pred: tensor, predicted HSI, shape [B, C, H, W]
            hsi: tensor, ground truth HSI, shape [B, C, H, W]
        '''
        b, c, h, w = hsi.shape
        # Calculate MRAE
        pred = pred.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        gt = hsi.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]

        mrae = torch.abs((pred - gt)/(gt + 1e-3)).mean(dim=-1).mean(dim=-1).mean()
        return mrae


class MRAE_SAM_Loss(nn.Module):
    def __init__(self, lambda_sam=1.0, eps=1e-3):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps

    def forward(self, pred, hsi):
        """
        Args:
            pred: [B, C, H, W]
            hsi:  [B, C, H, W]
        Returns:
            loss = MRAE + lambda * SAM
        """
        B, C, H, W = hsi.shape
        pred = pred.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        gt   = hsi.view(B, C, -1).permute(0, 2, 1)

        # --- MRAE ---
        mrae = torch.abs(pred - gt) / (torch.abs(gt) + self.eps)  # [B, HW, C]
        mrae_loss = mrae.mean()

        # --- SAM ---
        dot = torch.sum(pred * gt, dim=-1)               # [B, HW]
        norm_pred = torch.norm(pred, dim=-1)
        norm_gt   = torch.norm(gt, dim=-1)
        cos = dot / (norm_pred * norm_gt + self.eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        sam_loss = torch.acos(cos).mean()

        return mrae_loss + self.lambda_sam * sam_loss
    

class MSE_SAM_Loss(nn.Module):
    def __init__(self, lambda_sam=0.1, eps=1e-8):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps

    def forward(self, pred, hsi):
        """
        Args:
            pred: predicted HSI, [B, C, H, W]
            hsi:  ground truth HSI, [B, C, H, W]
        Returns:
            loss = MSE + lambda * SAM
        """
        b, c, h, w = hsi.shape

        # ---- reshape: [B, H*W, C] ----
        pred = pred.view(b, c, -1).permute(0, 2, 1)
        gt   = hsi.view(b, c, -1).permute(0, 2, 1)

        # ---- 1) MSE ----
        mse_loss = torch.mean((pred - gt) ** 2)

        # ---- 2) SAM ----
        dot = torch.sum(pred * gt, dim=-1)                       # [B, HW]
        norm_pred = torch.norm(pred, dim=-1)                     # [B, HW]
        norm_gt   = torch.norm(gt, dim=-1)                       # [B, HW]
        cos = dot / (norm_pred * norm_gt + self.eps)             # [B, HW]
        cos = torch.clamp(cos, -1.0, 1.0)                        # 数值安全
        sam_loss = torch.acos(cos).mean()                       # 标量

        # ---- 3) 组合 ----
        return mse_loss + self.lambda_sam * sam_loss
    

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