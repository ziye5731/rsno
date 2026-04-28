import torch


def PSNR(output, gt):
    '''
    Calculate PSNR, assumed that pixel values range from 0 to 1.
    Args:
        output (Tensor[N, C, H, W]):
        gt (Tensor[N, C, H, W]): 
    Returns:
        psnr (Tensor[N]):
    '''
    mse = (output-gt).pow(2).mean(dim=[-3,-2,-1])  # [N]
    return -10*torch.log10(mse)


def SSIM(output, gt):
    '''
    Calculate SSIM, assumed that pixel values range from 0 to 1.
    Args:
        output (Tensor[N, C, H, W]):
        gt (Tensor[N, C, H, W]): 
    Returns:
        ssim (Tensor[N]):
    '''
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.mean(output, dim=(-3, -2, -1), keepdim=True)
    mu_y = torch.mean(gt, dim=(-3, -2, -1), keepdim=True)

    sigma_x = torch.var(output, dim=(-3, -2, -1), keepdim=True)
    sigma_y = torch.var(gt, dim=(-3, -2, -1), keepdim=True)
    sigma_xy = torch.mean((output - mu_x) * (gt - mu_y), dim=(-3, -2, -1), keepdim=True)

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = numerator / denominator
    ssim = torch.mean(ssim_map, dim=(-3, -2, -1))  # Average over spatial dimensions

    return ssim


def SAM(output, gt):
    '''
    Calculate Spectral Angle Mapper (SAM).
    Args:
        output (Tensor[N, C, H, W] or [C, H, W]):
        gt (Tensor[N, C, H, W] or [C, H, W]):
    Returns:
        sam (Tensor[N] or []):
    '''
    # Flatten spatial dimensions
    if output.dim() == 4:  # [N, C, H, W]
        output_flat = output.view(output.size(0), output.size(1), -1)  # [N, C, H*W]
        gt_flat = gt.reshape(gt.size(0), gt.size(1), -1)  # [N, C, H*W]
    elif output.dim() == 3:  # [C, H, W]
        output_flat = output.view(1, output.size(0), -1)  # [1, C, H*W]
        gt_flat = gt.reshape(1, gt.size(0), -1)  # [1, C, H*W]
    else:
        raise ValueError("Input tensors must have 3 or 4 dimensions.")

    # Compute dot product and norms
    dot_product = torch.sum(output_flat * gt_flat, dim=1)  # [N, H*W]
    norm_output = torch.norm(output_flat, dim=1)  # [N, H*W]
    norm_gt = torch.norm(gt_flat, dim=1)  # [N, H*W]

    # Avoid division by zero
    cos_theta = dot_product / (norm_output * norm_gt + 1e-10)  # [N, H*W]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to valid range for arccos

    # Compute SAM
    sam_map = torch.acos(cos_theta)  # [N, H*W]
    sam = torch.mean(sam_map, dim=1)  # Average over spatial dimensions

    if output.dim() == 3:  # If input was [C, H, W], return a single value
        sam = sam.squeeze(0)

    return sam


def RMSE(output, gt):
    '''
    Calculate RMSE, assumed that pixel values range from 0 to 1.
    Args:
        output (Tensor[N, C, H, W]):
        gt (Tensor[N, C, H, W]): 
    Returns:
        rmse (Tensor[N]):
    '''
    mse = (output - gt).pow(2).mean(dim=[-3, -2, -1])  # [N]
    return torch.sqrt(mse)


def MRAE(output, gt):
    '''
    Calculate Mean Relative Absolute Error (MRAE), assumed that pixel values range from 0 to 1.
    Args:
        output (Tensor[N, C, H, W]):
        gt (Tensor[N, C, H, W]): 
    Returns:
        mrae (Tensor[N]):
    '''
    abs_diff = torch.abs(output - gt)
    mean_gt = torch.mean(gt, dim=[-3, -2, -1], keepdim=True)  # Mean over spatial dimensions
    mrae = torch.mean(abs_diff / (mean_gt + 1e-10), dim=[-3, -2, -1])  # Avoid division by zero
    return mrae
