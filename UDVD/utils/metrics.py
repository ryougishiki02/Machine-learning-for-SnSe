import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def ssim(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)

    # print(clean.shape,noisy.shape)

    win_size = 7  # 确保窗口大小适合您的图像尺寸

    # for c, n in zip(clean, noisy):
        # print(c.shape,n.shape)
        # print(f"Frame range after transpose: min={c.min()}, max={c.max()}")
        # print(f"Frame range after transpose: min={n.min()}, max={n.max()}")
    return np.array([structural_similarity(c, n, data_range=255, multichannel=True, channel_axis=-1, win_size=win_size) for c, n in zip(clean, noisy)]).mean()

def psnr(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)
    
    # if normalized:
    #     return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
    # else:
    return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
