import torch
import torch.nn.functional as F
import skimage.restoration as skr
import numpy as np
from skimage.metrics import structural_similarity

def loss_function(output, truth, mode="loglike", sigma=25, device="cpu"):
    if(mode == "mse"):
        # print(truth.size(0), F.mse_loss(output, truth, reduction="sum"))
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif(mode == "loglike"):
        eps = 1e-5
        N,C,H,W = truth.shape
        mean = output[0:N, 0:C, 0:H, 0:W].permute(0,2,3,1).reshape(N, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0,2,3,1)
        truth = truth.permute(0,2,3,1).reshape(N, H, W, C, 1)
        ax = torch.zeros(N, H, W, int(C*C)).to(device)
        I = torch.eye(C).reshape(1,1,1,C,C).repeat(N, H, W, 1, 1).to(device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, H, W, C, C)
        sigma2I = (((sigma**2)+eps)*I.permute(1,2,3,4,0)).permute(4,0,1,2,3)
        variance = torch.matmul(ax.transpose(3,4), ax) + sigma2I #(sigma**2)*I

        # # 确保矩阵正定性
        # inv_variance = torch.inverse(variance + eps * I)
        # likelihood = 0.5 * torch.matmul(torch.matmul((truth - mean).transpose(3, 4), inv_variance), (truth - mean))
        # likelihood = likelihood.reshape(N, H, W)
        # det_term = 0.5 * torch.logdet(variance + eps * I)  # 使用logdet代替log(det)
        # likelihood += det_term
        # print(f"likelihood min value: {likelihood.min().item()}, likelihood max value: {likelihood.max().item()}")
        # loss = torch.mean(likelihood.mean(dim=(1, 2)) - 0.1 * sigma)

        # 原代码
        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(3,4), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N,H,W)
        likelihood += 0.5*torch.log(torch.det(variance))
#         loss = torch.mean(likelihood)
        loss = torch.mean(likelihood.mean(dim=(1,2)) - 0.1*sigma)

        # print("Final loss value:", loss.item())
    elif(mode == 'combine'):
        alpha = 0.5
        mse = F.mse_loss(output, truth)
        clean = output.cpu().detach().numpy().astype(np.float32).transpose(0, 2, 3, 1)
        noisy = truth.cpu().detach().numpy().astype(np.float32).transpose(0, 2, 3, 1)
        win_size = 7  # 确保窗口大小适合您的图像尺寸
        ssim_values = np.array([
            structural_similarity(c, n, data_range=1, multichannel=True, channel_axis=-1, win_size=win_size)
            for c, n in zip(clean, noisy)]).mean()
        ssim_values = torch.tensor(ssim_values)
        ssim = 1 - ssim_values

        loss = alpha * mse + (1 - alpha) * ssim

        print(mse, ssim)
    return loss
