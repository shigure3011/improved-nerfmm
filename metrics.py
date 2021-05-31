import torch
import lpips
from skimage.metrics import structural_similarity as ssim


loss_fn = lpips.LPIPS(net='alex')


def mse(source, target):
    value = (source - target)**2
    return torch.mean(value)


def met_psnr(pred, gt):
    return -10*torch.log10(mse(pred, gt))


def met_ssim(pred, gt):
    return ssim(pred, gt)


def met_lpips(pred, gt):
    return loss_fn(pred, gt)
