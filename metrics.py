import torch
from kornia.losses import ssim as dssim


def mse(source, target):
    value = (source - target)**2
    return torch.mean(value)


def met_psnr(pred, gt):
    return -10*torch.log10(mse(pred, gt))


def met_ssim(pred, gt, reduction='mean'):
    image_pred = pred.unsqueeze(0)
    image_gt = gt.unsqueeze(0)
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]
