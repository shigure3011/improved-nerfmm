from utils import find_files

import os
import imageio
import skimage
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale

import torch
from torch.utils.data.dataset import Dataset


class NeRFMMDataset(Dataset):
    def __init__(self, data_dir, downscale=1.):
        super().__init__()
        img_paths = find_files(data_dir)
        imgs = []
        assert len(img_paths) > 0, "no object in the data directory: [{}]".format(data_dir)
        #------------
        # load all imgs into memory
        #------------
        for path in tqdm(img_paths, '=> Loading data...'):
            img = imageio.imread(path)[:, :, :3]
            img = skimage.img_as_float32(img)
            img = rescale(img, 1./downscale, anti_aliasing=True, multichannel=True)
            imgs.append(img)
        self.imgs = imgs
        self.H, self.W, _ = imgs[0].shape
        print("=> dataset: size [{} x {}] for {} images".format(self.H, self.W, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.from_numpy(self.imgs[index]).reshape([-1, 3])
        index = torch.tensor([index]).long()
        return index, img, None, None


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, 'poses_bounds.npy'))  # (N_images, 17)

    c2ws = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = c2ws[0, :, -1]

    # correct c2ws: original c2ws has rotation in form "down right back", change to "right up back".
    # See https://github.com/bmild/nerf/issues/34
    c2ws = np.concatenate([c2ws[..., 1:2], -c2ws[..., :1], c2ws[..., 2:4]], -1)

    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws

    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal,     # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


class ColmapDataset(Dataset):
    def __init__(self, colmap_dir, downscale=1.):
        super().__init__()
        data_dir = os.path.join(colmap_dir, 'images')
        img_paths = find_files(data_dir)
        imgs = []
        assert len(img_paths) > 0, "no object in the data directory: [{}]".format(data_dir)
        
        #------------
        # load all imgs into memory
        #------------
        for path in tqdm(img_paths, '=> Loading data...'):
            img = imageio.imread(path)[:, :, :3]
            img = skimage.img_as_float32(img)
            img = rescale(img, 1./downscale, anti_aliasing=True, multichannel=True)
            imgs.append(img)
        self.imgs = imgs
        self.H, self.W, _ = imgs[0].shape
        print("=> dataset: size [{} x {}] for {} images".format(self.H, self.W, len(self.imgs)))

        #------------
        # load all poses into memory
        #------------
        meta = read_meta(self.scene_dir, True)
        self.c2ws = meta['c2ws']
        self.c2ws = torch.from_numpy(self.c2ws).float()
        self.focal = float(meta['focal'])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = torch.from_numpy(self.imgs[index]).reshape([-1, 3])
        index = torch.tensor([index]).long()
        focal = self.focal
        c2w = self.c2ws[index]
        return index, img, c2w, focal
