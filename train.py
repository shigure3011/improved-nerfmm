import utils
from metrics import *
from logger import Logger
from checkpoints import CheckpointIO
from dataio.dataset import NeRFMMDataset, ColmapDataset, get_ndc_rays
from models.frameworks import create_model
from models.volume_rendering import volume_render
from models.perceptual_model import get_perceptual_loss
from models.cam_params import CamParams, get_rays, plot_cam_rot, plot_cam_trans, get_camera2world

import os
import time
import copy
import functools
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict
import lpips as lpips_lib

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def mse_loss(source, target):
    value = (source - target)**2
    return torch.mean(value)


def visualize_depth(x):
    """
    depth: (H, W)
    """
    x = torch.nan_to_num(x) # change nan to 0
    mi = x.min() # get minimum depth
    ma = x.max()
    x = 1 - (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    return x


class NeRFMinusMinusTrainer(nn.Module):
    def __init__(
            self,
            model,
            perceptual_net: Optional[nn.Module] = None):
        super().__init__()

        # necessary to duplicate weights correctly across gpus. hacky workaround
        self.model = model
        if perceptual_net is not None:
            self.perceptual_net = perceptual_net


    def forward(self,
                args,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                target_s: torch.Tensor,
                render_kwargs_train: dict,
                H=None, W=None):

        render_kwargs_train["network_fn"] = self.model.get_coarse_fn()
        render_kwargs_train["network_fine"] = self.model.get_fine_fn()

        losses = OrderedDict()

        rgb, depth, extras = volume_render(
            rays_o=rays_o,
            rays_d=rays_d,
            detailed_output=True,
            **render_kwargs_train)

        # reconstruction loss
        # could be rendered as an mse image
        losses['rgb/fine'] = mse_loss(rgb, target_s)
        losses['rgb/fine'] *= args.training.w_img

        if args.model.N_importance > 0 and args.model.use_fine_model:
            losses['rgb/coarse'] = mse_loss(extras['rgb0'], target_s)

        loss = 0
        for v in losses.values():
            loss += v
        losses['total'] = loss

        with torch.no_grad():
            losses['train/psnr'] = met_psnr(rgb, target_s)

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)]
        )


def get_parallelized_training_function(
        device_ids=None,
        wrapper_cls=NeRFMinusMinusTrainer,
        **kwargs):

    if device_ids is None or len(device_ids) == 1:
        # for single gpus, no parallel training.
        return wrapper_cls(**kwargs)
    else:
        return nn.DataParallel(
            wrapper_cls(**kwargs),
            device_ids=device_ids
        )


def main_function(args):

    device_ids = args.device_ids
    # for nn.DataParallel, the model and original input must be on device_ids[0] device
    device = "cuda:{}".format(device_ids[0])

    exp_dir = args.training.exp_dir

    print("=> Experiments dir: {}".format(exp_dir))

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring='tensorboard',
        monitoring_dir=os.path.join(exp_dir, 'events'))

    # backup codes
    utils.backup(os.path.join(exp_dir, 'backup'))

    # save configs
    utils.save_config(args.to_dict(), os.path.join(exp_dir, 'config.yaml'))

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'))

    # datasets: just pure images.
    if not args.data.colmap:
        dataset = NeRFMMDataset(args.data.data_dir, downscale=args.data.downscale)
    else:
        dataset = ColmapDataset(args.data.data_dir, downscale=args.data.downscale)
    dataloader = DataLoader(dataset, 
        batch_size=args.data.get('batch_size', None), 
        shuffle=True)
    valloader = copy.deepcopy(dataloader)

    # Camera parameters to optimize
    so3_representation = args.model.so3_representation

    cam_param = CamParams.from_config(
        num_imgs=len(dataset), 
        H0=dataset.H, W0=dataset.W, 
        so3_repr=so3_representation,
        intr_repr=args.model.intrinsics_representation,
        initial_fov=args.model.initial_fov)


    # Create nerf model
    model, render_kwargs_train, render_kwargs_test, grad_vars = create_model(
        args, model_type=args.model.framework)
        

    # move models to GPU
    model.to(device)
    cam_param.to(device)
    
    print(model)
    print("=> Nerf params: ", utils.count_trainable_parameters(model))

    def build_optimizer():
        # Create optimizer
        optimizer_nerf = optim.Adam(
            params=grad_vars, lr=args.training.lr_nerf, betas=(0.9, 0.999)
        )
        optimizer_intr = optim.Adam(
            params=[cam_param.f], lr=args.training.lr_param, betas=(0.9, 0.999)
        )
        optimizer_extr = optim.Adam(
            params=[cam_param.phi, cam_param.t], lr=args.training.lr_param,  betas=(0.9, 0.999)
        )
        return optimizer_nerf, optimizer_intr, optimizer_extr


    def build_lr_scheduler(op_nerf, op_intr, op_extr, last_epoch=-1):
        lr_scheduler_nerf = optim.lr_scheduler.StepLR(
            op_nerf,
            step_size=args.training.step_size_nerf,
            gamma=args.training.lr_anneal_nerf,
            last_epoch=last_epoch
        )
        lr_scheduler_intr = optim.lr_scheduler.StepLR(
            op_intr,
            step_size=args.training.step_size_param,
            gamma=args.training.lr_anneal_param
        )
        lr_scheduler_extr = optim.lr_scheduler.StepLR(
            op_extr,
            step_size=args.training.step_size_param,
            gamma=args.training.lr_anneal_param
        )
        return lr_scheduler_nerf, lr_scheduler_intr, lr_scheduler_extr

    optimizer_nerf, optimizer_intr, optimizer_extr = build_optimizer()

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer_nerf=optimizer_nerf,
        # optimizer_param=optimizer_param,
        optimizer_intr=optimizer_intr,
        optimizer_extr=optimizer_extr,
        cam_param=cam_param
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys)
    logger.load_stats('stats.p')    # this will be used for plotting
    it = load_dict.get('global_step', -1)
    epoch_idx = load_dict.get('epoch_idx', 0)

    lr_scheduler_nerf, lr_scheduler_intr, lr_scheduler_extr = build_lr_scheduler(
        optimizer_nerf, optimizer_intr, optimizer_extr, last_epoch=epoch_idx-1)

    # perceptual net model
    perceptual_net = None
    if args.training.w_perceptual > 0:
        from models.perceptual_model import CLIP_for_Perceptual
        perceptual_net = CLIP_for_Perceptual()
        # from models.perceptual_model import VGG16_for_Perceptual
        # perceptual_net = VGG16_for_Perceptual()

    # Training loop
    trainer = get_parallelized_training_function(
        model=model,
        perceptual_net=perceptual_net,
        device_ids=device_ids,
    )

    def do_nvs(ep):
        if ep <= 200:
            return ep % 5 == 0
        elif ep <= 2000:
            return ep % 100 == 0
        else:
            return ep % 500 == 0

    def do_val(ep):
        if ep <= 200:
            return (ep + 1) % 20 == 0
        else:
            return (ep + 1) % 50 == 0

    def do_eval(ep):
        return (ep + 1) % 500 == 0


    def train(num_ep, stage='train', ep_offset=0): # choose stage between [pre, refine]
        nonlocal epoch_idx  # global epoch_idx
        nonlocal it
        if stage == 'pre':
            stage_desc = "pre-train stage"
        elif stage == 'train':
            stage_desc = "train stage"
        else:
            raise RuntimeError("wrong stage")
        tstart = t0 = time.time()
        valid_every_nth = int(10) #Valid set
        
        #############################################################
        if stage == 'pre':
            render_kwargs_train['N_samples'] = render_kwargs_test['N_samples'] = 128
            render_kwargs_train['N_importance'] = render_kwargs_test['N_importance'] = 0  
        else:
            render_kwargs_train['N_samples'] = render_kwargs_test['N_samples'] = 64
            render_kwargs_train['N_importance'] = render_kwargs_test['N_importance'] = 128

        args.model.N_importance = render_kwargs_train['N_importance']
        #############################################################

        with tqdm(range(num_ep), desc=stage_desc) as pbar:
            pbar.update(epoch_idx - ep_offset)
            while epoch_idx - ep_offset < num_ep:
                flags = False
                local_epoch_idx = epoch_idx  - ep_offset
                pbar.update()
                # print('Start epoch {}'.format(local_epoch_idx))
                # with tqdm(dataloader) as pbar:
                for ind, img, c2w, focal, bound in dataloader:
                    fx = fy = focal

                    if stage == 'train' and ind % valid_every_nth == 0:
                        continue

                    t_it = time.time()
                    it += 1
                    pbar.set_postfix(it=it, ep=epoch_idx)                  

                    if not args.data.colmap:
                        if stage == 'pre':
                            R, t, fx, fy = cam_param(ind.to(device).squeeze(-1))
                            c2w = get_camera2world(R, t, cam_param.so3_repr)
                        else:
                            fx, fy = cam_param.get_focal()
                            c2w = cam_param.get_camera2world(ind.to(device).squeeze(-1))

                    rays_o, rays_d, select_inds = get_rays(
                        device,
                        c2w, fx, fy, dataset.H, dataset.W,
                        args.data.N_rays)
                    rays_o, rays_d = get_ndc_rays(dataset.H, dataset.W, 
                                                  fx, fy, 1.0, rays_o, rays_d)

                    # [(B,) N_rays, 3]
                    target_rgb = torch.gather(img.to(device), -2, torch.stack(3*[select_inds],-1)) 

                    ret = trainer(
                        args,
                        rays_o=rays_o,
                        rays_d=rays_d,
                        target_s=target_rgb,
                        render_kwargs_train=render_kwargs_train
                    )
                    losses = ret['losses']
                    extras = ret['extras']

                    for k, v in losses.items():
                        # print("{}:{} - > {}".format(k, v.shape, v.mean().shape))
                        losses[k] = torch.mean(v)

                    if not flags and do_val(local_epoch_idx):
                        print('Train losses: ', losses['total'].item())
                        flags = True
                        
                    optimizer_nerf.zero_grad()
                    # optimizer_param.zero_grad()
                    optimizer_intr.zero_grad()
                    optimizer_extr.zero_grad()
                    losses['total'].backward()
                    optimizer_nerf.step()
                    # optimizer_param.step()
                    optimizer_intr.step()
                    optimizer_extr.step()

                    #-------------------
                    # logging
                    #-------------------
                    # log lr
                    logger.add('learning_rates', 'nerf', optimizer_nerf.param_groups[0]['lr'], it=it)
                    # logger.add('learning_rates', 'camera parameters', optimizer_param.param_groups[0]['lr'], it=it)
                    logger.add('learning_rates', 'camera intrinsics', optimizer_intr.param_groups[0]['lr'], it=it)
                    logger.add('learning_rates', 'camera extrinsics', optimizer_extr.param_groups[0]['lr'], it=it)

                    # log losses
                    for k, v in losses.items():
                        logger.add('losses', k, v.data.cpu().numpy().item(), it)

                    # log extras
                    # for k, v in extras.items():
                    names = ["rgb", "sigma"]
                    for n in names:
                        p = "whole"
                        key = "raw.{}".format(n)
                        logger.add("extras_{}".format(n), "{}.mean".format(
                            p), extras[key].mean().data.cpu().numpy().item(), it)
                        logger.add("extras_{}".format(n), "{}.min".format(
                            p), extras[key].min().data.cpu().numpy().item(), it)
                        logger.add("extras_{}".format(n), "{}.max".format(
                            p), extras[key].max().data.cpu().numpy().item(), it)
                        logger.add("extras_{}".format(n), "{}.norm".format(
                            p), extras[key].norm().data.cpu().numpy().item(), it)

                    if args.training.i_backup > 0 and it % args.training.i_backup == 0 and it > 0:
                        # print("Saving backup...")
                        checkpoint_io.save(
                            filename='{:08d}.pt'.format(it),
                            global_step=it, epoch_idx=epoch_idx)

                    if it == 0 or (args.training.i_save > 0 and time.time() - t0 > args.training.i_save):
                        print('Saving checkpoint...')
                        checkpoint_io.save(
                            filename='latest.pt'.format(it),
                            global_step=it, epoch_idx=epoch_idx)
                        # this will be used for plotting
                        logger.save_stats('stats.p')
                        t0 = time.time()
                
                if (stage == 'pre' and num_ep == epoch_idx + 1):
                    print('Saving checkpoint...')
                    checkpoint_io.save(
                        filename='latest.pt'.format(it),
                        global_step=it + 1, epoch_idx=epoch_idx)
                    logger.save_stats('stats.p')

                #----------------
                # things to do each epoch
                #----------------
                lr_scheduler_nerf.step()
                # lr_scheduler_param.step()
                lr_scheduler_intr.step()
                lr_scheduler_extr.step()

                #-------------------
                # plot camera parameters
                #-------------------
                save_output_img = do_nvs(local_epoch_idx) or do_val(local_epoch_idx)
                
                logger.add_figure(plot_cam_trans(cam_param), 
                    "camera/extr translation on xy", it, save_img=save_output_img)
                logger.add_figure(plot_cam_trans(cam_param, about = 'yz'), 
                    "camera/extr translation on yz", it, save_img=save_output_img)
                logger.add_figure(plot_cam_trans(cam_param, about = 'xz'), 
                    "camera/extr translation on xz", it, save_img=save_output_img)
                

                # log camera parameters
                logger.add_vector('camera', 'extr_phi', cam_param.phi.data, it)
                logger.add_vector('camera', 'extr_t', cam_param.t.data, it)
                fx, fy = cam_param.get_focal()
                logger.add('camera', 'fx', fx.item(), it)
                logger.add('camera', 'fy', fy.item(), it)


                #-------------------
                # eval with gt
                #-------------------
                if do_val(local_epoch_idx):
                    with torch.no_grad():
                        psnr = []
                        for ind, img, c2w, focal, bound in dataloader:
                            if ind % valid_every_nth != 0:
                                continue

                            fx = fy = focal
                            if not args.data.colmap:
                                if stage == 'pre':
                                    R, t, fx, fy = cam_param(ind.to(device).squeeze(-1))
                                    c2w = get_camera2world(R, t, cam_param.so3_repr)
                                else:
                                    fx, fy = cam_param.get_focal()
                                    c2w = cam_param.get_camera2world(ind.to(device).squeeze(-1))

                            rays_o, rays_d, select_inds = get_rays(
                                device,
                                c2w, fx, fy, dataset.H, dataset.W,
                                -1)
                            rays_o, rays_d = get_ndc_rays(dataset.H, dataset.W, 
                                                          fx, fy, 1.0, rays_o, rays_d)

                            # [N_rays, 3]
                            target_rgb = img.to(device)

                            val_rgb, val_depth, val_extras = volume_render(
                                rays_o=rays_o,
                                rays_d=rays_d,
                                detailed_output=True,   # to return acc map and disp map
                                **render_kwargs_test)
                            
                            to_img = functools.partial(
                                utils.lin2img, H=dataset.H, W=dataset.W, batched=render_kwargs_test['batched'])
                            
                            pred = to_img(val_rgb)
                            gt = to_img(target_rgb)
                            psnr.append(met_psnr(pred, gt))
                    
                    logger.add('val', 'psnr', sum(psnr) / len(psnr), it=it)
                    print('Val psnr: {0}'.format(sum(psnr) / len(psnr)))
                    logger.add_imgs(to_img(val_rgb), 'val/pred', it)
                    logger.add_imgs(to_img(target_rgb), 'val/gt', it)
                    logger.add_imgs(to_img(val_extras['disp_map'].unsqueeze(-1)), 'val/pred_disp', it)
                    logger.add_imgs(visualize_depth(to_img(val_depth.unsqueeze(-1))), 'val/pred_depth', it)
                    logger.add_imgs(to_img(val_extras['acc_map'].unsqueeze(-1)), 'val/pred_acc', it)


                if do_eval(local_epoch_idx):
                    with torch.no_grad():
                        lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(device)
                        
                        psnr = []
                        ssim = []
                        lpips = []                
        
                        for ind, img, c2w, focal, bound in dataloader:
                            if ind % valid_every_nth != 0:
                                continue

                            fx = fy = focal
                            if not args.data.colmap:
                                if stage == 'pre':
                                    R, t, fx, fy = cam_param(ind.to(device).squeeze(-1))
                                    c2w = get_camera2world(R, t, cam_param.so3_repr)
                                else:
                                    fx, fy = cam_param.get_focal()
                                    c2w = cam_param.get_camera2world(ind.to(device).squeeze(-1))

                            rays_o, rays_d, select_inds = get_rays(
                                device,
                                c2w, fx, fy, dataset.H, dataset.W,
                                -1)
                            rays_o, rays_d = get_ndc_rays(dataset.H, dataset.W, 
                                                          fx, fy, 1.0, rays_o, rays_d)

                            # [N_rays, 3]
                            target_rgb = img.to(device)

                            val_rgb, val_depth, val_extras = volume_render(
                                rays_o=rays_o,
                                rays_d=rays_d,
                                detailed_output=True,   # to return acc map and disp map
                                **render_kwargs_test)
                            
                            to_img = functools.partial(
                                utils.lin2img, H=dataset.H, W=dataset.W, batched=render_kwargs_test['batched'])
                            
                            pred = to_img(val_rgb)
                            gt = to_img(target_rgb)
                            psnr.append(met_psnr(pred, gt).item())
                            ssim.append(met_ssim(pred, gt).item())
                            lpips.append(lpips_vgg_fn(
                                pred.unsqueeze(0),
                                gt.unsqueeze(0),
                                normalize = True).item())
                    
                    logger.add('test', 'psnr', sum(psnr) / len(psnr), it=it)
                    logger.add('test', 'ssim', sum(ssim) / len(ssim), it=it)
                    logger.add('test', 'lpips', sum(lpips) / len(lpips), it=it)
                    
                #-------------------
                # novel view synthesis
                #-------------------
                if stage != 'pre' and args.training.get('novel_view_synthesis', False) and do_nvs(local_epoch_idx):
                    with torch.no_grad():
                        # average camera extrinsics
                        R = cam_param.phi.data.clone().mean(0)
                        t = cam_param.t.data.clone().mean(0)

                        fx, fy = cam_param.get_focal()
                        fx, fy = fx.data.clone(), fy.data.clone()

                        # [N_rays, 3], [N_rays, 3], [N_rays]
                        # when logging val images, scale the resolution to be 1/16 just to save time.
                        rays_o, rays_d, select_inds = get_rays(
                            R, t, fx, fy, dataset.H, dataset.W, -1,
                            representation=so3_representation)

                        # [N_rays, 3]
                        target_rgb = img.to(device)

                        val_rgb, val_depth, val_extras = volume_render(
                            rays_o=rays_o,
                            rays_d=rays_d,
                            detailed_output=False,  # only return rgb and depth
                            **render_kwargs_test)

                    to_img = functools.partial(
                        utils.lin2img, H=dataset.H, W=dataset.W, batched=render_kwargs_test['batched'])

                    logger.add_imgs(to_img(val_rgb), "novel_view/rgb", it)
                    logger.add_imgs(to_img(val_depth.unsqueeze(-1)), "novel_view/depth", it)
                
                #------------
                # update epoch index
                #------------
                epoch_idx += 1

    if not args.data.colmap:
        num_epoch_pre = args.training.get('num_epoch_pre', 0)
    else:
        num_epoch_pre = 0
    
    if num_epoch_pre > 0:
        if epoch_idx < num_epoch_pre:
            #-------------
            # Pre-training stage: will just use cam_param
            #-------------
            print('Start pre-training..., ep={}, in {}'.format(epoch_idx, exp_dir))
            train(num_epoch_pre, 'pre', ep_offset=0)

            #-------------
            # drop all models with only cam_param left
            #-------------
            optimizer_nerf, optimizer_intr, optimizer_extr = build_optimizer()
            lr_scheduler_nerf, lr_scheduler_intr, lr_scheduler_extr = build_lr_scheduler(
                optimizer_nerf, optimizer_intr, optimizer_extr, last_epoch=epoch_idx-num_epoch_pre-1)
            def weight_reset(m):
                reset_parameters = getattr(m, "reset_parameters", None)
                if callable(reset_parameters):
                    m.reset_parameters()
            model.apply(weight_reset)   # recursively: from children to root.
        
        print("Start refinement... ep={}, in {}".format(epoch_idx, exp_dir))
    else:
        print("Start training... ep={}, in {}".format(epoch_idx, exp_dir))

    # freeze all camera parameters
    for param in cam_param.parameters():
        param.requires_grad = False

    cam_param.process_poses()

    train(args.training.num_epoch, 'train', ep_offset=num_epoch_pre)

    final_ckpt = 'final_{:08d}.pt'.format(it)
    print('Saving final to {}'.format(final_ckpt))
    checkpoint_io.save(
        filename=final_ckpt,
        global_step=it, epoch_idx=epoch_idx)
    # this will be used for plotting
    logger.save_stats('stats.p')

if __name__ == "__main__":
    # Arguments
    parser = utils.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = utils.load_config(args, unknown)
    main_function(config)
