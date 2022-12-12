import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import mlflow
import pytorch3d
from PIL import Image
from tqdm import tqdm
from src.get_bottles import get_data, MyCam
from src.Network import NeuralRadianceField_torch
from src.ray_utils import get_origin_direc

# obtain the utilized device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    print(
        'CPU being used, check once!'
    )
    device = torch.device("cpu")

# All tensors by default will be created as Float and on cuda device, most of these tensors will be created in render only
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Return all the rays for these files, whether shuffled or not
def get_rays_all_files(cameras, images, shuffle = True):
    H, W, K, poses = cameras.get()
    rays = np.stack([get_origin_direc(H, W, K, p) for p in poses], 0) # [N, 2, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, 3, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, 3, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [N*H*W, 3, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    if shuffle:
        np.random.shuffle(rays_rgb)
    return rays_rgb



if __name__ == '__main__':
    ## Args
    args = {}
    args['volume_extent_world'] = 5.0
    args['volume_extent_min'] = 0.1
    render_size_factor = 1

    viz_freq = 100
    viz_train = False
    val_freq = 100

    args['dataset'] = 'bottles' #[bottles, cow]
    args['resize_ratio'] = 100/800
    args['train_size'] = 50
    args['val_size'] = 5
    args['white_bkg'] = True

    args['learning_rate'] = 5e-4
    args['iteration_for_lr_decay'] = 100000
    args['num_iterations'] = 30000
    args['accum_iter'] = 1
    # args['num_iterations'] after counting gradient accumulation
    args['num_iterations'] *= args['accum_iter']

    args['train_n_rays_per_image'] = 4096 // args['train_size']
    args['val_n_rays_per_image'] = 4096 // args['val_size']
    args['n_pts_per_ray'] = 128
    args['stratified_sampling'] = True

    args['num_layers'] = 4
    skip_attach_after = 3
    args['skips']=[skip_attach_after-1] if skip_attach_after is not None else []

    model_save_path = f"models/{args['dataset']}_{args['num_iterations']}_{args['resize_ratio']}_{args['train_size']}_{args['num_layers']}_{args['accum_iter']}.tar"
    args['n_harmonic_functions_pos'] = 10
    args['n_harmonic_functions_dir'] = 4
    n_harmonic_functions = 10

    log_mlflow = False
    if log_mlflow:
        mlflow.set_experiment('Lower size')
        mlflow.start_run('Base')

        mlflow.log_params(args)

    ## Get Data
    def split(tensor, tsize):
        return tensor[:-tsize], tensor[-tsize:]

    def split_cameras(cameras, tsize):
        args = cameras.get()
        c2ws_split = split(args[-1],tsize)
        return MyCam(*args[:-1], c2ws_split[0]), MyCam(*args[:-1], c2ws_split[1])

    files, cameras, images, silhouettes = get_data(train_not_test = True, num_files = args['train_size'] + args['val_size'], resize_ratio = args['resize_ratio'])

    train_files, val_files = split(files, args['val_size'])
    train_images, val_images = split(images, args['val_size'])
    train_sil, val_sil = split(silhouettes, args['val_size'])
    train_cameras, val_cameras = split_cameras(cameras, args['val_size'])

    # Get rays: origin and direction. Val_img_rgb_rays will be used to plot full size image
    train_rgb_rays = get_rays_all_files(train_cameras, train_images, shuffle = True)
    val_rgb_rays = get_rays_all_files(val_cameras, val_images, shuffle = True)
    val_img_rgb_rays = get_rays_all_files(train_cameras, train_images, shuffle = False)


train_rays_rgb = get_rays_all_files(train_cameras, train_images, shuffle = True) # [N*H*W, 3(ro+rd+rgb), 3]
val_rays_rgb = get_rays_all_files(val_cameras, val_images, shuffle = True)
val_img_rays_rgb = get_rays_all_files(train_cameras, train_images, shuffle = False) # Unshuffled, will be used for generating full size render


## Torch land start
train_rays_rgb = torch.Tensor(train_rays_rgb).to(device)
val_rays_rgb = torch.Tensor(val_rays_rgb).to(device)
val_img_rays_rgb = torch.Tensor(val_img_rays_rgb).to(device)


neural_radiance_field = NeuralRadianceField_torch(D = args['num_layers'],
                                            skips = args['skips'],
                                            n_harmonic_functions_pos = args['n_harmonic_functions_pos'],
                                           n_harmonic_functions_dir = args['n_harmonic_functions_dir']).to(device)
rays_rgb = train_rays_rgb
i_batch = 0
batch = rays_rgb[i_batch:i_batch+args['n_rays_per_batch']] # [B, 2+1, 3]
batch = torch.transpose(batch, 0, 1)	# [2+1, B, 3]
batch_rays, target_s = batch[:2], batch[2]

i_batch += args['n_rays_per_batch']
if i_batch >= rays_rgb.shape[0]:
	# fix this as rays_rgb is just a var
	# print("Shuffle data after an epoch as in implementation!")
	# rand_idx = torch.randperm(rays_rgb.shape[0])
	# rays_rgb = rays_rgb[rand_idx]
	i_batch = 0



def get_pts_from_ray_batch(rays_o, rays_d, near, far, npts_per_ray, stratified):
    t_vals = torch.linspace(0., 1., steps=npts_per_ray)

    # camera domain negative z vals, but you can simply think of them as depth from camera
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([rays_o.shape[0], npts_per_ray])

    # if stratified pickup samples b/w [0th point, mid of 0&1, mid of 1,2, ....., mid of n-2,n-1, n-1] to get npts
    if stratified > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])  # [N_rays, npts - 1]    mid of each interval
        upper = torch.cat([mids, z_vals[...,-1:]], -1)  # [Nrays, npts]     mids + n-1th point
        lower = torch.cat([z_vals[...,:1], mids], -1)   # [Npts, npts]      0th point + mids
        
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    # Notice rays_d is used which are unnormalized, with this a rectangular front moves along z else we would get a spherical front.
    return rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

def ray_march(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# Render works in torch land, i.e. everything is in torch
def render(ray_batch, nerf_fun):
    # Create needed variables
    rays_o, rays_d = ray_batch
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # [N_rays, 3]
    near, far = args['znear'] * torch.ones_like(rays_d[...,:1]), args['zfar'] * torch.ones_like(rays_d[...,:1])

    pts = get_pts_from_ray_batch(rays_o, rays_d, near, far, args['n_pts_per_ray'], args['stratified_sampling']) # [N_rays, N_samples, 3]
    
    # Expand viewdirs so that we can concatenate and send them. 
    # We will anyway have to expand while concatenating with the features in the model
    viewdirs = viewdirs[:,None].expand(pts.shape)   # [N_rays, N_samples, 3]
    
    c_sigma  = nerf_fun(torch.cat([pts, viewdirs], -1))
    return pts

pts = render(batch_rays, neural_radiance_field)