'''
References: https://pytorch3d.org/tutorials/fit_simple_neural_radiance_field
https://github.com/yenchenlin/nerf-pytorch
'''

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
import mlflow
import pytorch3d
from PIL import Image
from tqdm import tqdm

def get_origin_direc(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # Basically, focal length == 1 unit of z direction.
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Notice how translation wasn't added as it gets cancelled to make them vectors
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product
    # Translation in c2w is the location of camera in world frame and is the origin of each ray
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

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

def get_pts_from_ray_batch(rays_o, rays_d, near, far, npts_per_ray, stratified):
    t_vals = torch.linspace(0., 1., steps=npts_per_ray)

    # camera domain negative z vals, you can simply think of them as depth from camera
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
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    return pts, z_vals

# Hierarchical sampling (section 5.2)
def hier_sample(z_vals, weights, N_samples, epsilon=1e-5):
    bins = .5 * (z_vals[...,1:] + z_vals[...,:-1])    # [N_rays, N_samples-1]
    weights = weights[...,1:-1] # [N_rays, N_samples-2]

    '''
    bins: znear     |       |       |       |       |   zfar
    wts:                *       *       *       *   
    '''

    # Get pdf
    weights = weights + epsilon # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # [N_rays, N_samples-1]

    '''
    bins: znear     |       |       |       |       |   zfar
    cdf:            0       *       *       *       *(==1)   
    '''

    # Samples from uniform dist to inverse the results.
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inv_ind = torch.searchsorted(cdf, u, right=True)

    # lower index for a u and upper index for a u
    lower = torch.max(torch.zeros_like(inv_ind-1), inv_ind-1)
    upper = torch.min((cdf.shape[-1]-1) * torch.ones_like(inv_ind), inv_ind)
    ind = torch.stack([lower, upper], -1)  # (batch, N_samples, 2)

    # Get the cdf and bins
    shape = [ind.shape[0], ind.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(shape), 2, ind)
    bins_g = torch.gather(bins.unsqueeze(1).expand(shape), 2, ind)

    # Get the samples
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def ray_march(c_sigma, z_vals, rays_d, white_bkgd = True):
    rgb, sigma = c_sigma[...,:3], c_sigma[...,3]

    # use zvals to calc distances along the ray
    dists = z_vals[...,1:] - z_vals[...,:-1] # [N_rays, N_samples-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples] by appending 1e10 i.e. infinite distance thus fully opaque
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # alpha for alpha compositing we will do next
    alpha = 1. - torch.exp(-sigma*dists) # [N_rays, N_samples]

    # alpha compositing weights, alpha * prod(alpha_j), j in [1, i-1]
    # weights: [N_rays, N_samples]. weigths to be used for heirarchichal sampling
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3], Cr

    #depth_map: [N_rays]. Depth map
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, weights, depth_map

