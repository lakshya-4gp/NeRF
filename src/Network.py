# %matplotlib inline
# %matplotlib notebook
import os
import sys
import time
import json
import glob
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=1.0):
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField_torch(torch.nn.Module):
    def __init__(self, D=8, skips=[4], n_harmonic_functions_pos=10, n_harmonic_functions_dir=4, n_batches=16, n_hidden_neurons=256, 
                    raw_noise_std = 1.0, activation = torch.nn.ReLU(), set_bias = False):
        super().__init__()
        
        # The harmonic embedding layer converts input 3D coordinates
        self.harmonic_embedding_pos = HarmonicEmbedding(n_harmonic_functions_pos)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        self.skips = skips
        self.n_batches = n_batches
        self.raw_noise_std = raw_noise_std  # Add raw noise to densities before passing them through ReLU
        self.activation_module = activation
        
        # The dimension of the harmonic embedding.
        emb_dim_pos = n_harmonic_functions_pos * 2 * 3
        emb_dim_dir = n_harmonic_functions_dir * 2 * 3
        
        # main_linears are the main backbone
        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(emb_dim_pos, n_hidden_neurons)] + 
                [torch.nn.Linear(n_hidden_neurons, n_hidden_neurons) if i not in self.skips else torch.nn.Linear(n_hidden_neurons + emb_dim_pos, n_hidden_neurons) 
                 for i in range(D-1)])
        
        # A linear layer with no activation, used before getting colors, Shown with orange color in paper
        self.feature_layer = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        
        # Color conversion layers
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + emb_dim_dir, n_hidden_neurons//2),
            self.activation_module,
            torch.nn.Linear(n_hidden_neurons//2, 3),
            torch.nn.Sigmoid(),
        )  
        
        # The density layer converts the features of self.mlp to raw sigma, but noise is added and activation is applied to convert to raw sigma
        self.density_layer = torch.nn.Linear(n_hidden_neurons, 1)
        
        # Pytorch3d mentioned to set this as crucial for convergence. THis allows initial densities to be close to zero
        if set_bias:
            self.density_layer.bias.data[0] = -1.5        
                
    def _get_densities(self, features):
        sigma =  self.density_layer(features)
        # add noise to the sigma if there is some std dev specified, 1 recommended and kept as default
        if self.raw_noise_std > 0.:
            sigma += torch.randn(sigma.shape) * self.raw_noise_std
        return self.activation_module(sigma)

    
    def _get_colors(self, features, rays_directions):
        # Use a linear layer with no activation first
        features = self.feature_layer(features)

        # Normalize the ray_directions to unit l2 norm and get embeddings
        rays_embedding = self.harmonic_embedding_dir(torch.nn.functional.normalize(rays_directions, dim=-1))

        # Concatenate ray direction embeddings with features
        return self.color_layer(torch.cat((features, rays_embedding),dim=-1))
    
    def apply_mlp(self, embeds):
        h = embeds
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.activation_module(h)
            if i in self.skips:
                h = torch.cat([embeds, h], -1)
        return h
    
    def forward(self,ray_bundle):
        """
        Args:
            ray_bundle: [..., 6] with first 3 as ray_points and last 3 as ray direct

        Returns:
            rgb_sigma: [..., 4] with first 3 as sigmoid applied color, and last 1 as self.activation(sigma + noise)
        """

        rays_points_world, directions = torch.split(ray_bundle, [3, 3], dim=-1)
        
        embeds = self.harmonic_embedding_pos(rays_points_world)

        features = self.apply_mlp(embeds)
        
        return torch.cat([self._get_colors(features, directions), self._get_densities(features)], -1)
    
    def batched_forward(self, inputs, chunk = 4096):
        """
            Constructs a forward that applies to smaller batches, only helpful if model in eval mode, else anyway the entire graph is remembered
        """
        return torch.cat([self.forward(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)





#=====================================================================================================================================
# This class has inputs similar to pytorch3d renderer expected
class NeuralRadianceField_Pytorch3d(torch.nn.Module):
    def __init__(self, D=8, skips=[4], n_harmonic_functions_pos=10, n_harmonic_functions_dir=4, n_batches=16, n_hidden_neurons=256, activation = torch.nn.Softplus(beta=10), set_bias = True):
        super().__init__()
        
        # The harmonic embedding layer converts input 3D coordinates
        self.harmonic_embedding_pos = HarmonicEmbedding(n_harmonic_functions_pos)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        self.skips = skips
        self.n_batches = n_batches

        self.activation_module = activation
        
        # The dimension of the harmonic embedding.
        emb_dim_pos = n_harmonic_functions_pos * 2 * 3
        emb_dim_dir = n_harmonic_functions_dir * 2 * 3
        
        # main_linears are the main backbone
        self.main_linears = torch.nn.ModuleList(
            [torch.nn.Linear(emb_dim_pos, n_hidden_neurons)] + 
                [torch.nn.Linear(n_hidden_neurons, n_hidden_neurons) if i not in self.skips else torch.nn.Linear(n_hidden_neurons + emb_dim_pos, n_hidden_neurons) 
                 for i in range(D-1)])
        
        # A linear layer with no activation, used before getting colors
        # Shown with orange color in paper
        self.feature_layer = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        
        # Color conversion layers
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + emb_dim_dir, n_hidden_neurons//2),
            self.activation_module,
            torch.nn.Linear(n_hidden_neurons//2, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )  
        
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            self.activation_module,
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )
        
        # Pytorch3d mentioned to set this as crucial for convergence. THis allows initial densities to be close to zero
        if set_bias:
            self.density_layer[0].bias.data[0] = -1.5        
                
    def _get_densities(self, features):
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()
    
    def _get_colors(self, features, rays_directions):
        # Use a linear layer with no activation first
        features = self.feature_layer(features)
        
        spatial_size = features.shape[:-1]
        
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )
        
        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding_dir(
            rays_directions_normed
        )
        
        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )
        
        # Concatenate ray direction embeddings with 
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)
    
    def apply_mlp(self, embeds):
        h = embeds
        for i, l in enumerate(self.main_linears):
            h = self.main_linears[i](h)
            h = self.activation_module(h)
            if i in self.skips:
                h = torch.cat([embeds, h], -1)
        return h
    
  
    def forward(
        self, 
        ray_bundle: RayBundle,
        **kwargs,
    ):
        """
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins
                directions:
                lengths:

        Returns:
            rays_densities: [..., num_points_per_ray, 1]
            rays_colors: [minibatch, ..., num_points_per_ray, 3]
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]
        
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding_pos(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]
        features = self.apply_mlp(embeds)
        
        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        
        return rays_densities, rays_colors
    
    def batched_forward(
        self, 
        ray_bundle: RayBundle,
        **kwargs,        
    ):
        """
        Batched version of forward, only to be used for visualization, won't help with reducing memory for training as the graph has to be kept in memory
        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), self.n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]
        
        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors