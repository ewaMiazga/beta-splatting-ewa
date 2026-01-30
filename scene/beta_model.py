#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pyexpat import features
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, apply_depth_colormap
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.compress_utils import compress_png, decompress_png, sort_param_dict
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat.rendering import rasterization
import json
import time
from .beta_viewer import BetaRenderTabState

from utils.spherical_utils import nasg


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class BetaModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def sb_params_activation(sb_params):
            softplus_sb_params = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
            sb_params = torch.cat([softplus_sb_params, sb_params[..., 3:]], dim=-1)
            return sb_params

        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.sb_params_activation = sb_params_activation
        self.beta_activation = beta_activation


        if not self.use_beta:
            self.position_size = 3
            self.shape_size = 2
            self.color_size = 3
            self.params_size = 8  # total params per degree: position + shape + weight

            def pos_activation(x):
                return torch.tanh(x)

            def weight_activation(x):
                return torch.tanh(x)

            self.pos_activation = lambda x: torch.tanh(x)
            self.weight_activation = lambda x: torch.tanh(x)

    def __init__(self, sh_degree: int = 0, sb_number: int = 2, use_beta: bool = False, use_gmm_colors: bool = False, use_gmm_colors_cuda: bool = False):
        self.use_beta = use_beta
        self.use_gmm_colors = use_gmm_colors
        self.use_gmm_colors_cuda = use_gmm_colors_cuda
        self.active_sh_degree = 0
        if self.use_beta:
            self.max_sh_degree = sh_degree
            self.sb_number = sb_number
        else:
            self.max_sh_degree = sb_number
        
        self._xyz = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self._sb_params = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        self.setup_functions()

    def capture(self):
        if self.use_beta:
            return (
                self.active_sh_degree,
                self._xyz,
                self._sh0,
                self._shN,
                self._sb_params,
                self._scaling,
                self._rotation,
                self._opacity,
                self._beta,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._sh0,
                self._shN,
                self._sb_params,
                self._features_dc,
                self._features_pos,
                self._features_shape,
                self._features_weight,
                self._scaling,
                self._rotation,
                self._opacity,
                self._beta,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )

    def restore(self, model_args, training_args):
        if self.use_beta:
            (
                self.active_sh_degree,
                self._xyz,
                self._sh0,
                self._shN,
                self._sb_params,
                self._scaling,
                self._rotation,
                self._opacity,
                self._beta,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        else: 
            (
                self.active_sh_degree,
                self._xyz,
                self._sh0,
                self._shN,
                self._sb_params,
                self._features_dc,
                self._features_pos,
                self._features_shape,
                self._features_weight,
                self._scaling,
                self._rotation,
                self._opacity,
                self._beta,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_shs(self):
        sh0 = self._sh0
        shN = self._shN
        return torch.cat((sh0, shN), dim=1)

    @property
    def get_sb_params(self):
        return self.sb_params_activation(self._sb_params)

    @property
    def get_features_pos(self):
        return self.pos_activation(self._features_pos)

    @property
    def get_features_weight(self):
        return self.weight_activation(self._features_weight)

    @property
    def get_features(self):
        sh0 = self._sh0
        
        if self.use_gmm_colors:
            # Linear layout: [sh0, all_pos, all_shape, all_weight]
            rest = torch.cat((
                self.get_features_pos, 
                self._features_shape, 
                self.get_features_weight
            ), dim=1)
            return torch.cat((sh0, rest), dim=1)
        
        elif self.use_gmm_colors_cuda:
            # Interleaved layout: [pos, shape, weight] * lobes + base_color
            pos = self.get_features_pos.reshape(-1, self.max_sh_degree, self.position_size)
            shape = self._features_shape.reshape(-1, self.max_sh_degree, self.shape_size)
            weight = self.get_features_weight.reshape(-1, self.max_sh_degree, self.color_size)
            
            # Interleave: [pos, shape, weight] for each lobe
            interleaved = torch.cat([pos, shape, weight], dim=2)  # [N, max_sh_degree, params_size]
            flattened = interleaved.reshape(-1, 1, self.max_sh_degree * self.params_size)# [N, 1, max_sh_degree * params_size]

            return torch.cat((sh0.reshape(-1, 1, 3), flattened), dim=2)  # [N, 1, total_params_per_point]

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # self.use_beta = True
        # self.max_sh_degree = 0
        # self.sb_number = 2

        print("Use beta:", self.use_beta)
        print("Use GMM colors:", self.use_gmm_colors)
        print("Use GMM colors CUDA:", self.use_gmm_colors_cuda)
        if self.use_beta:
            print("max_sh_degree:", self.max_sh_degree)
            print("sb_number:", self.sb_number)
        else:
            print("max_sh_degree:", self.max_sh_degree)
        #print("sb_number:", self.sb_number)

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        if self.use_beta:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            shs = (
                torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
                .float()
                .cuda()
            )
            shs[:, :3, 0] = fused_color
            shs[:, 3:, 1:] = 0.0

            # [r, g, b, theta, phi, beta]
            sb_params = torch.zeros(
                (fused_point_cloud.shape[0], self.sb_number, 6), device="cuda"
            )

            # Initialize theta and phi uniformly across the sphere for each primitive and view-dependent parameter
            theta = torch.pi * torch.rand(
                fused_point_cloud.shape[0], self.sb_number
            )  # Uniform in [0, pi]
            phi = (
                2 * torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)
            )  # Uniform in [0, 2pi]

            sb_params[:, :, 3] = theta
            sb_params[:, :, 4] = phi
        
        elif self.use_gmm_colors:
            print("Initializing features for non-beta mode")
            print("# Nasg lobes is: ", self.max_sh_degree)
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            shs = (
                torch.rand((fused_color.shape[0], 1, (self.max_sh_degree) * self.params_size + 3))
                .float()
                .cuda() * 2.0 - 1.0) * 2.0
            
            pos_start = 0
            pos_end = self.position_size * self.max_sh_degree     
            shs[:, :, pos_start:pos_end] = 0.0 # + 0.1 * torch.randn_like(features[:, :, pos_start:pos_end]) # positions
            
            weight_start = -3 + (-1) * self.max_sh_degree * self.color_size #* self.max_sh_degree - self.color_size
            weight_end   = -3
            shs[:, :, weight_start:weight_end] = 0.5 #0.5 + 0.05 * torch.randn_like(features[:, :, weights_start:weights_end]) # weights

            shape_start = self.position_size * self.max_sh_degree
            shape_end   = -3 + (-1) * self.max_sh_degree * self.color_size #-3 * self.max_sh_degree - self.color_size
            if self.max_sh_degree  == 0:
                shape_end = 0
            
            # Initialize ALL shape parameters for all degrees
            # Calculate shape params per degree: total - position - color
            shape_params_per_deg = self.shape_size
            for deg in range(self.max_sh_degree):
                for param_idx in range(shape_params_per_deg):
                    idx = shape_start + deg * shape_params_per_deg + param_idx
                    shs[:,:, idx] = math.log(0.5)  # Initialize all shape params to log(0.5)

            shs[:, 0, -3:] = fused_color #torch.log(fused_color + 1.0)

        elif self.use_gmm_colors_cuda:
            # fused_color: [N, 3]
            # shs: [N, 1, total_params_per_point]

            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            # Allocate tensor
            features = torch.empty((fused_color.shape[0], 1, (self.max_sh_degree) * self.params_size + 3)).float().cuda()

            for i in range(self.max_sh_degree):
                start = i * self.params_size
                # positions
                features[:, 0, start : start + self.position_size] = 0.0
                # shape
                features[:, 0, start + self.position_size : start + self.position_size + self.shape_size] = math.log(0.5)
                # weight
                features[:, 0, start + self.position_size + self.shape_size : start + self.position_size + self.shape_size + self.color_size] = 0.5

            # Finally, append base color at the very end
            features[:, 0, -3:] = fused_color
        
        #if self.use_beta:
        #    print("First SH coefficients:", shs[0, :, :])
        #elif self.use_gmm_colors_cuda:
        #    print("First features:", features[0, :])
        #else:
        #    print("First features:", shs[0, :])

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = (
            knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[:, 1:] ** 2
        ).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        ## Betas initialization
        betas = torch.zeros_like(opacities)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        if self.use_beta:
            self._sh0 = nn.Parameter(
            shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._shN = nn.Parameter(
            shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._sb_params = nn.Parameter(sb_params.requires_grad_(True))
        elif self.use_gmm_colors:
            self._sh0 = nn.Parameter(
            shs[:, :, -3:].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._features_pos = nn.Parameter(
            shs[:, :, pos_start:pos_end].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._features_shape = nn.Parameter(
            shs[:, :, shape_start:shape_end].transpose(1, 2).contiguous().requires_grad_(True)
            )
            self._features_weight = nn.Parameter(
            shs[:, :, weight_start:weight_end].transpose(1, 2).contiguous().requires_grad_(True)
            )
        elif self.use_gmm_colors_cuda:
            self._sh0 = nn.Parameter(
                features[:, :, -3:].transpose(1, 2).contiguous().requires_grad_(True)
            )
            
            # Extract interleaved params: [pos, shape, weight] * max_sh_degree
            params_only = features[:, 0, :-3]  # [N, max_sh_degree * params_size]
            params_reshaped = params_only.reshape(
                features.shape[0], self.max_sh_degree, self.params_size
            )  # [N, max_sh_degree, params_size]
            
            # Extract each component type
            pos = params_reshaped[:, :, 0:self.position_size]  # [N, max_sh_degree, 3]
            shape = params_reshaped[:, :, self.position_size:self.position_size + self.shape_size]  # [N, max_sh_degree, 2]
            weight = params_reshaped[:, :, self.position_size + self.shape_size:]  # [N, max_sh_degree, 3]
            
            # Reshape and store as parameters
            self._features_pos = nn.Parameter(
                pos.reshape(features.shape[0], self.max_sh_degree * self.position_size)
                .unsqueeze(1)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_shape = nn.Parameter(
                shape.reshape(features.shape[0], self.max_sh_degree * self.shape_size)
                .unsqueeze(1)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_weight = nn.Parameter(
                weight.reshape(features.shape[0], self.max_sh_degree * self.color_size)
                .unsqueeze(1)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._sh0 = self._sh0[live_mask]
        if self.use_beta:
            self._shN = self._shN[live_mask]
            self._sb_params = self._sb_params[live_mask]
        else:
            self._features_pos = self._features_pos[live_mask]
            self._features_shape = self._features_shape[live_mask]
            self._features_weight = self._features_weight[live_mask]
        self._scaling = self._scaling[live_mask]
        self._rotation = self._rotation[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]

    def training_setup(self, training_args):
        if self.use_beta:
            l = [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init * self.spatial_lr_scale,
                    "name": "xyz",
                },
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {"params": [self._shN], "lr": training_args.sh_lr / 20.0, "name": "shN"},
                {
                    "params": [self._sb_params],
                    "lr": training_args.sb_params_lr,
                    "name": "sb_params",
                },
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
                },
                {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
                {
                    "params": [self._scaling],
                    "lr": training_args.scaling_lr,
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                },
            ]
        else:
            l = [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init * self.spatial_lr_scale,
                    "name": "xyz",
                },
                {"params": [self._sh0], "lr": training_args.sh_lr, "name": "sh0"},
                {
                    "params": [self._features_pos],
                    "lr": training_args.features_pos_lr,
                    "name": "features_pos",
                },
                {
                    "params": [self._features_shape],
                    "lr": training_args.features_shape_lr,
                    "name": "features_shape",
                },
                {
                    "params": [self._features_weight],
                    "lr": training_args.features_weight_lr,
                    "name": "features_weight",
                },
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
                },
                {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
                {
                    "params": [self._scaling],
                    "lr": training_args.scaling_lr,
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                },
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        # def lr_lambda(step):
        #     T_max = training_args.iterations
        #     return 0.5 * (1 + math.cos(math.pi * step / T_max))  # cosine decay

        # # Create scheduler that only adjusts specific groups
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=[(lambda step: 1.0 if pg["name"] == "xyz" else lr_lambda(step)) for pg in self.optimizer.param_groups]
        # )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._sh0.shape[1] * self._sh0.shape[2]):
            l.append("sh0_{}".format(i))

        if self.use_beta:
            for i in range(self._shN.shape[1] * self._shN.shape[2]):
                l.append("shN_{}".format(i))
            for i in range(self._sb_params.shape[1] * self._sb_params.shape[2]):
                l.append("sb_params_{}".format(i))
        else:
            ### TODO: Is it correct?
            for i in range(self._features_pos.shape[1]*self._features_pos.shape[2] + self._features_shape.shape[1]*self._features_shape.shape[2] + self._features_weight.shape[1]*self._features_weight.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append("opacity")
        l.append("beta")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sh0 = (
            self._sh0.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        
        if self.use_beta:
            sh0 = (
            self._sh0.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
            )
            shN = (
                self._shN.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            sb_params = (
                self._sb_params.transpose(1, 2)
                .detach()
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
        else:
            sh0 = (
                self._sh0.detach()
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            if self.use_gmm_colors_cuda:
                # Interleaved layout: [pos, shape, weight] * lobes
                pos = self._features_pos.squeeze(-1).reshape(-1, self.max_sh_degree, self.position_size)
                shape = self._features_shape.squeeze(-1).reshape(-1, self.max_sh_degree, self.shape_size)
                weight = self._features_weight.squeeze(-1).reshape(-1, self.max_sh_degree, self.color_size)
                interleaved = torch.cat([pos, shape, weight], dim=2)  # [N, max_sh_degree, params_size]
                nasg_params = interleaved.reshape(-1, self.max_sh_degree * self.params_size).detach().cpu().numpy()
            else:
                # Linear layout: [all_pos, all_shape, all_weight]
                nasg_params = torch.cat((self._features_pos, self._features_shape, self._features_weight), dim=1).detach().flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_beta:
            attributes = np.concatenate(
                (xyz, normals, sh0, shN, sb_params, opacities, betas, scale, rotation),
                axis=1,
            )
        else:
            attributes = np.concatenate(
                (xyz, normals, sh0, nasg_params, opacities, betas, scale, rotation),
                axis=1,
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    ### Not changed - idk for what is this method 
    def save_png(self, path):
        path = os.path.join(path, "png")
        mkdir_p(path)
        start_time = time.time()
        opacities = self.get_opacity
        N = opacities.numel()
        n_sidelen = int(N**0.5)
        n_crop = N - n_sidelen**2
        if n_crop:
            index = torch.argsort(opacities.squeeze(), descending=True)
            mask = torch.zeros(N, dtype=torch.bool, device=opacities.device).scatter_(
                0, index[:-n_crop], True
            )
            self.prune(mask.squeeze())
        meta = {}
        if self.use_beta:
            param_dict = {
                "xyz": self._xyz,
                "sh0": self._sh0,
                "shN": self._shN if self.max_sh_degree else None,
                "opacity": self._opacity,
                "beta": self._beta,
                "scaling": self._scaling,
                "rotation": self.get_rotation,
                "sb_params": self._sb_params if self.sb_number else None,
            }
            param_dict = sort_param_dict(param_dict, n_sidelen)
            for k in param_dict.keys():
                if param_dict[k] is not None:
                    if k == "sb_params":
                        for i in range(self.sb_number):
                            meta[f"sb_{i}_color"] = compress_png(
                                path, f"sb_{i}_color", param_dict[k][:, i, :3], n_sidelen
                            )

                            meta[f"sb_{i}_lobe"] = compress_png(
                                path, f"sb_{i}_lobe", param_dict[k][:, i, 3:], n_sidelen
                            )
                    elif k == "xyz":
                        meta[k] = compress_png(path, k, param_dict[k], n_sidelen, bit=32)
                    else:
                        meta[k] = compress_png(path, k, param_dict[k], n_sidelen)
        else:
            param_dict = {
                "xyz": self._xyz,
                "sh0": self._sh0,
                "features_pos": self._features_pos if self.max_sh_degree else None,
                "features_shape": self._features_shape if self.max_sh_degree else None,
                "features_weight": self._features_weight if self.max_sh_degree else None,
                "opacity": self._opacity,
                "beta": self._beta,
                "scaling": self._scaling,
                "rotation": self.get_rotation,
            }
            param_dict = sort_param_dict(param_dict, n_sidelen)
            for k in param_dict.keys():
                if param_dict[k] is not None:
                    if k in ["features_pos", "features_shape", "features_weight"]:
                        meta[k] = compress_png(path, k, param_dict[k], n_sidelen)
                    elif k == "xyz":
                        meta[k] = compress_png(path, k, param_dict[k], n_sidelen, bit=32)
                    else:
                        meta[k] = compress_png(path, k, param_dict[k], n_sidelen)

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        end_time = time.time()
        print(f"Compression time: {end_time - start_time:.2f} seconds")

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        betas = np.asarray(plydata.elements[0]["beta"])[..., np.newaxis]

        if self.use_beta:
            sh0 = np.zeros((xyz.shape[0], 3, 1))
            sh0[:, 0, 0] = np.asarray(plydata.elements[0]["sh0_0"])
            sh0[:, 1, 0] = np.asarray(plydata.elements[0]["sh0_1"])
            sh0[:, 2, 0] = np.asarray(plydata.elements[0]["sh0_2"])

            extra_f_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("shN_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            shs_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                shs_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            shs_extra = shs_extra.reshape(
                (shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("sb_params_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == self.sb_number * 6
            sb_params = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                sb_params[:, idx] = np.asarray(plydata.elements[0][attr_name])
            sb_params = sb_params.reshape((sb_params.shape[0], 6, self.sb_number))

        else:
            sh0 = np.zeros((xyz.shape[0], 1, 3))
            sh0[:, 0, 0] = np.asarray(plydata.elements[0]["sh0_0"])
            sh0[:, 0, 1] = np.asarray(plydata.elements[0]["sh0_1"])
            sh0[:, 0, 2] = np.asarray(plydata.elements[0]["sh0_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            
            if self.use_gmm_colors:
                extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
                features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
                features_extra = features_extra.reshape((features_extra.shape[0], 1, (self.max_sh_degree) * self.params_size))
            elif self.use_gmm_colors_cuda:
                extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
                features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                # Reshape (P,F*SH_coeffs) to (P, 1, SH_coeffs except DC)
                features_extra = features_extra.reshape((features_extra.shape[0], 1, (self.max_sh_degree) * self.params_size))

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sh0 = nn.Parameter(
            torch.tensor(sh0, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        
        if self.use_beta:
            self._shN = nn.Parameter(
            torch.tensor(shs_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
            )
            self._sb_params = nn.Parameter(
                torch.tensor(sb_params, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
        else:
            if self.use_gmm_colors_cuda:
                # Interleaved format: [N, 1, max_sh_degree * params_size]
                # Reshape and extract components
                params_reshaped = features_extra.reshape(-1, self.max_sh_degree, self.params_size)
                pos_np = params_reshaped[:, :, 0:self.position_size]
                shape_np = params_reshaped[:, :, self.position_size:self.position_size + self.shape_size]
                weight_np = params_reshaped[:, :, self.position_size + self.shape_size:]
                
                self._features_pos = nn.Parameter(
                    torch.tensor(pos_np.reshape(-1, self.max_sh_degree * self.position_size), dtype=torch.float, device="cuda")
                    .unsqueeze(-1)
                    .contiguous()
                    .requires_grad_(True)
                )
                self._features_shape = nn.Parameter(
                    torch.tensor(shape_np.reshape(-1, self.max_sh_degree * self.shape_size), dtype=torch.float, device="cuda")
                    .unsqueeze(-1)
                    .contiguous()
                    .requires_grad_(True)
                )
                self._features_weight = nn.Parameter(
                    torch.tensor(weight_np.reshape(-1, self.max_sh_degree * self.color_size), dtype=torch.float, device="cuda")
                    .unsqueeze(-1)
                    .contiguous()
                    .requires_grad_(True)
                )
            else:
                # Linear format: [all_pos, all_shape, all_weight]
                self._features_pos = nn.Parameter(
                    torch.tensor(features_extra[:,:,0:self.position_size*self.max_sh_degree], dtype=torch.float, device="cuda")
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                )
                self._features_shape = nn.Parameter(
                    torch.tensor(features_extra[:,:,self.position_size*self.max_sh_degree:-self.color_size *self.max_sh_degree], dtype=torch.float, device="cuda")
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                )
                self._features_weight = nn.Parameter(
                    torch.tensor(features_extra[:,:,-self.color_size * self.max_sh_degree:], dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    # still idk whats going on with this load_png method
    def load_png(self, path):
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        xyz = decompress_png(path, "xyz", meta["xyz"])
        sh0 = decompress_png(path, "sh0", meta["sh0"])

        if self.use_beta:
            shN = (
                decompress_png(path, "shN", meta["shN"])
                if self.max_sh_degree
                else np.zeros((xyz.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))
            )
        opacity = decompress_png(path, "opacity", meta["opacity"])
        beta = decompress_png(path, "beta", meta["beta"])
        scaling = decompress_png(path, "scaling", meta["scaling"])
        rotation = decompress_png(path, "rotation", meta["rotation"])
        if self.use_beta:
            if self.sb_number:
                sb_params_list = []
                for i in range(self.sb_number):
                    color = decompress_png(path, f"sb_{i}_color", meta[f"sb_{i}_color"])
                    direction = decompress_png(path, f"sb_{i}_lobe", meta[f"sb_{i}_lobe"])
                    # Concatenate along the feature dimension (expecting 3 channels each)
                    sb = np.concatenate(
                        [color, direction], axis=1
                    )  # shape: (num_points, 6)
                    sb_params_list.append(sb)
                # Stack to get shape (num_points, 6, sb_number)
                sb_params = np.stack(sb_params_list, axis=2)
            else:
                sb_params = np.zeros((xyz.shape[0], 6, self.sb_number))
        else:
            nasg_params_list = []
            for i in range(self.max_sh_degree):
                pos = decompress_png(path, f"features_pos_{i}", meta[f"features_pos_{i}"])
                shape = decompress_png(path, f"features_shape_{i}", meta[f"features_shape_{i}"])
                weight = decompress_png(path, f"features_weight_{i}", meta[f"features_weight_{i}"])
                nasg = np.concatenate([pos, shape, weight], axis=1)
                nasg_params_list.append(nasg)

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sh0 = nn.Parameter(
            torch.tensor(sh0, dtype=torch.float, device="cuda")
            .contiguous()
            .requires_grad_(True)
        )
        if self.use_beta:
            self._shN = nn.Parameter(
                torch.tensor(shN, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
            self._sb_params = nn.Parameter(
                torch.tensor(sb_params, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
        else:
            self._features_pos = nn.Parameter(
                torch.tensor(features_extra[:,:,0:self.position_size*self.max_sh_degree], dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_shape = nn.Parameter(
                torch.tensor(features_extra[:,:,self.position_size*self.max_sh_degree:self.position_size*self.max_sh_degree+3], dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
            self._features_weight = nn.Parameter(
                torch.tensor(features_extra[:,:,self.position_size*self.max_sh_degree+3:self.position_size*self.max_sh_degree+6], dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )

        self._opacity = nn.Parameter(
            torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._beta = nn.Parameter(
            torch.tensor(beta, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scaling, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rotation, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_sh0,
        new_shN,
        new_sb_params,
        new_features_pos,
        new_features_shape,
        new_features_weight,
        new_opacities,
        new_betas,
        new_scaling,
        new_rotation,
        reset_params=True,
    ):
        if self.use_beta:
            d = {
                "xyz": new_xyz,
                "sh0": new_sh0,
                "shN": new_shN,
                "sb_params": new_sb_params,
                "opacity": new_opacities,
                "beta": new_betas,
                "scaling": new_scaling,
                "rotation": new_rotation,
            }
        else:
            d = {
                "xyz": new_xyz,
                "sh0": new_sh0,
                "features_pos": new_features_pos,
                "features_shape": new_features_shape,
                "features_weight": new_features_weight,
                "opacity": new_opacities,
                "beta": new_betas,
                "scaling": new_scaling,
                "rotation": new_rotation,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        if self.use_beta:
            self._shN = optimizable_tensors["shN"]
            self._sb_params = optimizable_tensors["sb_params"]
        else:
            self._features_pos = optimizable_tensors["features_pos"]
            self._features_shape = optimizable_tensors["features_shape"]
            self._features_weight = optimizable_tensors["features_weight"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def replace_tensors_to_optimizer(self, inds=None):
        if self.use_beta:

            tensors_dict = {
                "xyz": self._xyz,
                "sh0": self._sh0,
                "shN": self._shN,
                "sb_params": self._sb_params,
                "opacity": self._opacity,
                "beta": self._beta,
                "scaling": self._scaling,
                "rotation": self._rotation,
            }
        else:
            tensors_dict = {
                "xyz": self._xyz,
                "sh0": self._sh0,
                "features_pos": self._features_pos,
                "features_shape": self._features_shape,
                "features_weight": self._features_weight,
                "opacity": self._opacity,
                "beta": self._beta,
                "scaling": self._scaling,
                "rotation": self._rotation,
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        if self.use_beta:
            self._shN = optimizable_tensors["shN"]
            self._sb_params = optimizable_tensors["sb_params"]
        else:
            self._features_pos = optimizable_tensors["features_pos"]
            self._features_shape = optimizable_tensors["features_shape"]
            self._features_weight = optimizable_tensors["features_weight"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(
            1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1)
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        if self.use_beta:
            return (
                self._xyz[idxs],
                self._sh0[idxs],
                self._shN[idxs],
                self._sb_params[idxs],
                new_opacity,
                self._beta[idxs],
                self._scaling[idxs],
                self._rotation[idxs],
            )
        else:
            return (
                self._xyz[idxs],
                self._sh0[idxs],
                self._features_pos[idxs],
                self._features_shape[idxs],
                self._features_weight[idxs],
                new_opacity,
                self._beta[idxs],
                self._scaling[idxs],
                self._rotation[idxs],
            )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        print(f"Relocate: {dead_mask.sum().item()}")
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        if self.use_beta:
            (
                self._xyz[dead_indices],
                self._sh0[dead_indices],
                self._shN[dead_indices],
                self._sb_params[dead_indices],
                self._opacity[dead_indices],
                self._beta[dead_indices],
                self._scaling[dead_indices],
                self._rotation[dead_indices],
            ) = self._update_params(reinit_idx, ratio=ratio)
        else:
            (
                self._xyz[dead_indices],
                self._sh0[dead_indices],
                self._features_pos[dead_indices],
                self._features_shape[dead_indices],
                self._features_weight[dead_indices],
                self._opacity[dead_indices],
                self._beta[dead_indices],
                self._scaling[dead_indices],
                self._rotation[dead_indices],
            ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        print(f"Add: {num_gs}, Now {target_num}")

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        if self.use_beta:
            (
                new_xyz,
                new_sh0,
                new_shN,
                new_sb_params,
                new_opacity,
                new_beta,
                new_scaling,
                new_rotation,
            ) = self._update_params(add_idx, ratio=ratio)
        else:
            (
                new_xyz,
                new_sh0,
                new_features_pos,
                new_features_shape,
                new_features_weight,
                new_opacity,
                new_beta,
                new_scaling,
                new_rotation,
            ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity

        if self.use_beta:
            self.densification_postfix(
                new_xyz,
                new_sh0,
                new_shN,
                new_sb_params,
                None,
                None,
                None,
                new_opacity,
                new_beta,
                new_scaling,
                new_rotation,
                reset_params=False,
            )
        else:
            self.densification_postfix(
                new_xyz,
                new_sh0,
                None,
                None,
                new_features_pos,
                new_features_shape,
                new_features_weight,
                new_opacity,
                new_beta,
                new_scaling,
                new_rotation,
                reset_params=False,
            )

        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        if mask == None:
            mask = torch.ones_like(self.get_beta.squeeze()).bool()

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)

        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)

        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        if self.use_beta:
            rgbs, alphas, meta = rasterization(
                means=self.get_xyz[mask],
                quats=self.get_rotation[mask],
                scales=self.get_scaling[mask],
                opacities=self.get_opacity.squeeze()[mask],
                betas=self.get_beta.squeeze()[mask],
                use_nasg=False,
                colors=self.get_shs[mask],
                viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=viewpoint_camera.image_width,
                height=viewpoint_camera.image_height,
                backgrounds=self.background.unsqueeze(0),
                render_mode=render_mode,
                covars=None,
                sh_degree=self.active_sh_degree,
                sb_number=self.sb_number,
                sb_params=self.get_sb_params[mask],
                packed=False,
            )
        elif self.use_gmm_colors:
                
            # calc colors with nasg function
            dir_pp = self.get_xyz[mask] - viewpoint_camera.camera_center.unsqueeze(0)
            dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-8)
            features = self.get_features.view(-1, (self.max_sh_degree) * self.params_size + 3)
            
            if self.active_sh_degree > 0:
                colors = nasg(dir_pp_normalized, features[mask, 3:], self.active_sh_degree, self.max_sh_degree)
                colors += features[mask, :3]
            else:
                colors = features[mask, :3]
                colors = torch.clamp_min(colors, 0.0)

            rgbs, alphas, meta = rasterization(
                means=self.get_xyz[mask],
                quats=self.get_rotation[mask],
                scales=self.get_scaling[mask],
                opacities=self.get_opacity.squeeze()[mask],
                betas=self.get_beta.squeeze()[mask],
                colors=colors,
                viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=viewpoint_camera.image_width,
                height=viewpoint_camera.image_height,
                backgrounds=self.background.unsqueeze(0),
                render_mode=render_mode,
                covars=None,
                sh_degree=None,
                sb_number=None,
                sb_params=None,
                packed=False,
                )

        elif self.use_gmm_colors_cuda:
            # ## TODO: not ready
            # ## Alternatively calc nasg in CUDA
            features = self.get_features.view(-1, (self.max_sh_degree) * self.params_size + 3)

            # Memory layout:
            # sh0 lobe1 lobe2 .., lobeN
            # sh0| pos1 shape1 weight1 | pos2 shape2 weight2 | ... | posN shapeN weightN
            #  3 |  3     2       3    |   3     2      3    | ... |  3     2       3

            # Assign zero to colors if no SH
            colors = torch.zeros((features.shape[0], 3), device=features.device)
            rgbs, alphas, meta = rasterization(
                means=self.get_xyz[mask],
                quats=self.get_rotation[mask],
                scales=self.get_scaling[mask],
                opacities=self.get_opacity.squeeze()[mask],
                betas=self.get_beta.squeeze()[mask],
                use_nasg=True,
                colors=colors,
                viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=viewpoint_camera.image_width,
                height=viewpoint_camera.image_height,
                backgrounds=self.background.unsqueeze(0),
                render_mode=render_mode,
                covars=None,
                sh_degree=self.active_sh_degree,
                sb_number=None,
                sb_params=features,
                packed=False,
            )

        # # Convert from N,H,W,C to N,C,H,W format
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
        }

    @torch.no_grad()
    def view(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer."""
        assert isinstance(render_tab_state, BetaRenderTabState)
        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        if center:
            xyz = self._xyz - self._xyz.mean(dim=0, keepdim=True)
        else:
            xyz = self._xyz

        render_mode = render_tab_state.render_mode
        mask = torch.logical_and(
            self._beta >= render_tab_state.b_range[0],
            self._beta <= render_tab_state.b_range[1],
        ).squeeze()
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        if self.use_beta:
            render_colors, alphas, meta = rasterization(
                means=xyz[mask],
                quats=self.get_rotation[mask],
                scales=self.get_scaling[mask],
                opacities=self.get_opacity.squeeze()[mask],
                betas=self.get_beta.squeeze()[mask],
                colors=self.get_shs[mask],
                viewmats=torch.linalg.inv(c2w).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
                backgrounds=self.background.unsqueeze(0),
                render_mode=render_mode if render_mode != "Alpha" else "RGB",
                covars=None,
                sh_degree=self.active_sh_degree,
                sb_number=self.sb_number,
                sb_params=self.get_sb_params[mask],
                packed=False,
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
            )
        else:


            # get camera center robustly (method / attribute / from c2w)
            cam_center = None
            if callable(getattr(camera_state, "get_camera_center", None)):
                cam_center = camera_state.get_camera_center()
            elif getattr(camera_state, "camera_center", None) is not None:
                cam_center = camera_state.camera_center
            elif getattr(camera_state, "c2w", None) is not None:
                c2w_np = camera_state.c2w
                # c2w may be numpy array or list-like
                c2w_arr = np.asarray(c2w_np)
                cam_center = c2w_arr[:3, 3]
            else:
                raise AttributeError("CameraState has no camera center (no get_camera_center, camera_center or c2w).")
            # convert to torch on the same device as model points
            if isinstance(cam_center, torch.Tensor):
                cam_center_t = cam_center.to(self._xyz.device).float().unsqueeze(0)
            else:
                cam_center_t = torch.from_numpy(np.asarray(cam_center)).float().to(self._xyz.device).unsqueeze(0)
            # calc colors with nasg function
            dir_pp = self.get_xyz[mask] - cam_center_t
            dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-8)
            features = self.get_features.view(-1, (self.max_sh_degree) * self.params_size + 3)
            
            if self.use_gmm_colors:
                if self.active_sh_degree > 0:
                    colors = nasg(dir_pp_normalized, features[mask, 3:], self.active_sh_degree, self.max_sh_degree)
                    colors += features[mask, :3]
                else:
                    colors = features[mask, :3]
                    colors = torch.clamp_min(colors, 0.0)

                render_colors, alphas, meta = rasterization(
                    means=self.get_xyz[mask],
                    quats=self.get_rotation[mask],
                    scales=self.get_scaling[mask],
                    opacities=self.get_opacity.squeeze()[mask],
                    betas=self.get_beta.squeeze()[mask],
                    colors=colors,
                    viewmats=torch.linalg.inv(c2w).unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=W,
                    height=H,
                    backgrounds=self.background.unsqueeze(0),
                    render_mode=render_mode,
                    covars=None,
                    sh_degree=None,
                    sb_number=None,
                    sb_params=None,
                    packed=False,
                )

            elif self.use_gmm_colors_cuda:
                # ## TODO: not ready
                # ## Alternatively calc nasg in CUDA
                features = self.get_features.view(-1, (self.max_sh_degree) * self.params_size + 3)

                # Memory layout:
                # sh0 lobe1 lobe2 .., lobeN
                # sh0| pos1 shape1 weight1 | pos2 shape2 weight2 | ... | posN shapeN weightN
                #  3 |  3     2       3    |   3     2      3    | ... |  3     2       3

                # Assign to colors placeholder
                colors = torch.zeros((features.shape[0], 3), device=features.device)
                render_colors, alphas, meta = rasterization(
                    means=self.get_xyz[mask],
                    quats=self.get_rotation[mask],
                    scales=self.get_scaling[mask],
                    opacities=self.get_opacity.squeeze()[mask],
                    betas=self.get_beta.squeeze()[mask],
                    use_nasg=True,
                    colors=colors,
                    viewmats=torch.linalg.inv(c2w).unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=W,
                    height=H,
                    backgrounds=self.background.unsqueeze(0),
                    render_mode=render_mode,
                    covars=None,
                    sh_degree=self.active_sh_degree,
                    sb_number=None,
                    sb_params=features,
                    packed=False,
                )

        render_tab_state.total_count_number = len(self.get_xyz)
        render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

        if render_mode == "Alpha":
            render_colors = alphas

        if render_colors.shape[-1] == 1:
            render_colors = apply_depth_colormap(render_colors)

        return render_colors[0].cpu().numpy()
