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

from argparse import ArgumentParser, Namespace
import sys
import os
from utils.spherical_utils import nasg, nasg_gabor, spherical_fb6, spherical_logistic, spherical_gaussian, asg, spherical_cauchy, spherical_beta, spherical_fb8

PARAMS_SIZE = {
        "fb8": 11,
        "slog": 7,
        "slog_ycbcr": 5,
        "sg": 6,
        "nasg": 8,
        "nasg_gabor": 9,
        "nasg_ycbcr": 6,
        "fb6": 9,
        "sc": 6,
        "sbeta": 6,
        "asg": 8
    }
        
POSITION_SIZE = {
        "fb6": 3,
        "fb8": 3,
        "slog": 2,
        "slog_ycbcr": 2,
        "sc": 3,
        "nasg": 3,
        "nasg_gabor": 3,
        "nasg_ycbcr": 3,
        "sg": 2,
        "sbeta": 2,
        "asg": 3
    }

COLOR_SIZE = {
        "fb8": 3,
        "slog": 3,
        "slog_ycbcr": 1,
        "sg": 3,
        "nasg": 3,
        "nasg_gabor": 3,
        "nasg_ycbcr": 1,
        "fb6": 3,
        "sc": 3,
        "sbeta": 3,
        "asg": 3
}

COLOR_FUNCTION = {
        "nasg": nasg,
        "nasg_gabor": nasg_gabor,
        "fb6": spherical_fb6,
        "fb8": spherical_fb8,
        "slog": spherical_logistic,
        "sg": spherical_gaussian,
        "asg": asg,
        "sc": spherical_cauchy,
        "sbeta": spherical_beta
    }

SHAPE_SIZE = {}

for method in COLOR_SIZE:
    total = PARAMS_SIZE.get(method, 0)
    pos = POSITION_SIZE.get(method, 0)
    color = COLOR_SIZE.get(method, 0)
    
    shape_size = total - pos - color
    SHAPE_SIZE[method] = shape_size


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self.sb_number = 2
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.cap_max = 1500000
        self.init_type = "sfm"
        
        super().__init__(parser, "Loading Parameters", sentinel)
        
        # Add custom argument for rendering_mode with choices (after super init)
        if not sentinel:
            group = parser.add_argument_group("Loading Parameters")
            group.add_argument(
                "--rendering_mode",
                type=str,
                required=True,
                choices=["beta", "gmm", "gmm_cuda"],
                help="Rendering mode (REQUIRED): 'beta' (SH+beta), 'gmm' (NASG Python), or 'gmm_cuda' (NASG CUDA)"
            )
            group.add_argument(
                "--gmm_color_mode",
                type=str,
                default="nasg",
                choices=list(COLOR_FUNCTION.keys()),
                help="GMM color mode (optional, default 'nasg'): 'nasg' or 'nasg_gabor' (only applies if rendering_mode is 'gmm' or 'gmm_cuda')"
            )

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        
        # Set boolean flags based on rendering_mode
        rendering_mode = getattr(args, 'rendering_mode', 'beta')
        gmm_color_mode = getattr(args, 'gmm_color_mode', 'nasg')  # Default to 'nasg' if not set

        g.use_beta = (rendering_mode == "beta")
        g.use_gmm_colors = (rendering_mode == "gmm")
        g.use_gmm_colors_cuda = (rendering_mode == "gmm_cuda")
        g.gmm_color_mode = gmm_color_mode
        
        # Also set on args for direct access
        args.use_beta = g.use_beta
        args.use_gmm_colors = g.use_gmm_colors
        args.use_gmm_colors_cuda = g.use_gmm_colors_cuda
        args.gmm_color_mode = g.gmm_color_mode
        return g


class ViewerParams(ParamGroup):
    def __init__(self, parser):
        self.port = 8080
        self.disable_viewer = False
        super().__init__(parser, "Viewer Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.sh_lr = 0.00025
        self.sb_params_lr = 0.0025
        self.features_pos_lr = 0.0025
        self.features_shape_lr = 0.0025
        self.features_weight_lr = 0.0025
        self.opacity_lr = 0.025
        self.beta_lr = 0.001
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.random_background = False
        self.noise_lr = 5e4
        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        super().__init__(parser, "Optimization Parameters")
