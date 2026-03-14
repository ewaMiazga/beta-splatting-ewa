import os
import torch
import sys
from scene import Scene, BetaModel
from argparse import ArgumentParser
from arguments import ModelParams


def training(args):
    beta_model = BetaModel(args.sh_degree, args.sb_number, args.use_beta, args.use_gmm_colors, args.gmm_color_mode, args.use_gmm_colors_cuda)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)
    print("scene loaded ")
    ply_path = os.path.join(
        args.model_path, "point_cloud", "iteration_" + args.iteration, "point_cloud.ply"
    )
    print("ply_path:", ply_path)
    if os.path.exists(ply_path):
        print("Evaluating " + ply_path)
        beta_model.load_ply(ply_path)
        scene.eval()

    is_compressed = False
    if is_compressed:
        png_path = os.path.join(
            args.model_path, "point_cloud", "iteration_" + args.iteration, "png"
        )
        if os.path.exists(png_path):
            print("Evaluating " + png_path)
            beta_model.load_png(png_path)
            scene.eval()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    ModelParams(parser)
    parser.add_argument(
        "--iteration", default="best", type=str, help="Iteration to evaluate"
    )
    args = parser.parse_args(sys.argv[1:])
    
    # Set rendering mode boolean flags - all False by default
    rendering_mode = getattr(args, 'rendering_mode', None)
    gmm_color_mode = getattr(args, 'gmm_color_mode', None)
    
    if rendering_mode is None:
        print("ERROR: --rendering_mode must be specified. Choose: 'beta', 'gmm', or 'gmm_cuda'")
        sys.exit(1)
    
    args.use_beta = (rendering_mode == "beta")
    args.use_gmm_colors = (rendering_mode == "gmm")
    args.use_gmm_colors_cuda = (rendering_mode == "gmm_cuda")
    args.gmm_color_mode = gmm_color_mode
    
    args.eval = True

    print("Evaluating " + args.model_path)

    training(args)
