import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.special import erf
import scipy.special as sc
from tqdm import tqdm
from typing import *

def get_param(index: int, params: torch.Tensor, max_gaussians: int, active_gaussians: int, l: int = 1) -> torch.Tensor:
    return params[:, index * max_gaussians:index * max_gaussians + active_gaussians * l]

def dot(a, b):
    return torch.sum(a * b, dim=2, keepdim=True)

def get_basis_parameterized(cosθ, cosϕ, cosτ):
    clamp_min = -0.999999
    clamp_max = 0.999999
    cosθ = torch.clamp(cosθ, clamp_min, clamp_max)
    cosϕ = torch.clamp(cosϕ, clamp_min, clamp_max)
    cosτ = torch.clamp(cosτ, clamp_min, clamp_max)

    sinθ = torch.sqrt(1.0 - cosθ * cosθ)
    sinϕ = torch.sqrt(1.0 - cosϕ * cosϕ)
    sinτ = torch.sqrt(1.0 - cosτ * cosτ)

    x = torch.stack([
        cosθ * cosϕ * cosτ - sinθ * sinτ, 
        sinθ * cosϕ * cosτ + cosθ * sinτ, 
        - sinϕ * cosτ], 
        dim=2)

    z = torch.stack([
        cosθ * sinϕ,
        sinθ * sinϕ, 
        cosϕ],
        dim=2)

    return [x, z]


## NASG
def nasg(v :torch.Tensor, coeffs :torch.Tensor, max_gaussians :int, active_gaussians :int):
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)
    
    eps = 5e-6 # this avoids breaking continuity at v = -z best so far at 5e-6 or more stable at 1e-5
    x, z = get_basis_parameterized(param(0), param(1), param(2)) #activation function applied here -> what ensures that they are within -1 to 1?

    λ = param(3).unsqueeze(2)
    a = param(4).unsqueeze(2)

    act  = lambda x : torch.exp(x)

    λ = act(λ)
    a = act(a)
    λ = torch.clamp(λ, max=1e4) #max=250)#max=0.5*1e4) #min=1e-6)#, max=1e4)
    a = torch.clamp(a, max=1e4) #max=0.5*1e4) #min=1e-6)#, max=1e4)

    vz = dot(v, z)

    mask_one = vz >= 1.0 - 1e-7
    mask_zero = vz <= - 1.0 + 1e-7
    valid = ~mask_one & ~mask_zero 

    placeholder = torch.zeros_like(vz)

    K_base = (vz[valid] + 1.0) * 0.5
    numerator = a * (dot(v, x) ** 2.0)
    K_exp  = eps + numerator[valid] / (1.0 - vz[valid] ** 2.0)

    exp = torch.pow(K_base, K_exp)

    pdf = (torch.exp(2.0 * (λ*valid)[valid] * ((exp*K_base) -1.0)) * exp) * (inv_nasg_norm(λ, a) * valid)[valid]
    placeholder[valid] = pdf

    placeholder = torch.where(~mask_one, placeholder, 1.)  # condition * (x - y) + y
    pdf = torch.where(~mask_zero, placeholder, 0.)

    colors = pdf * param(5, l=3).reshape(-1, active_gaussians, 3)

    return torch.sum(colors, dim=1)

def inv_nasg_norm(λ, a, eps = 1e-8):
    num = (2.0 * torch.pi * (1.0 + eps - torch.exp(-2.0 * λ)))
    denom = λ * torch.sqrt(1.0 + a)
    return  denom / num

## NASG Gabor
def nasg_gabor(v :torch.Tensor, coeffs :torch.Tensor, max_gaussians :int, active_gaussians :int):
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)

    eps = 5e-6 # this avoids breaking continuity at v = -z best so far at 5e-6 or more stable at 1e-5
    x, z = get_basis_parameterized(param(0), param(1), param(2)) #activation function applied here -> what ensures that they are within -1 to 1?

    λ = param(3).unsqueeze(2)
    a = param(4).unsqueeze(2)

    act  = lambda x : torch.exp(x)

    λ = act(λ)
    a = act(a)
    λ = torch.clamp(λ, max=1e4) #max=250)#max=0.5*1e4) #min=1e-6)#, max=1e4)
    a = torch.clamp(a, max=1e4) #max=0.5*1e4) #min=1e-6)#, max=1e4

    k = param(5).unsqueeze(2)

    # activate k
    k = (torch.tanh(k) + 1.0) * 20.0 # multiplier may change || bounded with initialization
    vz = dot(v, z)

    mask_one = vz >= 1.0 - 1e-7
    mask_zero = vz <= - 1.0 + 1e-7
    valid = ~mask_one & ~mask_zero

    placeholder = torch.zeros_like(vz)

    K_base = (vz[valid] + 1.0) * 0.5
    numerator = a * (dot(v, x) ** 2.0)
    K_exp  = eps + numerator[valid] / (1.0 - vz[valid] ** 2.0)

    exp = torch.pow(K_base, K_exp)

    # gabor term
    cosine_term = torch.cos(k * dot(v, x))[valid]
    pdf = (torch.exp(2.0 * (λ*valid)[valid] * ((exp*K_base) -1.0)) * exp) * ((1.0 + cosine_term) * 0.5) * (inv_nasg_norm(λ, a) * valid)[valid]
    placeholder[valid] = pdf

    placeholder = torch.where(~mask_one, placeholder, 1.)  # condition * (x - y) + y
    pdf = torch.where(~mask_zero, placeholder, 0.)

    colors = pdf * param(6, l=3).reshape(-1, active_gaussians, 3)

    return torch.sum(colors, dim=1)


## Utility functions
def is_direction(x: torch.Tensor) -> bool:
    return x.shape[-1] == 3 and torch.allclose(torch.norm(x, dim=-1), torch.tensor(1.0, device=x.device))

def gauss_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def log_factorial(x):
  return torch.lgamma(x + 1)

def factorial(x):
  return torch.exp(log_factorial(x))

def gamma(x):
  return torch.exp(torch.lgamma(x))

def pochhammer(a, n):
   return gamma(a + n) / gamma(a)

def euler2rotmat(cx, cy, cz):
    clamp_min = -0.999999
    clamp_max = 0.999999
    cx = torch.clamp(cx, clamp_min, clamp_max)
    cy = torch.clamp(cy, clamp_min, clamp_max)
    cz = torch.clamp(cz, clamp_min, clamp_max)

    sx = torch.sqrt(1.0 - cx * cx)
    sy = torch.sqrt(1.0 - cy * cy)
    sz = torch.sqrt(1.0 - cz * cz)

    x = torch.stack([cy * cz, cx * sz + sx * sy * cz, sx * sz - cx * sy * cz], dim=2)
    y = torch.stack([-cy * sz, cx * cz - sx * sy * sz, sx * cz + cx * sy * sz], dim=2)
    z = torch.stack([sy, -sx * cy, cx * cy], dim=2)
    return x, y, z

def polar2cart(cx, cy) -> torch.Tensor:
    clamp_min = -0.999999
    clamp_max = 0.999999
    cx = torch.clamp(cx, clamp_min, clamp_max)
    cy = torch.clamp(cy, clamp_min, clamp_max)

    sx = torch.sqrt(1.0 - cx * cx)
    sy = torch.sqrt(1.0 - cy * cy)

    return torch.stack([sx * cy, sx * sy, cx], dim=2)

def rgb2ycbcr(c: torch.Tensor) -> torch.Tensor:
    # Define transformation matrix for RGB to YCbCr
    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).to(c.device)

    # Offsets for Cb and Cr channels
    offset = torch.tensor([0, 128, 128], device=c.device)

    # Apply matrix transformation
    ycbcr_img = torch.tensordot(c, transform_matrix, dims=([2], [1])) + offset

    return ycbcr_img

def rgb2ycbcr_matlab(c: torch.Tensor) -> torch.Tensor:
    # Assuming input RGB in range [0, 1] like in MATLAB
    img_scaled = c * 255.0  # Scale from [0, 1] to [0, 255] like MATLAB expects

    # Define transformation matrix for RGB to YCbCr (scaled to match MATLAB's output range)
    transform_matrix = torch.tensor([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214]
    ]).to(c.device)

    # Offsets for Y, Cb, Cr channels (to match the MATLAB scaling)
    offset = torch.tensor([16, 128, 128], device=c.device)

    # Apply matrix transformation
    ycbcr_img = (torch.tensordot(img_scaled, transform_matrix, dims=([2], [1])) + offset) / 255.0

    return ycbcr_img

def ycbcr2rgb_matlab(c: torch.Tensor) -> torch.Tensor:
    # Define the inverse transformation matrix for YCbCr to RGB
    inv_transform_matrix = torch.tensor([
        [1.164,  0.000,  1.596],
        [1.164, -0.392, -0.813],
        [1.164,  2.017,  0.000]
    ]).to(c.device)

    # Subtract the offsets for Y, Cb, Cr channels
    offset = torch.tensor([16, 128, 128], device=c.device)
    c = c - offset

    # Apply the inverse matrix multiplication
    rgb = torch.tensordot(c, inv_transform_matrix, dims=([2], [0]))

    # Scale RGB back to [0, 1] range
    rgb = rgb / 255.0

    return rgb

def luma2rgb_matlab(Y: torch.Tensor) -> torch.Tensor:
    # Reshape the flat Y (luma) array into its original image shape
    Y*=255.0
    # Create neutral Cb and Cr channels (assuming Cb = 128, Cr = 128, which is gray)
    Cb = torch.full_like(Y, 128.0)
    Cr = torch.full_like(Y, 128.0)

    # Subtract offsets (16 for Y, 128 for Cb and Cr)
    Y = Y - 16
    Cb -= 128
    Cr -= 128

    # Compute the RGB values using the inverse transformation formulas
    R = 1.164 * Y + 1.596 * Cr
    G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
    B = 1.164 * Y + 2.017 * Cb

    # Stack the RGB values together
    rgb = torch.stack([R, G, B], dim=1)

    # Scale RGB back to [0, 1] range and clamp to ensure valid range
    rgb = rgb / 255.0 #torch.clamp(rgb_flat / 255.0, 0, 1)

    return rgb

# not used
def cart2polar(v):
    x, y, z = v[:,0], v[:,1], v[:,2]
    r = torch.sqrt(x * x + y * y + z * z)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=0)


## Spherical distribution functions
def nasg_ycbcr(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)

    eps = 5e-6 # this avoids breaking continuity at v = -z best so far at 5e-6 or more stable at 1e-5
    x, z = get_basis_parameterized(param(0), param(1), param(2)) #activation function applied here

    λ = torch.exp(param(3).unsqueeze(2))
    a = torch.exp(param(4).unsqueeze(2))

    λ = torch.clamp(λ, min=1e-6, max=1e4)
    a = torch.clamp(a, min=1e-6, max=1e4)

    vz = dot(v, z)

    mask_one = vz >= 1.0 - 1e-7
    mask_zero = vz <= - 1.0 + 1e-7
    valid = ~mask_one & ~mask_zero

    placeholder = torch.zeros_like(vz)

    K_base = (vz[valid] + 1.0) * 0.5
    numerator = a * (dot(v, x) ** 2.0)
    K_exp  = eps + numerator[valid] / (1.0 - vz[valid] ** 2.0)

    exp = torch.pow(K_base, K_exp)

    pdf = (torch.exp(2.0 * (λ*valid)[valid] * ((exp*K_base) -1.0)) * exp) * (inv_nasg_norm(λ, a) * valid)[valid]
    placeholder[valid] = pdf

    placeholder = torch.where(~mask_one, placeholder, 1.)  # condition * (x - y) + y
    pdf = torch.where(~mask_zero, placeholder, 0.)
    colors = pdf.reshape(-1, active_gaussians) * param(5).reshape(-1, active_gaussians) ## Try with softmax as well. But now instead of RGB that sums up 1, it is Y only so it may not make sense

    return luma2rgb_matlab(torch.sum(colors, dim=1))

def vMF(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    # https://www.jstor.org/stable/3213566?seq=1
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    κ = torch.exp(param(2)).unsqueeze(2) + 1e-6

    # compute the pdf
    pdf = (κ / (4.0 * torch.pi * torch.sinh(κ))) * torch.exp(κ * dot(μ, v))

    # multiply color weighted by pdf density
    colors = pdf * param(3, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def spherical_beta(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    # https://arxiv.org/pdf/2501.18630
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    β = param(2).unsqueeze(2) # alternatively activate with a scaled tanh or similar. In practice a reasonable range is -5 to 5

    # compute the pdf
    μv = dot(μ, v)

    # Unnormalized pdf

    β = torch.clamp(β, -5.0, 5.0)
    exponent = 4.0 * torch.exp(β)

    pdf = torch.pow(torch.clamp(μv, 1e-6, 1.0), exponent)

    # Normalized pdf
    # norm = 2.0 * torch.pi  / exponent + 1
    # pdf = pdf / norm

    # multiply color weighted by pdf density
    colors = pdf * param(3, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def spherical_logistic(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    # analytical derivative of the spherical logistic function for theta, phi was derived in the paper
    # https://link.springer.com/article/10.1007/s40304-018-00171-2
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    k = F.softplus(param(2)).unsqueeze(2)
    b = 1. + F.softplus(param(3)).unsqueeze(2)

    # clamp k, b to avoid numerical issues
    k = torch.clamp(k, min=1e-6, max=1e6)
    b = torch.clamp(b, min=1e-6, max=1e6)

    # compute the pdf
    exp = torch.exp(k * dot(μ, v))
    num = k * (b ** 2 + 2.0 * (b - 1.0) * (torch.cosh(k) - 1.0))
    denom = 4 * torch.pi * torch.sinh(k)
    C =  num / denom
    pdf = C * (exp / ((b - 1.0 + exp) ** 2.0))

    # multiply color weighted by pdf density
    colors = pdf * param(4, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def spherical_logistic_ycbcr(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    k = F.softplus(param(2)).unsqueeze(2)
    b = 1. + F.softplus(param(3)).unsqueeze(2)

    # clamp k, b to avoid numerical issues
    k = torch.clamp(k, min=1e-6, max=1e6)
    b = torch.clamp(b, min=1e-6, max=1e6)

    # compute the pdf
    exp = torch.exp(k * dot(μ, v))
    num = k * (b ** 2 + 2.0 * (b - 1.0) * (torch.cosh(k) - 1.0))
    denom = 4 * torch.pi * torch.sinh(k)
    C =  num / denom
    pdf = C * (exp / ((b - 1.0 + exp) ** 2.0))

    # multiply color weighted by pdf density
    colors = pdf.reshape(-1, active_gaussians) * param(4).reshape(-1, active_gaussians)
    return  luma2rgb_matlab(torch.sum(colors, dim=1))

def get_dirs(env_w = 800, env_h = 400):
    Az = ((torch.arange(env_w, dtype = torch.float32) + 0.5) / env_w - 0.5) * 2 * torch.pi
    El = ((torch.arange(env_h, dtype = torch.float32) + 0.5) / env_h) * torch.pi / 2.0
    Az, El = torch.meshgrid(Az, El)
    lx = torch.sin(El) * torch.cos(Az)
    ly = torch.sin(El) * torch.sin(Az)
    lz = torch.cos(El)
    ls = torch.stack((lx, ly, lz), dim = -1).permute(1, 0, 2).cuda()
    return ls

def spherical_gaussian(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    λ  = torch.exp(param(2)).unsqueeze(2)
    λ = torch.clamp(λ, min=1e-6, max=1e4)
    colors = param(3, l=3).reshape(-1, active_gaussians, 3)

    # # compute the pdf
    # norm = (2.0 * torch.pi / λ) * (1.0 - torch.exp(-2.0 * λ))
    # pdf = torch.exp(λ * (dot(μ, v) - 1.0)) / norm

    # --- PDF computation ---
    dot_val = dot(μ, v)
    norm = (2.0 * torch.pi / λ) * (1.0 - torch.exp(-2.0 * λ))
    exp_arg = λ * (dot_val - 1.0)
    pdf = torch.exp(exp_arg) / norm

    return torch.sum(pdf * colors, dim=1)


# hyp1f1_cache = precompute_hyp1f1_half_coeffs(a=0.5, b=1, k=70)  # >35 will cause overflow
def spherical_fb6(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    rotx, roty, rotz = euler2rotmat(param(0), param(1), param(2))

    k = torch.exp(param(3)).unsqueeze(2)
    β = torch.exp(param(4)).unsqueeze(2)

    # clamp beta and k to avoid numerical issues
    k = torch.clamp(k, min=1e-6, max=50)
    β = torch.clamp(β, min=1e-6, max=50)

    eta = torch.tanh(param(5)).unsqueeze(2)

    # compute the pdf
    switch = k <= 2. * β
    c = torch.zeros_like(k)

    hyp = hyp1f1(.5, 1., β[switch] * (1. + eta[switch]) * (k[switch] ** 2 / (4 * β[switch] ** 2) - 1), 20)
    c[switch] = 2 * torch.pi * torch.exp(β[switch] * (1 + k[switch]**2./(4*β[switch]**2.))) * torch.sqrt(torch.pi/β[switch]) * hyp
    c[~switch] = 2 * torch.pi * torch.exp(k[~switch]) / torch.sqrt((k[~switch] - 2*β[~switch]) * (k[~switch] + 2*β[~switch]*eta[~switch]))

    pdf = torch.exp(k * dot(rotz, v) + β * (dot(rotx, v)**2.0 - eta * (dot(roty, v)**2.0))) / c
    # pdf = torch.exp(k * dot(rotx, v) + β * (dot(roty, v)**2.0 - eta * (dot(rotz, v)**2.0))) / c # ?? 

    # multiply color weighted by pdf density
    colors = pdf * param(6, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def spherical_cauchy(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    # Broadcast v to match phi
    v = v.unsqueeze(1) # for broadcasting

    eps = 1e-6 # try with bigger eps 1e-3
    K_1 = 0.07957747154594767

    clamp_min = -0.999999
    clamp_max = 0.999999

    cosθ = torch.clamp(param(0), clamp_min, clamp_max)
    cosϕ = torch.clamp(param(1), clamp_min, clamp_max)
    cosτ = torch.clamp(param(2), clamp_min, clamp_max)

    # Get directional vectors (phi)
    phi = torch.stack([cosθ,
                       cosϕ,
                       cosτ],
                        dim=2)

    # Compute pdf
    diff = v - phi + eps
    norm = torch.sum(diff**2, dim=-1)
    phi_sum = torch.sum(phi, dim=1)
    pdf_numerator = torch.abs(1.0 - torch.norm(phi_sum, dim=-1))
    pdf = torch.pow(pdf_numerator.unsqueeze(1) / (norm + eps), 2.0)
    pdf *= K_1

    # Apply to color
    colors = pdf.unsqueeze(-1) * param(3, l=3).reshape(-1, active_gaussians, 3)  # (B, G, 3)

    return torch.sum(colors, dim=1) # (B, 3)

def spherical_fb4(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    # https://www.jstor.org/stable/pdf/2335218.pdf
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)  # for broadcasting

    # aquire the parameters
    μ = polar2cart(param(0), param(1))
    β = F.softplus(param(3)).unsqueeze(2)
    tau = β.detach() * 2. + F.softplus(param(2)).unsqueeze(2)

    # compute the pdf
    μv = dot(μ, v)

    two_tau = torch.sqrt(2.0 * tau)
    c = torch.sqrt(tau) / (2 * np.pi * np.sqrt(np.pi) * (gauss_cdf(two_tau * (1 - β)) - gauss_cdf(-two_tau * (1 + β))))
    pdf = c * torch.exp(-tau * (μv - β)**2)

    # multiply color weighted by pdf density
    colors = pdf * param(4, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def inf(x):
    return torch.isinf(x).any() | torch.isnan(x).any()

def spherical_fb8(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1).expand(-1, active_gaussians, -1)  # for broadcasting

    # aquire the parameters
    rotx, roty, rotz = euler2rotmat(param(0), param(1), param(2))
    k = torch.sigmoid(param(3)) * 5.0
    β = torch.sigmoid(param(4)) * 5.0
    eta = torch.tanh(param(5))
    μ = polar2cart(param(6), param(7))
    mat = torch.stack([rotx, roty, rotz], dim=-1)

    # compute the normalization term 
    c = spherical_fb8_norm_approx(k, β, eta, μ)

    term1 = dot(k.unsqueeze(-1) * μ, torch.einsum('ngij,ngj->ngi', mat, v))

    term2 = β * (torch.sum(roty * v.reshape(-1, active_gaussians, 3), dim=-1) ** 2.0 - eta * (torch.sum(rotz * v.reshape(-1, active_gaussians, 3), dim=-1) ** 2.0))

    # Normalized
    pdf = torch.exp(term1.reshape(-1, active_gaussians) + term2.reshape(-1, active_gaussians)) / c.unsqueeze(-1)

    # Unnormalized
    #pdf = torch.exp(term1.reshape(-1, active_gaussians) + term2.reshape(-1, active_gaussians))

    # multiply color weighted by pdf density
    colors = pdf.unsqueeze(-1).expand(-1, active_gaussians, -1) * param(8, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)


def compute_a(L, K, J, k, β, η, v):
    u = J + L + 0.5
    w = K + 0.5
    uv = J + L + K + 0.5
    a_num = (k ** (2.0 * (L+K))) * (β ** J) * (v[1,:] ** (2.0*L)) * (v[2,:] ** (2.0*K)) * gamma(w) * gamma(u)
    a_denom = factorial(2 * L) * factorial(2 * K) * factorial(J) * gamma(uv+1)
    prod = hyp0f1_torch(uv+1, (k**2) * (v[0,:] ** 2) / 4.0) * hyp2f1_torch(-J, w, 1-u, -η)
    return a_num * prod / (a_denom)

def compute_AB(L, K, J, k, β, η, v):
    s = 2 # alt seed
    sl1 = torch.arange(s * L, s * (L+1) - 1, device="cuda")
    sk1 = torch.arange(s * K, s * (K+1) - 1, device="cuda")
    sj1 = torch.arange(s * J, s * (J+1) - 1, device="cuda")
    a = torch.zeros_like(k)
    b = torch.zeros_like(k)
    for l1 in sl1:
        for k1 in sk1:
            for j1 in sj1:
                value = compute_a(l1, k1, j1, k, β, η, v)
                a += value
                b += torch.abs(value)
    return a, b

def compute_A(L, K, J, k, β, η, v):
    s = 2 # alt seed
    sl1 = torch.arange(s * L, s * (L+1) - 1, device="cuda")
    sk1 = torch.arange(s * K, s * (K+1) - 1, device="cuda")
    sj1 = torch.arange(s * J, s * (J+1) - 1, device="cuda")
    a = torch.zeros_like(k).reshape(-1)

    for l1 in sl1:
        for k1 in sk1:
            for j1 in sj1:
                value = compute_a(l1, k1, j1, k, β, η, v)
                a += value.sum(dim=1)#, keepdim=True)
    return a

def spherical_fb8_norm(k, β, η, v, eps=1e-6):
    norm = 2.0 * torch.sqrt(torch.tensor(torch.pi, device="cuda"))
    L, P_L, norm_cum = 0, torch.zeros_like(k), torch.zeros_like(k)
    while True:
        K, P_LK, S_L = 0, torch.zeros_like(k), torch.zeros_like(k)
        while True:
            J, P_LKJ, S_LK = 0, torch.zeros_like(k), torch.zeros_like(k)
            while True:
                ALKJ, BLKJ = compute_AB(L, K, J, k, β, η, v)
                mask_LKJ = (BLKJ < torch.abs(norm_cum) * eps) * (BLKJ <= P_LKJ) | J >= 8
                norm_cum += torch.where(mask_LKJ, 0.0, ALKJ)
                S_L += BLKJ
                S_LK += BLKJ
                if torch.all(mask_LKJ):
                    break
                P_LKJ = torch.where(mask_LKJ, torch.inf, BLKJ)
                J+=1
            mask_LK = (S_LK < torch.abs(norm_cum) * eps) * (S_LK <= P_LK) | K >= 2
            if torch.all(mask_LK):
                break
            P_LK = torch.where(mask_LK, torch.inf, S_LK)
            K+=1
        mask_L = (S_L < torch.abs(norm_cum) * eps) * (S_L <= P_L) | L >= 2
        if torch.all(mask_L):
                break
        P_L = torch.where(mask_L, torch.inf, S_L)
        L+=1
    return norm_cum * norm

def spherical_fb8_norm_approx(k, β, η, v, eps=1e-6):
    norm = 2.0 * torch.sqrt(torch.tensor(torch.pi, device="cuda"))
    norm_cum = torch.zeros_like(k).reshape(-1)
    vL = torch.arange(0, 2, device="cuda")
    vK = torch.arange(0, 2, device="cuda")
    vJ = torch.arange(0, 4, device="cuda")
    for L in vL:
        for K in vK:
            for J in vJ:
                norm_cum += compute_A(L, K, J, k, β, η, v)
    return norm_cum * norm

def spherical_fb8_norm_zeroth(k, β, η, v, eps=1e-6):
    norm = 2.0 * torch.sqrt(torch.tensor(torch.pi, device="cuda"))
    ALKJ = compute_A(0, 0, 0, k, β, η, v)
    return ALKJ * norm

def hyp1f1(a, b, x, n=40):
    """
    Memory-safe computation of 1F1(a, b, x)

    Args:
        a: Tensor, first parameter of 1F1.
        b: Tensor, second parameter of 1F1.
        x: Tensor, input to 1F1.
        n: int, number of terms to include in the series.

    Returns:
        Tensor representing 1F1(0.5, 1, x).
    """
    if len(x) == 0:
        return x

    result = torch.ones_like(x)  # Initialize the result tensor
    z = torch.ones_like(x)  # Initialize the first term of the series
    # Iteratively compute terms to avoid high memory usage
    for k in range(n):
        ab_poch = (a + k) / (b + k)
        z *= ab_poch * x/(k + 1)  # Add the current term
        result += z  # Add the current term to the result
    return result

def hyp0f1_torch(a, z, max_iter=20, tol=1e-10):
    """
    Approximates the confluent hypergeometric limit function 0F1(a; z) using a series expansion.

    Args:
    a: parameter (can be a tensor)
    z: input (can be a tensor)
    max_iter: number of terms to include in the series (default: 100)
    tol: tolerance for convergence (default: 1e-10)

    Returns:
    Approximation of 0F1(a; z)
    """
    # Initialize the series
    term = torch.ones_like(z)  # The initial term for n=0 is 1
    result = term.clone()

    pochhammer_value = torch.ones_like(a)  # Initialize (a)_0 = 1
    factorial = 1.0  # Initialize 0! = 1

    # Iteratively compute terms of the series
    for n in range(1, max_iter):
        # Update factorial and Pochhammer symbol (a)_n iteratively
        factorial *= n  # n!
        pochhammer_value *= (a + n - 1)  # Update Pochhammer (a)_n

        # Update term: z^n / ((a)_n * n!)
        term *= z / (pochhammer_value * factorial)
        result += term

        # Break if the next term is smaller than the tolerance
        if torch.max(torch.abs(term)) < tol:
            break

    return result

def hyp2f1_torch(a, b, c, z, max_iter=20, tol=1e-10):
    """
    Approximate the Gaussian hypergeometric function 2F1(a, b; c; z) using series expansion.
    The series is truncated after `max_iter` terms or when terms are smaller than `tol`.

    Args:
    a, b, c: Parameters of the hypergeometric function.
    z: Argument of the function (must satisfy |z| < 1 for convergence).
    max_iter: Maximum number of iterations (terms in the series).
    tol: Tolerance for truncating the series.

    Returns:
    Approximation of 2F1(a, b; c; z).
    """
    # Start with the 0th term
    term = torch.ones_like(z)
    result = term.clone()

    for n in range(1, max_iter):
        # Update the term using the Pochhammer symbols
        term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z

        # Add the new term to the result
        result += term

        # Stop if the term is small enough (converged)
        if torch.abs(term).max() < tol:
            break

    return result

def asg(v: torch.Tensor, coeffs: torch.Tensor, max_gaussians: int, active_gaussians: int) -> torch.Tensor:
    """Anisotropic Spherical Gaussian (ASG).

    G(v; [x,y,z], [λ, μ], c) = c * S(v;z) * exp(-λ*(v·x)² - μ*(v·y)²)

    where S(v;z) = max(v·z, 0).
    """
    param = partial(get_param, params=coeffs, max_gaussians=max_gaussians, active_gaussians=active_gaussians)
    v = v.unsqueeze(1)

    x, z = get_basis_parameterized(param(0), param(1), param(2))
    y = torch.cross(z, x, dim=2)  # bi-tangent: y = z × x

    λ = torch.exp(param(3).unsqueeze(2))
    μ = torch.exp(param(4).unsqueeze(2))
    λ = torch.clamp(λ, max=1e4)
    μ = torch.clamp(μ, max=1e4)

    dot_x = dot(v, x)
    dot_y = dot(v, y)
    dot_z = dot(v, z)

    # Smooth clamping: S(v; z) = max(v · z, 0)
    smooth = torch.clamp(dot_z, min=0.0)

    # ASG kernel
    pdf = smooth * torch.exp(-λ * dot_x ** 2 - μ * dot_y ** 2)

    colors = pdf * param(5, l=3).reshape(-1, active_gaussians, 3)
    return torch.sum(colors, dim=1)

def inv_spherical_logistic_norm(k,b):
    numer = 4 * torch.pi * torch.sinh(k)
    denom = k * (b ** 2 + 2.0 * (b - 1.0) * (torch.cosh(k) - 1.0))
    return denom / numer