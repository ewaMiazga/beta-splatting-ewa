from functools import partial
import torch

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

