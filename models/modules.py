import importlib
from math import floor
from typing import Optional

import numpy as np
import torch
from bayesian_torch.layers import LinearReparameterization
from torch import nn
from torch.nn import functional as F

from util.vis_io import get_grid_tensor


if torch.cuda.is_available():
    try:
        import tinycudann as tcnn
    except ImportError as e:
        print(
            f"Error: {e}! "
            "Please install tinycudann by: "
            "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
        )
        # exit()

def get_class_by_name(name):
    if '.' in name:
        module_name, class_name = name.rsplit('.', 1)
        print(module_name, class_name)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        except (ImportError, AttributeError) as e:
            print(f"Error: {e}")
            return None
    else:
        if name in globals():
            return globals()[name]
        else:
            print(f"Error: {name} not found in globals")
            return None

def get_model_by_name(name, param) -> nn.Module:
    cls = get_class_by_name(name)
    if cls:
        return cls(**param)
    else:
        raise ValueError(f"get_model_by_name didn't find the class {name}")

def interpolate_line_1d(line, t: torch.Tensor):
    """
    line: (D, L, C) tensor object
    t: (B,) normalized to [0,1]
    returns: (B, L, C)
    """
    D = line.shape[0]
    t_scaled = t * (D - 1)
    idx0 = torch.floor(t_scaled).long()
    idx1 = torch.clamp(idx0 + 1., min=0., max=D - 1).long()
    w = (t_scaled - idx0).view(-1, *[1] * (line.ndim - 1))
    v0 = line[idx0]          # (B, L, C)
    v1 = line[idx1]          # (B, L, C)
    return torch.lerp(v0, v1, w.to(v0.dtype))

def interpolate_line_1d_parallel(line, t: torch.Tensor):
    """
    line: (N, D, C) feature line (batched)
    t: (B, N) normalized to [0,1]
    returns: (B, N, C)
    """
    N, D, C = line.shape
    
    # t = t.clamp(0, 1)
    t_scaled = t * (D - 1)  # (B, N)
    idx0 = torch.floor(t_scaled).long()  # (B, N)
    idx1 = torch.clamp(idx0 + 1., min=0., max=D - 1).long()  # (B, N)
    w = (t_scaled - idx0).unsqueeze(-1)  # (B, N, 1)
    
    # Create indices for advanced indexing
    N_idx = torch.arange(N, device=line.device).unsqueeze(0).expand(t.shape[0], -1)  # (B, N)
    
    v0 = line[N_idx, idx0]  # (B, N, C)
    v1 = line[N_idx, idx1]  # (B, N, C)
    
    return torch.lerp(v0, v1, w.to(v0.dtype))

def get_nested_attr(obj, attr):
    attributes = attr.split('.')
    for attribute in attributes:
        obj = getattr(obj, attribute)
    return obj

def init_param(param, init, init_kwargs):
    if init == 'normal':
        if 'mean' not in init_kwargs:
            init_kwargs['mean'] = 0
            init_kwargs['std'] = 0.001
        nn.init.normal_(param, **init_kwargs)
    elif init == 'uniform':
        if 'a' not in init_kwargs:
            init_kwargs['a'] = -0.001
            init_kwargs['b'] = 0.001
        nn.init.uniform_(param, **init_kwargs)
    else:
        nn.init.zeros_(param)
        
def fwd_hdinr(model, x):
    x, cond, cond_i = x
    return model(x, cond)

def fwd_inrsurrogate(model, x):
    x, cond, cond_i = x
    B, N, _ = x.shape
    cond_rep = cond[:,None].repeat(1, x.shape[1], 1)
    x = torch.cat([x, cond_rep], dim=-1)
    x = x.view(-1, x.shape[-1])
    return model(x).view(B, N, -1)


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))
    
trunc_exp = _TruncExp.apply

class PositionalEncoding(nn.Module):
    # output dim multiplier: 2*n_frequencies
    def __init__(self, n_frequencies):
        super(PositionalEncoding, self).__init__()
        coeff = torch.pi * 2 ** torch.arange(n_frequencies)
        self.register_buffer("coeff", coeff)

    def forward(self, x):
        # x: (n, c)
        s = torch.sin(x[..., None] * self.coeff)
        c = torch.cos(x[..., None] * self.coeff)
        x_pe = torch.concat([s, c], dim=-1).view(*x.shape[:-1], -1)  # (n, c*n_frequencies*2)
        return x_pe

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=False)
        
    def forward(self, x):
        x = self.fc3(F.silu(self.fc1(x)) * self.fc2(x))
        return x

class ReGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation='ELU'):
        super(ReGLU, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else globals()[activation]()
        
    def forward(self, x):
        x = self.fc3(self.activation(self.fc1(x)) * self.fc2(x))
        return x
    
class CondLatentLines(nn.Module):
    def __init__(
        self, cond_dim, line_dims, latent_shape,
        latent_init='uniform', latent_init_kwargs={'a':-0.001, 'b':0.001},
        refiner_residual=True,
    ):
        super(CondLatentLines, self).__init__()
        self.refiner_residual = refiner_residual
        self.cond_dim = cond_dim
        self.latent_shape = latent_shape
        self.line_dimid = list(range(cond_dim))
        self.line_dims = line_dims
        self.is_eq_dim = len(set(line_dims)) == 1
        self.highest_dim = max(line_dims)
        self.refined = False
        self.refine_fdim = latent_shape[0] * cond_dim

        if not self.is_eq_dim:
            highest_res_x = get_grid_tensor([0], [1], [self.highest_dim])
            highest_res_x = highest_res_x.flatten()
            self.register_buffer('highest_res_x', highest_res_x)

        self.lines = torch.nn.ParameterList()
        for dim in self.line_dims:
            line = torch.nn.Parameter(
                torch.Tensor(dim, *latent_shape),
                requires_grad=True
            )
            init_param(line, latent_init, latent_init_kwargs)
            self.lines.append(line)

    def _get_upsampled_lines(self):
        """Get lines upsampled to the highest resolution and concatenated."""
        if self.is_eq_dim:
            return torch.cat(list(self.lines), dim=-1)

        high_res_lines = []
        for line in self.lines:
            if line.shape[0] == self.highest_dim:
                high_res_lines.append(line)
            else:
                upsampled_line = interpolate_line_1d(line, self.highest_res_x)
                high_res_lines.append(upsampled_line)
        return torch.cat(high_res_lines, dim=-1)

    def refine_transforms(self, transforms):
        """Refines the latent lines by upsampling, applying transformations, and adding a residual."""
        high_res_lines = self._get_upsampled_lines()
        
        # Refine the combined high-resolution lines
        self.refined_latent = transforms(high_res_lines)
        
        # Add residual connection
        if self.refiner_residual:
            self.refined_latent += high_res_lines
            
        # Reshape back to (cond_dim, high_res, C)
        self.refined_latent = self.refined_latent.view(self.highest_dim, -1, *self.latent_shape).permute(1, 0, 2)
        self.refined = True

    def _refined_forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the refined, combined latent representation.
        cond: (B, C) -> out: (B, C*F)
        """
        param_feats = interpolate_line_1d_parallel(self.refined_latent, cond)
        return param_feats.flatten(1)

    def _default_forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the original, separate latent lines.
        cond: (B, C) -> out: (B, L, C*F)
        """
        param_feats = []
        for i, dim_id in enumerate(self.line_dimid):
            p1d = cond[:, dim_id]
            f1d = interpolate_line_1d(self.lines[i], p1d)
            param_feats.append(f1d)
        
        return torch.cat(param_feats, dim=1)
    
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Dispatches to the appropriate forward method based on whether the model has been refined.
        """
        if self.refined:
            return self._refined_forward(cond)
        return self._default_forward(cond)


class MultiCondLatentLines(nn.Module):
    """
    Multiple CondLatentLines whose dimensions are specified in a list: line_dims_list
    """
    def __init__(self, cond_dim, line_dims_list: list, latent_shape,
                 latent_init='uniform', latent_init_kwargs={'a':-0.001, 'b':0.001},
                 pre_refiner_proj_kwargs=None,
                 ) -> None:
        super().__init__()
        self.line_dims_list = line_dims_list
        self.num_scales = len(line_dims_list)
        self.cond_dim = cond_dim
        self.latent_shape = latent_shape
        self.refined = False
        self.refine_fdim = latent_shape[0] * self.num_scales * cond_dim

        # Create a CondLatentLines for each set of line dimensions
        self.latent_lines_list = nn.ModuleList([])
        for line_dims in line_dims_list:
            latent_lines = CondLatentLines(
                cond_dim=cond_dim,
                line_dims=line_dims,
                latent_shape=latent_shape,
                latent_init=latent_init,
                latent_init_kwargs=latent_init_kwargs,
                refiner_residual=False, # Residual is handled in this class
            )
            self.latent_lines_list.append(latent_lines)

        # Find the highest resolution for combined refining
        all_dims = torch.tensor(line_dims_list)
        self.highest_line_dims = torch.max(all_dims, dim=0).values
        self.highest_dim = torch.max(self.highest_line_dims).item()

        highest_res_x_per_dim = torch.nn.ParameterList([])
        for d in range(cond_dim):
            highest_res_x = get_grid_tensor([0], [1], [self.highest_line_dims[d]])
            highest_res_x = nn.Parameter(highest_res_x.flatten(), requires_grad=False)
            highest_res_x_per_dim.append(highest_res_x)
        self.highest_res_x_per_dim = highest_res_x_per_dim
        highest_res_x = get_grid_tensor([0], [1], [self.highest_dim]).flatten()
        self.register_buffer('highest_res_x', highest_res_x)

        if pre_refiner_proj_kwargs is not None:
            self.pre_refiner_proj = get_model_by_name(pre_refiner_proj_kwargs['type'], pre_refiner_proj_kwargs['param'])

    def _get_unified_lines(self):
        """
        Unifies line sets by upsampling each to a common high resolution and concatenating features.
        """
        # Stage 1: Upsample lines in each dimension to the max resolution for that dimension
        max_scale_lines = [[] for _ in range(self.cond_dim)]
        for lines in self.latent_lines_list:
            for j, (target_res, current_res) in enumerate(zip(self.highest_line_dims, lines.line_dims)):
                if target_res == current_res:
                    line_data = lines.lines[j]
                else:
                    line_data = interpolate_line_1d(lines.lines[j], self.highest_res_x_per_dim[j])
                max_scale_lines[j].append(line_data)
        
        max_scale_lines = [torch.cat(dim_lines, dim=-1) for dim_lines in max_scale_lines]
        
        # Stage 2: Upsample all dimension lines to the single highest resolution
        unified_lines = []
        for max_scale_line in max_scale_lines:
            if max_scale_line.shape[0] == self.highest_dim:
                unified_lines.append(max_scale_line)
            else:
                upsampled_line = interpolate_line_1d(max_scale_line, self.highest_res_x)
                unified_lines.append(upsampled_line)
        
        return torch.cat(unified_lines, dim=-1)

    def refine_transforms(self, transforms, **kwargs):
        """
        Refines the latent lines by unifying them, applying transforms, and adding a residual.
        """
        lines_to_refine = self._get_unified_lines()

        if hasattr(self, 'pre_refiner_proj'):
            lines_to_refine = self.pre_refiner_proj(lines_to_refine)

        refined_lines = transforms(lines_to_refine)
        
        refined_lines = refined_lines + lines_to_refine
        
        self.refined_lines = refined_lines.view(self.highest_dim, self.cond_dim, -1).permute(1, 0, 2)
        self.refined = True

    def _default_forward(self, cond):
        features = [g(cond) for g in self.latent_lines_list]
        return torch.cat(features, dim=-1)

    def _refined_forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the unified and refined latent representation.
        """
        param_feats = interpolate_line_1d_parallel(self.refined_lines, cond)
        return param_feats.flatten(1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if self.refined:
            return self._refined_forward(cond)
        return self._default_forward(cond)




# dense feature grid initialize with Uniform(-1e-4, 1e-4) as instant-NGP paper
class DenseFeatGrid(nn.Module):
    """
    Represents a dense, regular feature grid.

    Args:
        grid_shape (tuple): The resolution of the grid (e.g., (X, Y, Z)).
        num_feat (int): The number of features per grid point.
        init_method (str): Initialization method for grid features.
        init_kwargs (dict): Arguments for the initialization method.
        refine_highest_res (tuple, optional): If provided, the grid will be upsampled
            to this resolution before applying refinement transformations.
        prerefine_feat_up_kwargs (dict, optional): If provided, specifies positional
            encoding to be applied to features before refinement.
    """
    def __init__(
        self,
        grid_shape,
        num_feat,
        init_method='uniform',
        init_kwargs={'a': -0.001, 'b': 0.001},
        refine_highest_res=None,
        prerefine_feat_up_kwargs=None,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.num_feat = num_feat
        self.refine_highest_res = refine_highest_res
        self.prerefine_feat_up_kwargs = prerefine_feat_up_kwargs
        self.refined = False

        self.feature_grid = nn.Parameter(
            torch.Tensor(1, num_feat, *self.grid_shape),
            requires_grad=True
        )
        init_param(self.feature_grid, init_method, init_kwargs)

        if self.prerefine_feat_up_kwargs:
            self.prerefine_pe = PositionalEncoding(prerefine_feat_up_kwargs['g'])

        if self.refine_highest_res:
            coords = get_grid_tensor([-1] * len(self.grid_shape), [1] * len(self.grid_shape), self.refine_highest_res)
            self.register_buffer('x_highest_res', coords.view(-1, len(self.grid_shape)))

    def _get_grid_for_refinement(self):
        """Prepares the grid features for the refinement process."""
        if self.refine_highest_res:
            # Upsample grid to the target high resolution
            grid_feats = F.grid_sample(
                self.feature_grid,
                self.x_highest_res.view([1] * self.x_highest_res.shape[-1] + list(self.x_highest_res.shape)),
                mode='bilinear',
                align_corners=True
            ).view(self.num_feat, -1).T
        else:
            # Use the original grid features directly
            grid_feats = self.feature_grid.squeeze(0).flatten(1).T

        if hasattr(self, 'prerefine_pe'):
            grid_feats = self.prerefine_pe(grid_feats)

        return grid_feats

    def refine_transforms(self, transforms, **kwargs):
        """
        Applies a transformation model to the grid features and stores the result.
        This is often used to "bake" a decoder MLP into the grid for faster inference.
        """
        grid_to_refine = self._get_grid_for_refinement()
        
        refined_features = transforms(grid_to_refine) + grid_to_refine
        
        output_shape = self.refine_highest_res or self.grid_shape
        num_output_features = refined_features.shape[-1]
        
        self.refined_grid = refined_features.T.view(1, num_output_features, *output_shape)
        self.refined = True

    def _sample_grid(self, x: torch.Tensor, use_refined: bool) -> torch.Tensor:
        """Samples features from the specified grid at coordinates x."""
        grid = self.refined_grid if self.refined and use_refined else self.feature_grid
        
        # Reshape coordinates for grid_sample: (1, 1, ..., N, D)
        sample_coords = x.view([1] * x.shape[-1] + list(x.shape))
        
        # print(grid.shape, sample_coords.shape, x.shape)
        feats = F.grid_sample(
            grid,
            sample_coords,
            mode='bilinear',
            align_corners=True
        )
        # Reshape from (1, C, 1, ..., N) to (N, C)
        feats = feats.view(grid.shape[1], -1).T
        return feats

    def forward(self, x: torch.Tensor, use_refined: bool = True) -> torch.Tensor:
        """
        Forward pass. Samples features from the grid at the given coordinates.
        By default, it uses the refined grid if available.
        """
        return self._sample_grid(x, use_refined)


class MultiDenseFeatGrid(nn.Module):
    """
    Manages multiple dense feature grids of varying resolutions.
    It can fuse features from these grids by concatenation and apply a refinement
    ("baking") process to create a single high-resolution grid for efficient inference.

    Args:
        grid_shapes (list): A list of shapes for each dense grid, e.g., [(64, 64), (128, 128)].
        num_feat (int): The number of features per grid point for each grid.
        refine_highest_res (tuple, optional): The target resolution for the combined refined grid.
            If None, it defaults to the resolution of the last grid in `grid_shapes`.
        prerefine_feat_up_kwargs (dict, optional): Configuration for feature enhancement
            (e.g., positional encoding) before the refinement transform is applied.
    """
    def __init__(
        self,
        grid_shapes: list,
        num_feat: int,
        refine_highest_res: Optional[tuple] = None,
        prerefine_feat_up_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.grid_shapes = grid_shapes
        self.num_feat = num_feat
        self.refined = False
        self.x_dims = len(grid_shapes[0])

        self.feature_grids = nn.ModuleList(
            [DenseFeatGrid(shape, num_feat) for shape in grid_shapes]
        )

        self.refine_highest_res = refine_highest_res or self.grid_shapes[-1]
        coords = get_grid_tensor([-1] * self.x_dims, [1] * self.x_dims, self.refine_highest_res)
        self.register_buffer('high_res_coords', coords.view(-1, self.x_dims))

        if prerefine_feat_up_kwargs:
            self.prerefine_pe = PositionalEncoding(prerefine_feat_up_kwargs['g'])

    def _get_combined_features_for_refinement(self) -> torch.Tensor:
        """Upsamples all grids to the highest resolution and concatenates their features."""
        high_res_feats = []
        for grid in self.feature_grids:
            # Sample each grid at the target high-resolution coordinates
            # use_refined=False ensures we are using the original, un-baked grid data.
            sampled_feats = grid(self.high_res_coords, use_refined=False)
            high_res_feats.append(sampled_feats)
        return torch.cat(high_res_feats, dim=-1)

    def refine_transforms(self, transforms, **kwargs):
        """
        Applies transformations to "bake" features into a single high-resolution grid.
        """
        # 1. Get features from all grids, upsampled to the target resolution
        features_to_refine = self._get_combined_features_for_refinement()

        if hasattr(self, 'prerefine_pe'):
            features_to_refine = self.prerefine_pe(features_to_refine)

        refined_features = transforms(features_to_refine)
        refined_features = refined_features + features_to_refine

        num_output_features = refined_features.shape[-1]
        self.refined_grid = refined_features.T.view(1, num_output_features, *self.refine_highest_res)
        self.refined = True

    def _refined_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Samples features from the single, combined, high-resolution refined grid."""
        sample_coords = x.view([1] * x.shape[-1] + list(x.shape))
        
        return F.grid_sample(
            self.refined_grid,
            sample_coords, # Reshape coords for grid_sample
            mode='bilinear',
            align_corners=True
        ).view(self.refined_grid.shape[1], -1).T  # Reshape to (N, C)

    def _default_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Samples features from each grid individually and concatenates the results."""
        features = [grid(x) for grid in self.feature_grids]
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Uses the refined grid if available, otherwise samples from
        the original grids and concatenates the features.
        """
        if self.refined:
            return self._refined_forward(x)
        else:
            return self._default_forward(x)


class TriPlane(nn.Module):
    def __init__(self, plane_res, num_feat, prerefine_feat_up_kwargs=None, refine_highest_res=None) -> None:
        super().__init__()
        self.refined = False
        self.plane_res = plane_res
        self.num_feat = num_feat
        self.planes = torch.nn.Parameter(
            torch.Tensor(3, num_feat, plane_res[0], plane_res[1]),
            requires_grad=True
        )
        self.plane_dimid = [(1, 2), (0, 2), (0, 1)] # corresponding to xy, xz, yz
        # initialize with Uniform(-1e-4, 1e-4)
        torch.nn.init.uniform_(self.planes, a=-0.001, b=0.001)
        
        self.prerefine_feat_up_kwargs = prerefine_feat_up_kwargs
        if prerefine_feat_up_kwargs != None:
            self.prerefine_PE_grid = PositionalEncoding(prerefine_feat_up_kwargs['g'])
            
        if refine_highest_res is None:
            refine_highest_res = self.plane_res
            self.if_interp_last_grid = False
        self.refine_highest_res = refine_highest_res
        high_res_x = get_grid_tensor([-1, -1], [1, 1], self.refine_highest_res)
        high_res_x = high_res_x.view(-1, 2)
        self.register_buffer('high_res_x', high_res_x)
        
        
    def refine_transforms(self, transforms, **kwargs):
        # Interpolate each plane to high resolution
        if self.refine_highest_res[0] != self.plane_res[0]:
            x = self.high_res_x
            planes = self.planes
        
            x0 = x[..., self.plane_dimid[0]]
            f2d_0 = F.grid_sample(planes[[0]],
                                x0.view(([1]*x0.shape[-1]) + list(x0.shape)),
                                mode='bilinear', align_corners=True).squeeze().T
            x1 = x[..., self.plane_dimid[1]]
            f2d_1 = F.grid_sample(planes[[1]],
                                x1.view(([1]*x1.shape[-1]) + list(x1.shape)),
                                mode='bilinear', align_corners=True).squeeze().T
            x2 = x[..., self.plane_dimid[2]]
            f2d_2 = F.grid_sample(planes[[2]],
                                x2.view(([1]*x2.shape[-1]) + list(x2.shape)),
                                mode='bilinear', align_corners=True).squeeze().T
        else:
            f2d_0 = self.planes[0].flatten(1).T
            f2d_1 = self.planes[1].flatten(1).T
            f2d_2 = self.planes[2].flatten(1).T
        # prerefine ops
        if self.prerefine_feat_up_kwargs is not None:
            f2d_0 = self.prerefine_PE_grid(f2d_0)
            f2d_1 = self.prerefine_PE_grid(f2d_1)
            f2d_2 = self.prerefine_PE_grid(f2d_2)

        # Apply transforms
        self.refined_triplane = [transforms(f2d_0, midx=0), transforms(f2d_1, midx=1), transforms(f2d_2, midx=2)]
        # Residual connection if enabled

        self.refined_triplane = torch.stack([r + h for r, h in zip(self.refined_triplane, [f2d_0, f2d_1, f2d_2])], dim=0)

        # Reshape back to triplane format (3 planes concatenated)
        # Assuming output should be (N, 3*num_feat) -> split into 3 planes
        feat_per_plane = self.refined_triplane.shape[-1]
        self.refined_triplane = self.refined_triplane.permute(0, 2, 1).view(3, feat_per_plane, *self.refine_highest_res)
        self.refined = True

    def forward(self, x: torch.Tensor, used_refined=True) -> torch.Tensor:
        if self.refined and used_refined:
            planes = self.refined_triplane
        else:
            planes = self.planes
        feats = 0
        for i, dimid in enumerate(self.plane_dimid):
            xi = x[..., dimid]
            f2d = F.grid_sample(planes[[i]],
                xi.view(([1]*xi.shape[-1]) + list(xi.shape)),
                mode='bilinear', align_corners=True)
            feats = feats + f2d.view(f2d.shape[1], -1).permute(1, 0)
        return feats
    
class DRRGrid(nn.Module):
    def __init__(self,
                 grid_type,
                 grid_kwargs:dict,
                 refiner_kwargs:dict=None,
                 if_refine=False
                 ) -> None:
        super().__init__()
        self.if_refine = if_refine
        self.grid = get_model_by_name(grid_type, grid_kwargs)
        if refiner_kwargs is not None:
            self.sp = get_model_by_name(refiner_kwargs["type"], refiner_kwargs["param"])

    def refine_spatial(self):
        '''
        refine the spatial encoder to speed up inference.
        '''
        assert hasattr(self, 'sp'), "SR not defined."
        self.grid.refine_transforms(self.sp)
        self.have_refined_spatial = True

    def forward(self, x: torch.Tensor):
        if self.if_refine and (self.training or not self.have_refined_spatial) and hasattr(self, 'sp'):
            self.refine_spatial()
        return self.grid(x)

class HashGrid(nn.Module):
    """
    A wrapper around the tiny-cuda-nn HashGrid encoding that makes it a clean
    and reusable nn.Module.

    Args:
        n_input_dims (int): The dimensionality of the input coordinates (e.g., 2 for 2D, 3 for 3D).
        n_levels (int): The number of levels in the multi-resolution grid.
        n_features_per_level (int): The number of features stored at each grid level.
        log2_hashmap_size (int): Log base 2 of the hash table size.
        base_resolution (int): The resolution of the coarsest grid level.
        max_resolution (int): The resolution of the finest grid level. If None, it is
                              calculated based on other parameters to match the Instant-NGP paper.
    """
    def __init__(
        self,
        n_input_dims: int,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: Optional[int] = 512,
    ):
        super().__init__()
        self.n_input_dims = n_input_dims
        
        # Calculate the geometric growth factor `per_level_scale`
        if n_levels > 1:
            per_level_scale = np.exp(
                (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
            )
        else:
            # If there's only one level, the scale is irrelevant but must be set.
            per_level_scale = 1.0

        # The configuration dictionary for the tcnn.Encoding module
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        # The core encoding module from tiny-cuda-nn
        self.encoding = tcnn.Encoding(
            n_input_dims=self.n_input_dims,
            encoding_config=encoding_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoding(x)


class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=2, activation='ELU', use_norm=True, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        if use_norm:
            self.GLU_layer = nn.Sequential(
                RMSNorm(in_dim),
                ReGLU(in_dim, floor(in_dim * hidden_mult), in_dim, activation)
            )
        else:
            self.GLU_layer = ReGLU(in_dim, floor(in_dim * hidden_mult), in_dim, activation)

        if out_dim != in_dim:
            self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x + self.GLU_layer(x) if self.skip_connection else self.GLU_layer(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x

class GLUFeatMLP(nn.Module):
    def __init__(self, feat_dim, out_dim, num_layers, hidden_mult=2, block_activation='ELU', use_norm=True,
                 skip_connection=True):
        super().__init__()
        
        self.net = nn.Sequential(
            *[GLUBlock(feat_dim, feat_dim, hidden_mult, block_activation, use_norm, skip_connection) for _ in range(num_layers - 1)],
            GLUBlock(feat_dim, out_dim, hidden_mult, block_activation, use_norm, skip_connection)
        )

    def forward(self, x):
        return self.net(x)
    
class GLUMLP(nn.Module):
    def __init__(self, in_dim, feat_dim, out_dim, num_layers, hidden_mult=2, block_activation='ELU', use_norm=True,
                 skip_connection=True):
        super().__init__()
        layers = []
        
        # First layer: in_dim -> feat_dim
        layers.append(nn.Linear(in_dim, feat_dim))
        
        # Hidden layers: feat_dim -> feat_dim
        for _ in range(num_layers):
            layers.append(GLUBlock(feat_dim, feat_dim, hidden_mult, block_activation, use_norm, skip_connection))
        
        # Final layer: feat_dim -> out_dim
        if feat_dim != out_dim:
            layers.append(nn.Linear(feat_dim, out_dim))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GLUMLPList(nn.Module):
    def __init__(self, num_models, in_dim, feat_dim, out_dim, num_layers, hidden_mult=2, block_activation='ELU', use_norm=False,
                 skip_connection=True):
        super().__init__()
        self.net = nn.ModuleList()
        for _ in range(num_models):
            model = GLUMLP(in_dim, feat_dim, out_dim, num_layers, hidden_mult, block_activation, use_norm, skip_connection)
            self.net.append(model)

    def forward(self, x, midx):
        return self.net[midx](x)

class ModelList(nn.Module):
    def __init__(self, num_models, model_class, model_kwargs):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(num_models):
            mkwargs = model_kwargs if not isinstance(model_kwargs, (list, tuple)) else model_kwargs[i]
            model = get_model_by_name(model_class, mkwargs)
            self.net.append(model)

    def forward(self, x, midx, *args, **kwargs):
        return self.net[midx](x, *args, **kwargs)

class FeatMLP(nn.Module):
    def __init__(self, feat_dim, num_layers, out_dim, activation='ReLU', out_activation='Identity'):
        super().__init__()
        self.register_buffer('output_noise', torch.tensor(0.03)) # made compatible to MLP
        in_out_dims = [[feat_dim, feat_dim] for _ in range(num_layers)]
        in_out_dims[-1][1] = out_dim
        self.net = nn.Sequential()
        for i in range(num_layers):
            activation = activation if i < num_layers - 1 else out_activation
            self.net.append(LinearLayer(in_out_dims[i][0], in_out_dims[i][1], activation))
    
    def forward(self, x):
        return self.net(x)
    
class StackedModels(nn.Module):
    def __init__(self, num_models, model_class, model_kwargs, in_proj_kwargs=None, out_proj_kwargs=None):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_models):
            mkwargs = model_kwargs if not isinstance(model_kwargs, (list, tuple)) else model_kwargs[i]
            model = get_model_by_name(model_class, mkwargs)
            self.net.append(model)
        
        if in_proj_kwargs is not None:
            self.in_proj = get_model_by_name(in_proj_kwargs['type'], in_proj_kwargs['param'])
        if out_proj_kwargs is not None:
            self.out_proj = get_model_by_name(out_proj_kwargs['type'], out_proj_kwargs['param'])

    def forward(self, x, *args, **kwargs):
        if hasattr(self, 'in_proj'):
            x = self.in_proj(x)
        x = self.net(x, *args, **kwargs)
        if hasattr(self, 'out_proj'):
            x = self.out_proj(x)
        return x

class MLP(nn.Module):
    def __init__(
        self, in_dim, feat_dim, feat_activation, num_layers, out_dim,
        out_activation="Identity", use_layer_norm=False, output_noise=0.03, dropout=0.,
        use_aleatoric=False, use_mse=True, init_method=None,
        is_mfvi=False,
        prior_mean=0.,
        prior_variance=1.,
        posterior_mu_init=0.,
        posterior_rho_init=-3,
    ) -> None:
        super().__init__()
        self.use_mse = use_mse
        self.use_aleatoric = use_aleatoric
        self.register_buffer('output_noise', torch.tensor(output_noise))
        self.dropout = dropout
        
        if feat_activation == 'Sine':
            layer_class = SineLayer
        else:
            layer_class = LinearLayer
            
        self.net = []
        self.net.append(
            layer_class(in_dim, feat_dim, activation=feat_activation, is_first=True, dropout=dropout,use_layer_norm=use_layer_norm,
                        is_mfvi=is_mfvi,
                        prior_mean=prior_mean,
                        prior_variance=prior_variance,
                        posterior_mu_init=posterior_mu_init,
                        posterior_rho_init=posterior_rho_init)
        )
        
        for i in range(num_layers):
            self.net.append(layer_class(feat_dim, feat_dim, activation=feat_activation, is_first=False, dropout=dropout, use_layer_norm=use_layer_norm,
                            is_mfvi=is_mfvi,
                            prior_mean=prior_mean,
                            prior_variance=prior_variance,
                            posterior_mu_init=posterior_mu_init,
                            posterior_rho_init=posterior_rho_init)
            )

        self.net.append(LinearLayer(feat_dim, out_dim, out_activation, use_layer_norm=False,
                                    is_mfvi=is_mfvi,
                                    prior_mean=prior_mean,
                                    prior_variance=prior_variance,
                                    posterior_mu_init=posterior_mu_init,
                                    posterior_rho_init=posterior_rho_init)
                        )
        self.net = nn.Sequential(*self.net)


    def get_loss(self, pred, gt):
        loss_list = []
        total_loss = 0
        if not self.use_mse:
            if self.use_aleatoric:
                pred, output_noise = pred
                l1 = torch.mul(torch.exp(-output_noise)/self.lambda_div_loss, (pred-gt)**2)
                l2 = output_noise
            else:
                output_noise = self.output_noise
                l1 = torch.mul(1/output_noise, (pred-gt)**2)
                l2 = torch.log(output_noise)
            likelihood_data_loss = .5*(l1 + l2)
            likelihood_data_loss = likelihood_data_loss.mean()
            loss_list.append({'name':'nll', 'value':likelihood_data_loss})
            total_loss += likelihood_data_loss
        else:
            likelihood_data_loss = F.mse_loss(pred, gt)
            loss_list.append({'name':'mse', 'value':likelihood_data_loss})
            total_loss += likelihood_data_loss
            
        for component in loss_list:
            component['ratio'] = (component['value'].detach() / total_loss.detach()).item()
            
        return loss_list, total_loss
    
    def forward(self, coords):
        pred = self.net(coords)
        if self.use_aleatoric:
            pred_noise = self.var_head(pred)
            pred = self.pred_head(pred)
            pred = (pred, pred_noise)
        
        return pred
    
class LinearLayer(nn.Module):
    def __init__(self, in_features,
                out_features,
                activation,
                bias=True,
                dropout=0.,
                is_mfvi=False,
                prior_mean=0.,
                prior_variance=1.,
                posterior_mu_init=0.,
                posterior_rho_init=-3,
                 **kwargs):
        super().__init__()
        self.dropout = dropout

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if is_mfvi:
            self.linear = LinearReparameterization(
                in_features, out_features, bias=bias,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )
            
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else globals()[activation]()
        self.dropout = nn.Dropout(self.dropout)
        self.fwd = nn.ModuleList([self.linear])
        if self.activation is not None and not isinstance(self.activation, nn.Identity):
            self.fwd.append(self.activation)
        if dropout > 0.:
            self.fwd.append(self.dropout)
            
        self.fwd = nn.Sequential(*self.fwd)
    
    def kl_loss(self):
        return self.linear.kl_loss()
    
    def forward(self, x):
        return self.fwd(x)

# Author: Vincent Sitzmann
# Source: https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
#

class Siren(nn.Module):
    def __init__(self, in_dim, feat_dim, num_layers, out_dim, outermost_linear=True, 
                first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_dim, feat_dim,
                        is_first=True, omega_0=first_omega_0))

        for i in range(num_layers):
            self.net.append(SineLayer(feat_dim, feat_dim,
                            is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(feat_dim, out_dim)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / feat_dim) / hidden_omega_0, 
                                np.sqrt(6 / feat_dim) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(feat_dim, out_dim, 
                          is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def get_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Calculates the training loss. This version is simplified to only use MSE loss.
        """
        loss_list = []
        total_loss = torch.tensor(0.0, device=pred.device)
        
        # Simple Mean Squared Error loss
        likelihood_data_loss = F.mse_loss(pred, gt)
        loss_list.append({'name': 'mse', 'value': likelihood_data_loss})
        total_loss += likelihood_data_loss
        
        # Calculate the contribution ratio of each loss component
        if total_loss > 0:
            for component in loss_list:
                component['ratio'] = (component['value'].detach() / total_loss.detach()).item()
                
        return loss_list, total_loss
    
    def forward(self, coords):
        pred = self.net(coords)
        return pred


class SineLayer(nn.Module):
  # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
  
  # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
  # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
  # hyperparameter.
  
  # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
  # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=30., dropout=0., use_norm=False, **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.dropout = dropout

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fwd = nn.Sequential()
        self.fwd.append(self.linear)
        if use_norm:
            self.fwd.append(nn.LayerNorm(out_features))
        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return self.dropout(torch.sin(self.omega_0 * self.fwd(input)))
    