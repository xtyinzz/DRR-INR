import os
from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from time import time

from util.vis_io import load_data, get_random_points, get_grid_tensor
from util.eval_util import idx_flat_to_nd

def knn(query:torch.Tensor, key:torch.Tensor, k=8):
    '''
    query: (N, D)
    key: (M, D)
    '''
    distances = torch.cdist(query, key)
    return torch.topk(distances, k, dim=-1, largest=False, sorted=False)

def eval_online_downsample(ds, fields, scale_factor, device):
    fields = [ds.inverse_transform(f) for f in fields]
    down_fields = []
    for field_i in range(len(fields)):
        down_field = torch.nn.functional.interpolate(
            fields[field_i][None, None].to(device), scale_factor=scale_factor,
            mode='trilinear', align_corners=True
        ).squeeze()
        down_field = ds.transform(down_field).cpu()
        down_fields.append(down_field)
    down_fields = torch.stack(down_fields, dim=0)
    return down_fields

def idw_interpolation_on_values(dists, neighbor_vals, power=1, epsilon=1e-8):
    """
    Performs Inverse Distance Weighting on a set of scalar values.

    Args:
        dists (torch.Tensor): The distances from the query point to the neighbors.
                              Shape: (batch_size, k) or (1, k).
        neighbor_vals (torch.Tensor): The values at the neighbor locations.
                                      Shape: (batch_size, k).
        power (int): The power to raise the inverse distance to.
        epsilon (float): A small value to prevent division by zero.

    Returns:
        torch.Tensor: The interpolated values. Shape: (batch_size,).
    """
    # Inverse of the distance, adding epsilon to avoid division by zero
    inv_dists = 1.0 / (dists + epsilon)
    
    # Calculate weights
    weights = inv_dists.pow(power)
    
    # Normalize weights so they sum to 1
    sum_of_weights = torch.sum(weights, dim=-1, keepdim=True)
    
    # Calculate the weighted sum of neighbor values
    interpolated_vals = torch.sum(neighbor_vals * weights, dim=-1) / sum_of_weights.squeeze(-1)
    # print('IDW INTERP shapes - dists:', dists.shape, 'neighbor_vals:', neighbor_vals.shape, 'weights:', weights.shape, 'interpolated_vals:', interpolated_vals.shape)
    return interpolated_vals

def generate_noise(shape, std, threshold, device='cpu'):
    std = torch.tensor(std, device=device)[None]
    threshold = torch.tensor(threshold, device=device)[None]
    noise = torch.clip(torch.randn(shape, device=device) * std, -threshold, threshold)
    return noise
    
def load_files(data_paths, dims, dim_scale_factor=1.0):
    data_all = []
    scale_mode = 'bilinear' if len(dims) == 2 else 'trilinear'
    for data_path in data_paths:
        data = load_data(data_path, dims)
        data = torch.from_numpy(data).float()
        if dim_scale_factor != 1.0:
            data = torch.nn.functional.interpolate(
                data[None, None], scale_factor=dim_scale_factor, 
                mode=scale_mode, align_corners=True
            )
        data_all.append(data.squeeze())
    data_all = torch.stack(data_all, dim=0)
    return data_all
        
def imp_func(data, minval, maxval, num_bins):
    freq = 0
    nBlocks = 16
    block_size = data.numel() // nBlocks
    for bidx in range(nBlocks):
        block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
        freq += block_freq
    freq = freq.type(torch.double)
    importance = 1. / (freq + 1.e-10)
    
    bin_width = 1.0 / num_bins
    max_binidx_f = float(num_bins-1)
    importance_idx = torch.clamp((data - minval) / bin_width, min=0.0, max=max_binidx_f).type(torch.long)
    return importance, importance_idx

def get_point_importance(data, val_bins, imp_field_max_size, normalize=True):
    
    point_cluster_size = max(data.numel() // imp_field_max_size, 1) # for multinomial, number of categories cannot exceed 2^24

    binned_imp, curr_impidx = imp_func(data, 0.0, 1.0, val_bins)
    curr_sample_weights = binned_imp[curr_impidx].reshape(-1, point_cluster_size).sum(1)
    if normalize:
        curr_sample_weights /= curr_sample_weights.sum()
    return binned_imp, curr_sample_weights, point_cluster_size

def field_imp_sampling(point_imp, point_cluster_size, batch_size):
    rnd_idx = torch.multinomial(point_imp, batch_size, replacement=True)
    rnd_idx = rnd_idx * point_cluster_size
    if point_cluster_size > 1:
        rnd_idx += torch.randint(high=point_cluster_size, size=rnd_idx.shape, device=point_imp.device)
    return rnd_idx

def spherical_to_cartesian(coords):
    """
    Converts spherical coordinates (r, lat, lon) to Cartesian coordinates (x, y, z) with broadcasting.
    
    Args:
        r: Radius (ndarray, shape (m,) or scalar).
        lat_deg: Latitude in degrees (-90 to 90) (ndarray, shape (n,) or scalar).
        lon_deg: Longitude in degrees (0 to 360) (ndarray, shape (p,) or scalar).
    
    Returns:
        x, y, z: Cartesian coordinates with broadcasted shapes.
    """
    r, lon_deg, lat_deg = coords[..., 0], coords[..., 1], coords[..., 2]
    # Convert degrees to radians
    lat_rad = torch.deg2rad(lat_deg)  # Shape: (n,)
    lon_rad = torch.deg2rad(lon_deg)  # Shape: (p,)
    # Compute Cartesian coordinates with broadcasting
    x = r * torch.cos(lat_rad) * torch.cos(lon_rad)  # Shape: (m, n, p)
    y = r * torch.cos(lat_rad) * torch.sin(lon_rad)  # Shape: (m, n, p)
    z = r * torch.sin(lat_rad)                      # Shape: (m, n, p)
    return torch.stack([x, y, z], axis=-1)

class HDFieldDataset(Dataset):
    """
    High-dimensional scalar field dataset with conditioning parameters.
    
    Args:
        data_dir (str): Directory containing scalar field data files.
        cond_path (str): Path to the conditioning parameters (e.g., time, camera position).
        cond_batch (int): Number of conditioning parameter samples per batch.
        spatial_log2batch (int): Log2 of the spatial batch size.
        dims (np.ndarray): Dimensions of the scalar field [xdim, ydim, zdim].
        train_num (int): Number of training samples. If None, all samples are used.
        use_logdata (bool): Whether to apply logarithmic transformation to the data.
        device (str or torch.device): Device to load tensors ('cpu' or 'cuda').
        
    In-device: dims, cond
    """
    def __init__(
        self,
        data_dir: str,
        cond_path: str,
        cond_batch: int = 2,
        spatial_log2batch: int = 19,
        dims: np.ndarray = np.array([512, 512, 512]),
        cond_split_path: str = None,
        train_num: int = None,
        use_logdata: bool = False,
        device: str = None,
        data_min: float=None,
        data_max: float=None,
        data_mean: float=None,
        data_std: float=None,
        impsmp_cond_path: str = None,
        impsmp_coord_path: str = None,
        use_noise: bool=False,
        x_noise_std: float=0,
        x_noise_threshold: float=0,
        cond_noise_std: float=0,
        cond_noise_threshold: float=0,
        cond_noise_idw_knn: int=8,
        noise_interp_value: bool=False,
        # spatial downsampling
        dim_scale_factor: float=1.0,
        invert_coords: bool=True,
    ) -> None:
        super().__init__()

        # Device setup
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Initialize variables
        self.data_dir = data_dir
        self.cond_path = cond_path
        self.cond_split_path = cond_split_path
        self.dims = torch.tensor(dims)
        self.dims_orig = torch.tensor(dims)
        self.use_logdata = use_logdata
        self.cond_batch = cond_batch
        self.spatial_batch = 1 << spatial_log2batch
        self.train_num = train_num
        self.use_noise = use_noise
        self.x_noise_std = x_noise_std
        self.x_noise_threshold = x_noise_threshold
        self.cond_noise_std = cond_noise_std
        self.cond_noise_threshold = cond_noise_threshold
        self.invert_coords = invert_coords
        self.cond_noise_idw_knn = cond_noise_idw_knn
        self.noise_interp_value = noise_interp_value
        # Additional initializations
        
        self.dim_scale_factor = dim_scale_factor
        self.data_paths = sorted(glob(os.path.join(self.data_dir, '*')))
        self.data = None  # Will be set when data is loaded
        self.cond = None  # Will be loaded from cond_path
        self.cond_idx = None  # To be initialized after loading conditioning data
        self.cond_idx_unseen = []  # Initialized as an empty list
        self.data_shape = torch.Size([*self.dims, 0])  # To hold [xdim, ydim, zdim, num_files]

        # Load the conditioning parameters and data
        self._load_conditioning_parameters()
        self._load_data()

        self.spatial_batch_count = self.dims.prod() // self.spatial_batch
        self.dataset_length = len(self.cond) * self.spatial_batch_count
        # Compute num_dims
        self.num_dims = self.cond.shape[-1] + len(self.dims)

        # Calculate min/max for data normalization

        self.data_min = data_min
        self.data_max = data_max
        self.data_mean = data_mean
        self.data_std = data_std
        # if self.data_min is None or self.data_max is None:
        #     self.set_minmax()
        self.transform_fn = self.std_transform if data_mean is not None and data_std is not None else self.transform
        for i in range(len(self.data)):
            self.data[i] = self.transform_fn(self.data[i])

        # self.transform(self.data)
        print('Transformed data with log=', self.use_logdata,
               'min:', self.data_min, 'max:', self.data_max,
               'mean:', self.data_mean, 'std:', self.data_std)
        # self.data = self.transform(self.data)

        # Update data shape to reflect actual loaded data
        self.data_shape = torch.Size([len(self.data), *self.dims])  # Assuming self.data shape [N, xdim, ydim, zdim]
        
        # if imp info provided, do imp sampling on that level, else uniform random
        
        self.impsmp_conds = None
        if impsmp_cond_path is not None:
            print('Loading cond-level importance sampling parameters...', impsmp_cond_path)
            self.impsmp_conds = torch.from_numpy(np.load(impsmp_cond_path))
            print('pre repeat impsmp_conds shape:', self.impsmp_conds.shape)
            self.impsmp_conds = torch.repeat_interleave(self.impsmp_conds, self.spatial_batch_count, dim=0)
            print('post repeat impsmp_conds shape:', self.impsmp_conds.shape,
                  self.impsmp_conds[self.spatial_batch_count-5:self.spatial_batch_count+5])
        
        self.impsmp_coords = None
        if impsmp_coord_path is not None:
            print('Loading coord-level importance sampling parameters...', impsmp_coord_path)
            impsmp_coord_files = sorted(glob(os.path.join(impsmp_coord_path, '*')))
            impsmp_coord_files = [impsmp_coord_files[i] for i in self.cond_idx]
            
            self.impsmp_coords = []
            for p in impsmp_coord_files:
                imp_field_dict = np.load(p, allow_pickle=True).item()
                self.impsmp_coords.append(imp_field_dict['point_imp'])
            self.impsmp_coords = torch.stack(self.impsmp_coords, axis=0)
            self.impsmp_coord_cluster_size = imp_field_dict['point_cluster_size']
        
        print(f'Dataset loaded and transformed. Shape: {self.data_shape}')
        print(f'\tlog-transform (T/F): {self.use_logdata}')
        print(f'\tmin max: {self.data_min} {self.data_max}')
        print(f'\tmean std: {self.data_mean} {self.data_std}')
        print(f'\ttransformed min max: {self.data.min()} {self.data.max()}')
        print(f'\ttransformed mean std: {self.data.mean()} {self.data.std()}')
        print(f'\tConditioning parameters shape: {self.cond.shape}')
        print(f'\tUnseen conditioning parameters shape: {self.cond_unseen.shape}')
        print(f'\tSpatial batch size: {self.spatial_batch}')
        print(f'\tSpatial batch count per field: {self.spatial_batch_count}')
        print(f'\ttotal number of items: {self.dataset_length}')
        print(f'\ttotal number of batches: {self.dataset_length / self.cond_batch : .2f}')
        print(f'\tbatch/total ratio - member/cond: {self.cond_batch / len(self.cond) * 100:.4f}%')
        print(f'\tbatch/total ratio - coordinates: {self.spatial_batch / self.dims.prod() * 100:.4f}%')
        print(f'\tFinal spatial dimensions: {self.dims}')
        print(f'\tNumber of dimensions: {self.num_dims}')
        print(f'\tData Aug options:')
        print(f'\t\tCoordinate noise std: {self.x_noise_std}')
        print(f'\t\tCoordinate noise threshold: {self.x_noise_threshold}')
        print(f'\t\tConditioning noise std: {self.cond_noise_std}')
        print(f'\t\tConditioning noise threshold: {self.cond_noise_threshold}')
        print(f'\t\tNoise interpolation value: {self.noise_interp_value}',  '\n')

    def _load_conditioning_parameters(self):
        """
        Load the conditioning parameters and indices for training and unseen data.
        """
        cond = torch.from_numpy(np.load(self.cond_path)).float()
        self.data_idx = cond[:, 0].long()  # Indices for corresponding data files
        cond = cond[:, 1:]  # Conditioning features
        
        if self.cond_split_path is not None:
            self.cond_all = cond
            splits = np.load(self.cond_split_path, allow_pickle=True).item()
            self.cond_idx = torch.from_numpy(splits['cond_idx'])
            self.cond_idx_unseen = torch.from_numpy(splits['cond_idx_unseen'])
            self.cond = cond[self.cond_idx].float()
            if len(self.cond_idx_unseen) > 0:
                self.cond_unseen = cond[self.cond_idx_unseen].float()
            else:
                self.cond_unseen = torch.tensor([])
        else:
            self.cond_idx = torch.randperm(len(cond))[:self.train_num]
            self.cond_idx_unseen = torch.tensor([i for i in range(len(cond)) if i not in self.cond_idx])
            self.cond_all = cond
            self.cond = cond[self.cond_idx]
            self.cond_unseen = cond[self.cond_idx_unseen]
        print('Conditioning parameters loaded. Shape:', self.cond.shape, 'unseen:', self.cond_unseen.shape)

    def _load_data(self, dims=None):
        """`
        Load the scalar field data based on the conditioning parameters.
        """

        if dims is None:
            dims = self.dims.tolist()
        start_time = time()

        data_paths = [self.data_paths[self.data_idx[idx]] for idx in self.cond_idx]
        self.data = load_files(data_paths, dims, self.dim_scale_factor)
        if self.data.shape[1] != self.dims[0]:
            self.dims = torch.tensor(self.data.shape[1:1+len(self.dims)])
        end_time = time()
        print(f"Data loading took {end_time - start_time:.2f} seconds")
        print('Training Data loaded. Shape:', self.data.shape, self.dims)

        
    def _load_data_unseen(self, dims=None):
        """
        Load the scalar field data based on the conditioning parameters.
        """
        start_time = time()
        if dims is None:
            dims = self.dims.tolist()

        data_paths = [self.data_paths[self.data_idx[idx]] for idx in self.cond_idx_unseen]
        if len(self.cond_idx_unseen) > 0:
            self.data_unseen = load_files(data_paths, dims)
        else:
            self.data_unseen = torch.tensor([])
        end_time = time()
        print(f"Data loading took {end_time - start_time:.2f} seconds")
        for i in range(len(self.data_unseen)):
            self.data_unseen[i] = self.transform_fn(self.data_unseen[i])
        print('Unseen data loaded and transformed. Shape:', self.data_unseen.shape, 'min:', self.data_unseen.min(), 'max:', self.data_unseen.max())

    def set_minmax(self):
        self.data_min = self.data.min()
        self.data_max = self.data.max()

    @torch.no_grad()
    def transform(self, vals):
        """
        Normalize the scalar field data to [0, 1].
        """
        if self.use_logdata:
            vals = torch.log(vals)
        vals = (vals - self.data_min) / (self.data_max - self.data_min)
        return vals

    @torch.no_grad()
    def std_transform(self, vals):
        """
        Normalize the scalar field data to [0, 1].
        """
        if self.use_logdata:
            vals = torch.log(vals)
        vals = (vals - self.data_mean) / self.data_std
        return vals

    @torch.no_grad()
    def inverse_transform(self, vals):
        """
        Inverse transformation to retrieve original scalar field values.
        """
        vals = vals * (self.data_max - self.data_min) + self.data_min
        if self.use_logdata:
            vals = torch.exp(vals)
        return vals

    def get_all_coords(self):
        """
        Get all spatial coordinates in the dataset.
        Returns:
            torch.Tensor: Normalized spatial coordinates.
        """
        x = get_grid_tensor([-1] * len(self.dims), [1] * len(self.dims), self.dims)
        print('coords, x shape:', x.shape, self.dims)
        x = x.view(-1, len(self.dims))
        return x

    def get_all_coords_unseen(self):
        """
        Get all spatial coordinates in the dataset.
        Returns:
            torch.Tensor: Normalized spatial coordinates.
        """
        x = get_grid_tensor([-1] * len(self.dims), [1] * len(self.dims), self.dims_orig)
        print('unseen coords, x shape:', x.shape, self.dims_orig)
        x = x.view(-1, len(self.dims_orig))
        return x

    def __len__(self):
        return self.dataset_length
    
    def add_noise(self, x, c):
        if self.x_noise_std > 0:
            x = x + generate_noise(
                x.shape, self.x_noise_std, self.x_noise_threshold
            )
        if self.cond_noise_std > 0: 
            c = c + generate_noise(
                c.shape, self.cond_noise_std, self.cond_noise_threshold
            )
        return x, c
    
    def __getitem__(self, i):
        """
        Retrieve a batch of spatial indices, condition, condition index, and values from the dataset.
        This version uses a more efficient data augmentation strategy.
        """
        latent_i = i // self.spatial_batch_count
        original_cond = self.cond[latent_i]

        if self.impsmp_coords is not None:
            imp_field = self.impsmp_coords[latent_i]
            spatial_indices = field_imp_sampling(
                imp_field, self.impsmp_coord_cluster_size, self.spatial_batch
            )
            spatial_indices_int = idx_flat_to_nd(spatial_indices, self.dims)
        else:
            # Get original integer-based spatial indices
            spatial_indices_int = get_random_points(self.spatial_batch, self.dims)

        # Normalize spatial coordinates to the range [-1, 1] for grid_sample
        spatial_indices_norm = spatial_indices_int.float() / (self.dims - 1) * 2 - 1
        final_cond = original_cond
        final_spatial_indices = spatial_indices_norm.flip(-1)  # Flip for grid_sample

        # Case 1: No noise at all. Sample directly and return.
        if self.cond_noise_std == 0. and self.x_noise_std == 0.:
            values = self.data[tuple(spatial_indices_int.T)]
            
        # Case 2: Spatial noise ONLY. Interpolate on the single selected field.
        elif self.cond_noise_std == 0. and self.x_noise_std > 0.:
            noisy_spatial_indices = spatial_indices_norm + generate_noise(
                spatial_indices_norm.shape, self.x_noise_std, self.x_noise_threshold
            )
            noisy_spatial_indices = torch.clamp(noisy_spatial_indices, -1.0, 1.0)
            final_spatial_indices = noisy_spatial_indices.flip(-1)

            if self.noise_interp_value:
                # Select the single field corresponding to the latent index
                field = self.data[latent_i].unsqueeze(0).unsqueeze(0) # Shape: (1, 1, D, H, W)
                
                # Prepare grid for sampling
                grid_coords = final_spatial_indices.view(([1]*final_spatial_indices.shape[-1]) + list(final_spatial_indices.shape))

                # Interpolate on the SINGLE field
                values = F.grid_sample(field, grid_coords, mode='bilinear', align_corners=True).squeeze()
            else:
                # If interpolation is off, sample original location but return noisy coordinates
                values = self.data[tuple(spatial_indices_int.T)]
        # Case 3: Condition noise is present (with or without spatial noise). Use the full kNN pipeline.
        elif self.cond_noise_std > 0.:
            noisy_cond = original_cond + generate_noise(
                original_cond.shape, self.cond_noise_std, self.cond_noise_threshold
            )
            final_cond = noisy_cond
            final_cond = torch.clamp(final_cond, 0, 1.0)
            
            # Spatial noise can also be applied in this case
            noisy_spatial_indices = spatial_indices_norm
            if self.x_noise_std > 0.:
                noisy_spatial_indices = spatial_indices_norm + generate_noise(
                    spatial_indices_norm.shape, self.x_noise_std, self.x_noise_threshold
                )
                noisy_spatial_indices = torch.clamp(noisy_spatial_indices, -1.0, 1.0)
            final_spatial_indices = noisy_spatial_indices.flip(-1)

            if self.noise_interp_value:
                # This is the full, efficient pipeline from our previous discussion
                knn_dist, knn_idx = knn(noisy_cond[None], self.cond, k=self.cond_noise_idw_knn)
                neighbor_fields = self.data[knn_idx.squeeze(0)]
                input_fields = neighbor_fields.unsqueeze(0)
                # Prepare grid for sampling - CORRECTED
                grid_coords = final_spatial_indices.view(([1]*final_spatial_indices.shape[-1]) + list(final_spatial_indices.shape))
                sampled_neighbor_values = F.grid_sample(input_fields, grid_coords, mode='bilinear', align_corners=True)
                neighbor_vals_for_idw = sampled_neighbor_values.squeeze().T
                
                if self.spatial_batch == 1 and neighbor_vals_for_idw.dim() == 1:
                    neighbor_vals_for_idw = neighbor_vals_for_idw.unsqueeze(0)

                values = idw_interpolation_on_values(knn_dist, neighbor_vals_for_idw, power=2)
            else:
                values = self.data[tuple(spatial_indices_int.T)]

        # Fallback for any unhandled logic (e.g., if noise std is set but interp is false)
        else:
            values = self.data[tuple(spatial_indices_int.T)]

        values = values.unsqueeze(-1) if values.dim() == 1 else values
        return final_spatial_indices, final_cond, latent_i, values

class HDCurviFieldDataset(HDFieldDataset):
    
    def __init__(
        self, data_dir, cond_path, cond_batch = 2, spatial_log2batch = 19,
        dims = np.array([512, 512, 512]), cond_split_path = None,
        train_num = None, use_logdata = False, device = None,
        data_min: float=None,
        data_max: float=None,
        data_mean: float=None,
        data_std: float=None,
        impsmp_cond_path: str = None,
        impsmp_coord_path: str = None,
        use_noise: bool=False,
        x_noise_std: float=0,
        x_noise_threshold: float=0,
        cond_noise_std: float=0,
        cond_noise_threshold: float=0,
        coord_path = None,
        coord_transform = None,
        dim_scale_factor: float=1.0,
        invert_coords: bool=True,
    ):
        super().__init__(
            data_dir=data_dir,
            cond_path=cond_path,
            cond_batch=cond_batch,
            spatial_log2batch=spatial_log2batch,
            dims=dims,
            cond_split_path=cond_split_path,
            train_num=train_num,
            use_logdata=use_logdata,
            device=device,
            data_min=data_min,
            data_max=data_max,
            data_mean=data_mean,
            data_std=data_std,
            impsmp_cond_path=impsmp_cond_path,
            impsmp_coord_path=impsmp_coord_path,
            use_noise=use_noise,
            x_noise_std=x_noise_std,
            x_noise_threshold=x_noise_threshold,
            cond_noise_std=cond_noise_std,
            cond_noise_threshold=cond_noise_threshold,
            dim_scale_factor=dim_scale_factor,
            invert_coords=invert_coords
        )
        self.data = self.data.view(self.data.shape[0], -1) # (XYZ, 3)
        self.coords = torch.from_numpy(np.load(coord_path)).float()
        self.coords_orig = self.coords
        if dim_scale_factor != 1.0:
            mode = 'bilinear' if self.coords.shape[-1] == 2 else 'trilinear'
            coords_view = self.coords[None, None].transpose(1, -1).squeeze(-1)
            coords_view = torch.nn.functional.interpolate(
                coords_view, scale_factor=dim_scale_factor,
                mode=mode, align_corners=True
            )
            self.coords = coords_view.unsqueeze(-1).transpose(1, -1).squeeze()
        self.coords = self.coords.view(-1, self.coords.shape[-1]) # (XYZ, 2 or 3)
        self.coords_len = len(self.coords)
        points_per_field = self.dims.prod()
        if self.spatial_batch >= points_per_field:
            self.spatial_batch = points_per_field
            self.spatial_batch_count = 1
        self.set_coord_transform(coord_transform)
    
    def get_all_coords(self):
        # flip to keep spatial dimension order as X,Y,Z in the channel-first (or image DHW convention) format
        return self.coords.flip(-1)

    def get_all_coords_unseen(self):
        # flip to keep spatial dimension order as X,Y,Z in the channel-first (or image DHW convention) format
        return self.coords_orig.flip(-1)
    
    def set_coord_transform(self, coord_transform):
        self.coord_transform = coord_transform
    
    def __getitem__(self, i):
        latent_i = i // self.spatial_batch_count
        spatial_indices = torch.randint(0, self.coords_len-1, (self.spatial_batch,))
        # if sptial_batch_count == 1, then we can use the same spatial indices for all conditions, no random int
        # else, we need to generate new random spatial indices for each spatial dim for each condition
            
        values = self.data[latent_i, spatial_indices]
        # flip to keep spatial dimension order as X,Y,Z in the channel-first (or image DHW convention) format
        spatial_indices = self.coords[spatial_indices].flip(-1)
        if self.coord_transform is not None:
            spatial_indices = self.coord_transform(spatial_indices)
        cond = self.cond[latent_i]
        spatial_indices, cond = self.add_noise(spatial_indices, cond)
        # might add variational coordinate
        return spatial_indices, cond, latent_i, values[...,None]

class HDFieldDatasetEval(Dataset):
    """
    High-dimensional scalar field dataset with conditioning parameters.
    
    In-device: dims, cond
    """
    def __init__(
        self,
        ds: HDFieldDataset,
        split_spatial: str = 'train',
        use_transform: bool = True
    ) -> None:
        super().__init__()

        # Device setup
        self.ds = ds
        # get the testing data coords, conds, and values
        self.x = ds.get_all_coords() if split_spatial == 'train' else ds.get_all_coords_unseen()
        print(f'Eval coords loaded for split_spatial {split_spatial}. Shape:', self.x.shape, 'ds dims', ds.dims)
        self.cond = self.cond_idx = self.data = None
        
        self.transform = ds.transform
        self.inverse_transform = ds.inverse_transform
        self.use_transform = use_transform
        
        # get other attributes
        self.device = ds.device
        self.dims = ds.dims
        
        self.spatial_batch = self.dims.prod()
        self.spatial_batch_count = 1
        self.cond_batch = 1
        
        self.dataset_length = 0

    def __len__(self):
        return self.dataset_length
    
    def set_data(self, cond, cond_idx, data):
        self.cond, self.cond_idx, self.data = cond, cond_idx, data
        if len(self.data) > 0:
            self.data = self.data.view(self.data.shape[0], -1, 1)
            self.dataset_length = len(self.cond) * self.spatial_batch_count
            self.data_shape = torch.Size([*self.dims, len(self.data)])  # Assuming self.data shape [N, xdim, ydim, zdim]
        else:
            print('Empty data provided to HDFieldDatasetEval')
    # # of batches = len(cond_idx) * (dims.prod() // spatial_batch)
    def __getitem__(self, i):
        # total batch count = number_of_conds * number_of_batches_per_data
        # so for given i, the condition index is i // number_of_batches_per_data
        latent_i = i // self.spatial_batch_count
        spatial_i = i % self.spatial_batch_count
        cond = self.cond[latent_i]
        x = self.x[spatial_i*self.spatial_batch : (spatial_i+1)*self.spatial_batch]
        values = self.data[latent_i, spatial_i*self.spatial_batch:(spatial_i+1)*self.spatial_batch]
        
        if self.use_transform:
            values = self.transform(values)
        return x, cond, latent_i, values
    



class FieldDataset(Dataset):
    """
    Represents a dataset for a single scalar field.
    
    Each item from this dataset is a random batch of coordinates from the field
    and their corresponding scalar values.

    Args:
        data_path (str): Path to the single scalar field data file.
        dims (tuple or list): Dimensions of the scalar field (e.g., [512, 512, 512]).
        batch_size (int): The number of coordinate samples per item.
        steps_per_epoch (int): The number of items in one "epoch" of sampling.
        use_logdata (bool): Whether to apply a log transform to the data.
        data_mean (float, optional): Mean for standardization. If None, min-max is used.
        data_std (float, optional): Std dev for standardization. If None, min-max is used.
        x_noise_std (float): Standard deviation of noise to add to coordinates.
        noise_interp_value (bool): If True, interpolates values at noisy coordinate
                                   locations. Otherwise, returns original values.
    """
    def __init__(
        self,
        data_path: str,
        dims: tuple,
        log2batch: int = 19, # log2 of batch size
        use_logdata: bool = False,
        x_noise_std: float = 0.0,
        x_noise_threshold: float = 0.0,
        noise_interp_value: bool = False,
        dims_scale_factor: float = 1.0
    ):
        super().__init__()

        self.data_path = data_path
        self.dims = torch.tensor(dims)
        self.dims_orig = self.dims
        self.batch_size = 1 << log2batch
        self.steps_per_epoch = np.ceil(self.dims.prod().item() / self.batch_size).astype(int)
        self.use_logdata = use_logdata
        self.x_noise_std = x_noise_std
        self.x_noise_threshold = x_noise_threshold
        self.noise_interp_value = noise_interp_value
        self.dim_scale_factor = dims_scale_factor

        
        # Load the single scalar field
        self.data = self._load_data()
        
        # Setup normalization

        self.data_min = self.data.min()
        self.data_max = self.data.max()
        

        self.data = self.transform(self.data)
        print(f"Field normalized with min={self.data_min:.4f}, max={self.data_max:.4f}")
            
        print(f'Field loaded and transformed. Shape: {self.dims}')
        print(f'\tlog-transform (T/F): {self.use_logdata}')
        print(f'\tmin max: {self.data_min} {self.data_max}')
        print(f'\ttransformed min max: {self.data.min()} {self.data.max()}')
        print(f'\ttransformed mean std: {self.data.mean()} {self.data.std()}')
        print(f'\tBatch size: {self.batch_size}')
        print(f'\tSteps per epoch: {self.steps_per_epoch}')
        print(f'\tSpatial dimensions: {self.dims}')
        print(f'\tData Aug options:')
        print(f'\t\tCoordinate noise std: {self.x_noise_std}')
        print(f'\t\tCoordinate noise threshold: {self.x_noise_threshold}')
        print(f'\t\tNoise interpolation value: {self.noise_interp_value}', '\n')
            

    def _load_data(self):
        """Loads and returns the single scalar field as a tensor."""
        field = load_data(self.data_path, self.dims.tolist())
        field = torch.from_numpy(field).float()
        scale_mode = 'bilinear' if len(self.dims) == 2 else 'trilinear'
        if self.dim_scale_factor != 1.0:
            if len(self.dims) == 2 and len(field.shape) == 3:
                field = field.permute(2, 0, 1).unsqueeze(0)
            else:
                field = field[None, None]
            field = torch.nn.functional.interpolate(
                field, scale_factor=self.dim_scale_factor,
                mode=scale_mode, align_corners=True
            ).squeeze()
            
            if len(self.dims) == 2 and len(field.shape) == 3:
                field = field.permute(1, 2, 0)
        if self.dims[0] != field.shape[0]:
            self.dims = torch.tensor(field.shape[:-1])
        print('Loaded field shape:', field.shape, 'Final dims:', self.dims)
        return field
    
    def get_all_coords(self):
        """
        Get all spatial coordinates in the dataset.
        Returns:
            torch.Tensor: Normalized spatial coordinates.
        """
        x = get_grid_tensor([-1] * len(self.dims), [1] * len(self.dims), self.dims)
        print('coords, x shape:', x.shape, self.dims)
        x = x.view(-1, len(self.dims))
        return x

    def get_all_coords_unseen(self):
        """
        Get all spatial coordinates in the dataset.
        Returns:
            torch.Tensor: Normalized spatial coordinates.
        """
        x = get_grid_tensor([-1] * len(self.dims), [1] * len(self.dims), self.dims_orig)
        print('unseen coords, x shape:', x.shape, self.dims_orig)
        x = x.view(-1, len(self.dims_orig))
        return x
    
    
    @torch.no_grad()
    def transform(self, vals):
        """Normalizes data to the [0, 1] range."""
        if self.use_logdata:
            vals = torch.log(vals)
            self.data_min, self.data_max = vals.min(), vals.max()
        return (vals - self.data_min) / (self.data_max - self.data_min)

    def __len__(self):
        """The 'length' of the dataset is the number of random batches per epoch."""
        return self.steps_per_epoch

    def __getitem__(self, idx):
        """
        Returns one item: a batch of random coordinates and their values.
        The index `idx` is ignored as we sample randomly every time.
        """
        spatial_indices_int = get_random_points(self.batch_size, self.dims)
        spatial_indices_norm = (spatial_indices_int.float() / (self.dims - 1)) * 2 - 1
        final_coords = spatial_indices_norm.flip(-1)

        # Handle data augmentation (spatial noise)
        if self.x_noise_std > 0:
            # Add noise to the normalized coordinates
            noise = generate_noise(spatial_indices_norm.shape, self.x_noise_std, self.x_noise_threshold)
            noisy_coords = torch.clamp(spatial_indices_norm + noise, -1.0, 1.0)
            final_coords = noisy_coords # The model will see the noisy VP coordinates

            if self.noise_interp_value:
                # Interpolate values at the noisy locations using F.grid_sample
                grid_coords = noisy_coords.view(([1]*noisy_coords.shape[-1]) + list(noisy_coords.shape))
                
                # Handle both 2D and 3D cases, with or without channels
                field = self.data
                
                # Determine if scalar or vector field based on dimensions
                is_vector_field = len(field.shape) > len(self.dims)
                
                if is_vector_field:
                    field_for_sampling = field.permute(-1, *range(len(self.dims))).unsqueeze(0)  # (1, C, D, H, W) or (1, C, H, W)
                else:
                    field_for_sampling = field.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W) or (1, 1, H, W)
                
                values = F.grid_sample(
                    field_for_sampling, grid_coords, mode='bilinear', align_corners=True
                ).squeeze(0)
                
                if is_vector_field:
                    values = values.permute(*range(1, len(self.dims) + 1), 0)  # (D, H, W, C) or (H, W, C)
                    values = values.view(-1, field.shape[-1])  # Flatten spatial dims, keep channels
                else:
                    values = values.squeeze(0).view(-1)  # Remove channel dimension and flatten
            else:
                values = self.data[tuple(spatial_indices_int.T)]
        else:
            values = self.data[tuple(spatial_indices_int.T)]
        
        values = values.unsqueeze(-1) if values.dim() == 1 else values
        return final_coords, values

class FieldDatasetEval(Dataset):
    """
    High-dimensional scalar field dataset with conditioning parameters.
    
    In-device: dims, cond
    """
    def __init__(
        self,
        ds: FieldDataset,
        split_spatial: str = 'train',
        use_transform: bool = True,
        chunk_size: int = 128**3
    ) -> None:
        super().__init__()

        # Device setup
        self.ds = ds
        # get the testing data coords, and values
        self.x = ds.get_all_coords() if split_spatial == 'train' else ds.get_all_coords_unseen()
        print(f'Eval coords loaded for split_spatial {split_spatial}. Shape:', self.x.shape, 'ds dims', ds.dims)
        self.data = None
        self.chunk_size = chunk_size
        
        self.transform = ds.transform
        self.use_transform = use_transform
        
        # get other attributes
        self.device = ds.device
        self.dims = ds.dims
        
        self.spatial_batch = self.dims.prod()
        
        self.dataset_length = ds.dims.prod().item() // self.chunk_size

    def __len__(self):
        return self.dataset_length
    
    def set_data(self, data):
        self.data = data
        if len(self.data) > 0:
            self.data = self.data.view(-1, 1)
            self.data_shape = torch.Size([*self.dims, len(self.data)])  # Assuming self.data shape [N, xdim, ydim, zdim]
        else:
            print('Empty data provided to ScalarFieldDatasetEval')
            
    def __getitem__(self, i):
        # total batch count = number_of_chunks
        # so for given i, get the chunk coordinates and values
        start_idx = i * self.chunk_size
        end_idx = min((i + 1) * self.chunk_size, len(self.x))
        
        x = self.x[start_idx:end_idx]
        values = self.data[start_idx:end_idx]
        
        if self.use_transform:
            values = self.transform(values)
        return x, values
