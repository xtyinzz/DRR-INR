import torch
import numpy as np
import os

def imp_func(data, minval, maxval, num_bins):
    freq = 0
    nBlocks = 16
    block_size = data.numel() // nBlocks
    for bidx in range(nBlocks):
        block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
        freq += block_freq
    freq = freq.type(torch.double)
    # freq = torch.histc(data, bins=num_bins, min=minval, max=maxval).type(torch.double)
    importance = 1. / (freq + 1.e-10)
    # importance /= importance.sum()
    
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
        rnd_idx += torch.randint(high=point_cluster_size, size=rnd_idx.shape)
    return rnd_idx

def save_importance_data(data, imp_dir='temp_imp', imp_field_size=256**3, num_bins=64):
    os.makedirs(imp_dir, exist_ok=True)
    for i in range(len(data)):
        data = data[i].flatten()
        val_hist, point_imp, point_cluster_size = get_point_importance(data, val_bins=num_bins, imp_field_max_size=imp_field_size)
        save_npy = {
            'point_imp': point_imp,
            'point_cluster_size': point_cluster_size
        }
        np.save(os.path.join(imp_dir, f'temp_point_imp{i:02d}.npy'), save_npy)


