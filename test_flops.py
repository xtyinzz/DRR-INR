from models.srns import HDINRBase
from models.baselines.fainr.attention_MoE_cloverleaf import KVMemoryModel
from models.baselines.kplane import KPlaneField
from models.baselines.explorable_inr import INR_FG
import torch
import yaml
from fvcore.nn import FlopCountAnalysis
from configs.configs import Config
import argparse


def count_flops(model, x_shape, cond_dim, device='cpu', hdinr_precompute=False):
    x = torch.rand(*x_shape)
    c = torch.rand(x_shape[0], cond_dim)
    i = torch.tensor(list(range(x_shape[0])))

    x = (x, c, i)
    
    if isinstance(model, HDINRBase):
        cond_bank = torch.rand(x_shape[0], cond_dim)
        model.set_cond_bank(cond_bank)
        x = tuple([xx.to(device) for xx in x])
        x = (x[0], x[2], x[1])

    elif isinstance(model, INR_FG) or isinstance(model, KVMemoryModel) or isinstance(model, KPlaneField):
        x, cond, cond_i = x
        B, N, _ = x.shape
        cond_rep = cond[:,None].repeat(1, x.shape[1], 1)
        x = torch.cat([x, cond_rep], dim=-1)
        x = x.view(-1, x.shape[-1])
        x = x.to(device)

    with torch.no_grad():
        flops_1 = FlopCountAnalysis(model, x)
        flops_1.unsupported_ops_warnings(False)
        flops_1.uncalled_modules_warnings(False)
        total_tflops = flops_1.total() / 1e12
        
        if hdinr_precompute and isinstance(model, HDINRBase):
            if hasattr(model.spatial_encoder, 'refined_grid'):
                model.spatial_encoder.refined_grid = torch.nn.Parameter(model.spatial_encoder.refined_grid)
            if hasattr(model.latent, 'refined_lines'):
                model.latent.refined_lines = torch.nn.Parameter(model.latent.refined_lines)

    return total_tflops

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FLOPs Counter")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--config_idx', type=int, required=True, help='Config index')
    args = parser.parse_args()

    config_path = args.config_path
    config_idx = args.config_idx
    cond_dim = 6
    cond_batch, spatial_batch = int(1e2), int(1e3)
    num_runs = 10000
    sim_runs = 101
    x_shape = (cond_batch, spatial_batch, 3)
    ############ args ############

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = Config(config[config_idx])
    model = config.get_model(verbose=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    tflops = 0
    tflops_one = 0
    for i in range(sim_runs):
        if i == 0:
            tflops_one = count_flops(model, x_shape, cond_dim=cond_dim, device=device, hdinr_precompute=True)
        else:
            tflops += count_flops(model, x_shape, cond_dim=cond_dim, device=device, hdinr_precompute=False)
    tflops = (tflops / (sim_runs - 1)) * (num_runs - 1) + tflops_one
    print(f"Total TFLOPs approx for {num_runs} over {sim_runs} runs: {tflops:.6f}")
