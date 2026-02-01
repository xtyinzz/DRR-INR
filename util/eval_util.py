import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from util.vis_io import get_vti, write_vti
from torch.utils.tensorboard import SummaryWriter
from time import time
from pytorch_msssim import ssim
from lpips import LPIPS
from accelerate import Accelerator
from PIL import Image

def to8b(x): return (255./x.max()*(x-x.min())).astype(np.uint8)

def minmax_scale(tensor): return (tensor - tensor.min()) / \
    (tensor.max() - tensor.min())

def batched_compute_gpu(t_list, fn, batch_size, device, fn_kwargs={}):
    '''
    given a list of cpu tensors, calculate a stat from fn in gpu
    by batching and merge back to cpu
    
    t_list: list of tensors to be sliced along dim 0 and passed as arguments to fn
    fn: function that takes tensors as arguments
    '''
    
    # Get the number of items to batch over (assume all tensors have same size on dim 0)
    num_items = t_list[0].shape[0]
    
    stats = []
    for i in range(0, num_items, batch_size):
        # Slice each tensor in the list and move to device
        batch_inputs = [t[i:i+batch_size].to(device) for t in t_list]
        # Pass the sliced tensors as separate arguments to fn
        batch_result = fn(*batch_inputs, **fn_kwargs).to('cpu')
        stats.append(batch_result)
    
    stats = torch.tensor(stats)
    return stats



def batched_forward(
    model, x, batch_size, device='cpu',
    fwd_fn = lambda model, x: model.forward(x)
):
    '''
    batched inference with full output in cpu
    '''
    if device is None:
        device = next(model.parameters()).device
    preds_mc = [fwd_fn(model, x[i:i+batch_size].to(device)).to('cpu')
                for i in range(0, len(x), batch_size)]
    if hasattr(model, 'num_members') and model.num_members > 1:
        preds_mc = torch.concatenate(preds_mc, 1)
    elif hasattr(model, 'num_members') and model.num_members == 1:
        preds_mc = torch.concatenate(preds_mc, 1)[0]
    else:
        preds_mc = torch.concatenate(preds_mc, 0)
    return preds_mc



@torch.no_grad()
def eval_metrics(pred, gt):
    gt = gt.flatten()
    pred = pred.flatten().to(gt)
        
    mse = F.mse_loss(pred, gt).item()
    psnr = - 10.0*np.log10(mse)
    # print(f"evalutations:  psnr={psnr:.4f} mse={mse:.4e}")
    return {'mse': mse, 'psnr': psnr}


def idx_flat_to_nd(indices, shape):
    indices = indices.clone().flatten()
    if isinstance(shape, list):
        shape = np.array(shape)
    factors = [shape[i:].prod() for i in range(1, len(shape))]
    
    idxnd = torch.ones(len(indices), len(shape), dtype=indices.dtype).to(indices)
    for i in range(len(shape)-1):
        idxnd[:,i] = torch.div(indices, factors[i], rounding_mode='floor')
        indices -= idxnd[:,i]*factors[i]
    idxnd[:,-1] = indices
    # print(f'FlatTo3D={time()-s:.4f}')
    return idxnd

def log_to_writer(writer: SummaryWriter, iteration: int, losses: dict = None, figs: dict = None):
    if losses is not None:
        for key in losses.keys():
            if(losses[key] is not None):
                writer.add_scalar(str(key), losses[key], iteration)

    if figs is not None:
        for key in figs.keys():
            if(figs[key] is not None):
                writer.add_figure(str(key), figs[key], iteration)

def train_report(step, elapsed_time, loss_list, metrics, tb_writer, ei):
    '''
    loss_list: list of loss component dicts with keys {name, value, ratio}
    '''
    print(
        f"time={elapsed_time:.2f}s | ep={ei} step={step} | "
        f"Train-loss={metrics['loss']:.2e}",
        end='',
        flush=True
    )
    for l_component in loss_list:
        print(f", L{l_component['name']}={l_component['value']:.2e}", end='')
    print(
        f" | MSE={metrics['mse']:.2e}",
        f"PSNR={metrics['psnr']:.2f}",
        flush=True
    )
    metrics = {
        'train-loss': metrics['loss'],
        'train-mse': metrics['mse'],
        'train_psnr': metrics['psnr']
    }
    loss_dict = {f"train-{l['name']}":l['value'] for l in loss_list}
    metrics.update(loss_dict)
    log_to_writer(tb_writer, step, metrics)

def post_train_report(i, pred, y, loss, loss_list, tb_writer, tic, ei):
    elapsed_time = time() - tic
    with torch.no_grad():
        metrics = eval_metrics(pred, y)
        metrics['loss'] = loss.item()
        train_report(i, elapsed_time, loss_list, metrics, tb_writer, ei)


def batched_forward_hdinr(
    model, x, batch_size, device=None, out_device=None,
    fwd_fn = lambda model, x: model.forward(x)
):
    '''
    batched inference for hdinr model with input x, cond, cond_i,
    with full output in cpu (optional out_device)
    '''
    if device is None:
        device = next(model.parameters()).device
    
    if out_device is None:
        out_device = device
    
    num_coords = x[0].shape[1]
    x[1], x[2] = x[1].to(device), x[2].to(device)
    preds_mc = []
    for i in range(0, num_coords, batch_size):
        coords = x[0][:, i:i+batch_size].to(device).contiguous()
        batch_preds = fwd_fn(model, (coords, x[1], x[2])).to(out_device)
        preds_mc.append(batch_preds)
    del coords
    preds_mc = torch.concatenate(preds_mc, 1)
    return preds_mc


def batched_forward_inr(
    model, x, batch_size, device=None, out_device=None,
    fwd_fn = lambda model, x: model(x)
):
    '''
    batched inference for inr model with input x
    with full output in cpu (optional out_device)
    '''
    if device is None:
        device = next(model.parameters()).device
    
    if out_device is None:
        out_device = device

    num_coords = x.shape[0]

    preds_mc = []
    for i in range(0, num_coords, batch_size):
        coords = x[i:i+batch_size].to(device)
        # print(coords.shape)
        batch_preds = fwd_fn(model, coords).to(out_device)
        preds_mc.append(batch_preds)
    del coords

    preds_mc = torch.concatenate(preds_mc, 0)
    return preds_mc


def prepare_vti(predictions, y, dataset):
    """
    Prepare prediction and ground truth data for VTI format without writing files.
    
    Args:
        predictions: The predicted tensor data
        y: The ground truth tensor data
        dataset: The dataset object containing the dimensions
    
    Returns:
        tuple: (processed_predictions_vti, processed_y_vti) in VTI format
    """
    predictions, y = predictions.view(*dataset.dims).float(), y.view(*dataset.dims).float()
    if dataset.dims[0] == 512:
        predictions = torch.nn.functional.interpolate(
            predictions[None, None], scale_factor=0.5, 
            mode='trilinear', align_corners=True
        ).squeeze()
        y = torch.nn.functional.interpolate(
            y[None, None], scale_factor=0.5, 
            mode='trilinear', align_corners=True
        ).squeeze()
    
    predictions_vti = get_vti({'a': predictions.cpu().detach().clone().numpy()})
    y_vti = get_vti({'a': y.cpu().numpy()})

    return predictions_vti, y_vti


def post_train_eval_ddp(
    trainstep, accl:Accelerator, eval_dataloader:DataLoader,
    model:torch.nn.Module, fwd_fn, tb_writer: SummaryWriter=None,
    batch_size = 128**3, #16777216 # 256**3
    dump_idx = [],
    dump_path = '',
    dataset = None,
):
    """
    Evaluates the model in main process using Accelerate DDP setup.
    Args:
        trainstep (int): The current training step.
        accl (Accelerator): The accelerator object used for device placement and other utilities.
        eval_dataloader (DataLoader): DataLoader for the evaluation dataset.
        model (torch.nn.Module): The model to be evaluated.
        tb_writer (SummaryWriter): TensorBoard writer for logging evaluation metrics.
    Returns:
        dict: A dictionary containing evaluation metrics:
            - 'evalMSE' (float): Mean Squared Error of the evaluation.
            - 'evalPSNR' (float): Peak Signal-to-Noise Ratio of the evaluation.
    """
    mse_val = 0
    num_elems = 0
    start_time = time()
    mses = []
    psnrs = []
    ssims = []
    model.eval()
    time_s = time()
    # loader delivers a field at a time
    time_refine = time()
    if hasattr(model, 'refine_cond') and hasattr(model, 'cond_refiner'):
        model.refine_cond()
    if hasattr(model, 'refine_spatial') and hasattr(model, 'spatial_refiner'):
        model.refine_spatial()
    print(f"Refine time: {time()-time_refine:.2f}s")
    if_dump = False
    if (isinstance(dump_idx, list) and len(dump_idx) > 0):
        if_dump = True

    for step, (x, cond, cond_i, y) in enumerate(eval_dataloader):
        if if_dump and step not in dump_idx:
            continue
    
        x, cond, cond_i, y = x.to(accl.device), cond.to(accl.device), cond_i.to(accl.device), y.to(accl.device)
        with torch.no_grad():
            
            predictions = batched_forward_hdinr(
                model, [x, cond, cond_i], batch_size,
                device=accl.device, fwd_fn=fwd_fn
            )
            predictions = predictions.view(y.shape)

        rank_mse = torch.mean((y - predictions) ** 2)
        rank_count = y.numel()
        # Calculate SSIM for this rank
        if len(dataset.dims) > 1:
            rank_ssim = ssim(
                predictions.view(1, -1, *dataset.dims),
                y.view(1, -1, *dataset.dims),
                data_range=1,
                size_average=False
            ).squeeze()
        else:
            rank_ssim = torch.tensor(0.0, device=accl.device)

        rank_stat = torch.tensor([[rank_mse, rank_count, rank_ssim]], device=accl.device)
        rank_stat = accl.gather_for_metrics(rank_stat)
        rank_stat = rank_stat.view(-1, 3).cpu()
        rank_mse, rank_count, rank_ssim = rank_stat[:, 0], rank_stat[:, 1], rank_stat[:, 2]
        
        rank_psnr = -10*torch.log10(rank_mse)

        rank_mse = rank_mse.tolist()
        rank_psnr = rank_psnr.tolist()
        rank_ssim = rank_ssim.tolist()
        mses.extend(rank_mse)
        psnrs.extend(rank_psnr)
        ssims.extend(rank_ssim)

        formatted_rank_mses = [f"{num:.4e}" for num in rank_mse]
        formatted_rank_psnrs = [f"{num:.2f}" for num in rank_psnr]
        formatted_rank_ssims = [f"{num:.4f}" for num in rank_ssim]
        if accl.is_main_process:
            if if_dump:
                # mpas_npy = dataset.inverse_transform(predictions.flatten()).cpu().numpy()
                # np.save(os.path.join(f'{dump_path}', f'pred_{step}.npy'), mpas_npy)
                # gt_npy = dataset.inverse_transform(y.flatten()).cpu().numpy()
                # np.save(os.path.join(f'{dump_path}', f'gt_{step}.npy'), gt_npy)
                
                predictions, y = prepare_vti(predictions, y, dataset)
                write_vti(os.path.join(f'{dump_path}', f'pred_{step}.vti'), predictions)
                if not os.path.exists(os.path.join(f'{dump_path}', f'gt_{step}.vti')):
                    write_vti(os.path.join(f'{dump_path}', f'gt_{step}.vti'), y)
    
                
            print(f"Field{step:03} | time={time()-start_time:.2f} | per-field-time={time()-time_s:.2f} | "
                   f"field-PSNR={formatted_rank_psnrs} | field-MSE={formatted_rank_mses} | field-SSIM={formatted_rank_ssims}\n")
            time_s = time()
        
    total_time = time() - start_time
    avg_time = total_time / len(eval_dataloader)
    print('total time', total_time)
    print('avg time', avg_time)
    
    mse_val = np.array(mses).mean()
    psnr_val = -10*np.log10(mse_val)
    ssim_val = np.array(ssims).mean()
    metrics = {
        'test_mse': mse_val,
        'test_psnr': psnr_val,
        'test_ssim': ssim_val,
    }
    df = pd.DataFrame({'MSE': mses, 'PSNR': psnrs, 'SSIM': ssims})
    if tb_writer is not None and accl.is_main_process:
        print(f"Evaluation finished: MSE={mse_val:.4e}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")
        log_to_writer(tb_writer, trainstep, metrics)
    return metrics, df



def post_train_eval_ddp_inr(
    trainstep, accl:Accelerator, field_ds,
    model:torch.nn.Module, tb_writer: SummaryWriter=None,
    batch_size = 128**3, #16777216 # 256**3
    dump_path = '',
    args=None,
):
    """
    Evaluates the model in main process using Accelerate DDP setup.
    Args:
        trainstep (int): The current training step.
        accl (Accelerator): The accelerator object used for device placement and other utilities.
        eval_dataloader (DataLoader): DataLoader for the evaluation dataset.
        model (torch.nn.Module): The model to be evaluated.
        tb_writer (SummaryWriter): TensorBoard writer for logging evaluation metrics.
    Returns:
        dict: A dictionary containing evaluation metrics:
            - 'evalMSE' (float): Mean Squared Error of the evaluation.
            - 'evalPSNR' (float): Peak Signal-to-Noise Ratio of the evaluation.
    """
    model.eval()
    # loader delivers a field at a time

    x = field_ds.get_all_coords_unseen()
    if field_ds.dim_scale_factor < 1.0:
        field_ds.dim_scale_factor = 1.0
        field_ds.data = field_ds._load_data()
    y = field_ds.data.view(field_ds.dims.prod().item(), -1)
    lpips_fn = LPIPS(net='vgg').to(accl.device)
    
    start_time = time()
    with torch.no_grad():
        if hasattr(model, 'mlp_base') and hasattr(model.mlp_base, 'sp'):
            model.mlp_base.refine_spatial()
    with torch.no_grad():
        with accl.autocast():
            predictions = batched_forward_inr(
                model, x, batch_size,
                device=accl.device
            )
            
        total_time = time() - start_time

        mse_val = batched_compute_gpu((predictions, y), F.mse_loss, batch_size, accl.device).mean().item()
        psnr_val = -10*np.log10(mse_val)
        # Calculate SSIM in a batched manner to save GPU memory
        ssim_vals = []
        lpips_vals = []
        num_points = field_ds.dims.prod().item()
        
        # Determine window dimensions that maintain aspect ratio
        num_dims = len(field_ds.dims)
        aspect_ratios = torch.tensor(field_ds.dims, dtype=torch.float32) / field_ds.dims.prod()**(1/num_dims)
        window_size_1d = batch_size**(1/num_dims)
        window_dims = torch.round(window_size_1d * aspect_ratios).int().tolist()

        # Adjust window dimensions to match batch size
        actual_batch_size = np.prod(window_dims)

        for i in range(0, num_points, actual_batch_size):
            end_idx = min(i + actual_batch_size, num_points)
            if end_idx - i < actual_batch_size:
                continue

            pred_slice = predictions[i:end_idx]
            y_slice = y[i:end_idx]

            # Reshape slices to window dimensions for SSIM calculation
            # Add batch and channel dimensions: (1, 1, *window_dims)
            pred_window = pred_slice.permute(1, 0).view(1, -1, *window_dims).to(accl.device)
            y_window = y_slice.permute(1, 0).view(1, -1, *window_dims).to(accl.device)

            # Compute SSIM for the window
            with torch.no_grad():
                ssim_val = ssim(pred_window, y_window, data_range=1.0, size_average=False)
                lpips_val = lpips_fn(pred_window, y_window).squeeze()
            ssim_vals.append(ssim_val.item())
            lpips_vals.append(lpips_val.item())

        ssim_val = np.mean(ssim_vals) if ssim_vals else 0.0
        lpips_val = np.mean(lpips_vals) if lpips_vals else 0.0

    if accl.is_main_process:
        if len(args.dump_idx_cdTest_spFull) > 0:
            field_dump_path = os.path.join(dump_path, f'pred.jpg')
            # Save prediction as jpg using PIL
            pred_vis = predictions.view(*field_ds.dims, 3).cpu().numpy()
            pred_vis = np.clip(pred_vis, 0, 1)
            pred_vis = (pred_vis * 255).astype(np.uint8)
            if len(field_ds.dims) == 2:
                # 2D case: save directly as image
                img = Image.fromarray(pred_vis)
                img.save(field_dump_path)
            else:
                raise NotImplementedError("Dumping non-2D fields as images is not implemented.")
        
        print(f"Field evaluation | time={time()-start_time:.2f} | "
              f"MSE={mse_val:.4e} | PSNR={psnr_val:.2f} | SSIM={ssim_val:.4f} | LPIPS={lpips_val:.4f}")
        


    print('total inference time', total_time) 
    
    metrics = {
        'test_mse': mse_val,
        'test_psnr': psnr_val,
        'test_ssim': ssim_val,
        'test_lpips': lpips_val,
    }
    df = pd.DataFrame({'MSE': [mse_val], 'PSNR': [psnr_val], 'SSIM': [ssim_val], 'LPIPS': [lpips_val]})
    if tb_writer is not None and accl.is_main_process:
        print(f"Evaluation finished: MSE={mse_val:.4e}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
        log_to_writer(tb_writer, trainstep, metrics)
    return metrics, df