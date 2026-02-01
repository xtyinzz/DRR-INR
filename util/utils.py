import os
import random
import collections

import numpy as np
import torch
import psutil
from accelerate import Accelerator

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
minmax_scale = lambda tensor: (tensor - tensor.min()) / (tensor.max() - tensor.min())


Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))



def print_model_parameters(model):
    # Print total parameters at the start
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    print("Model Parameters:")
    print("-" * 50)
    
    # Print top-level module parameters and percentages
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        percentage = (module_params / total_params) * 100
        print(f"{name}: {module_params:,} params ({percentage:.2f}%)")
    
    print("-" * 50)
    
    # Print detailed parameters for each parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            percentage = (param_count / total_params) * 100
            print(f"{name}: {list(param.shape)} - {param_count:,} params ({percentage:.2f}%)")
    
    print("-" * 50)
    print(f"Total trainable parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    print(model)
    return total_params

def report_peak_ram_usage():
    import resource
    # Print peak memory usage (in MB)
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RAM usage: {peak_memory / 1024/1024:.2f} MB")  # Divide by 1024 to convert KB to MB

def report_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    print(f"RAM Usage: {ram_usage:.2f} GB")

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def dict_to_checkpoint_path(metrics_dict):
    """
    Convert a dictionary of metric results into a string suitable for use in a file path.
    For small values, use scientific notation to represent the numbers.
    
    Args:
    - metrics_dict: A dictionary where keys are metric names and values are their results.
    
    Returns:
    - A string that can be used in a checkpoint file path.
    """
    def format_value(value):
        """Helper function to format values: use scientific notation for small values."""
        if abs(value) < 1e-4:
            return f"{value:.2e}"  # Scientific notation for small values
        else:
            return f"{round(value, 4)}"  # Round for larger values
    
    # Format each key-value pair in the format: key-value
    metrics_str = "_".join(f"{key}-{format_value(value)}" for key, value in metrics_dict.items())
    
    # Replace any invalid characters (e.g., '/', ':') with valid ones (like '-')
    metrics_str = metrics_str.replace("/", "-").replace(":", "-")
    
    return metrics_str


def get_batch_model_paths(batch_model_dir: str):
    model_dirs = sorted([f.path for f in os.scandir(batch_model_dir) if f.is_dir()])
    model_paths = []
    for mdir in model_dirs:
        last_model = sorted(os.listdir(mdir))[-1]
        last_model_path = os.path.join(mdir, last_model)
        model_paths.append(last_model_path)
    return model_paths


def save_checkpoint_ddp(accl:Accelerator, log_dir,
                    epoch: int=0,
                    metrics: dict={},):
    loss_str = dict_to_checkpoint_path(metrics)
    pckpt = os.path.join(log_dir, f'ep{epoch:04}_{loss_str}')
    accl.save_state(pckpt)
    print(f'checkpoint saved at {pckpt}', flush=True)


def report_gpumem(device=0):
  toGb = 1024*1024*1024
  totalm = torch.cuda.get_device_properties(0).total_memory / toGb
  max_alloc = torch.cuda.max_memory_allocated() / toGb
  max_rsv = torch.cuda.max_memory_reserved() / toGb
  pmaxalloc = 100*max_alloc / totalm
  pmaxrsv = 100*max_rsv / totalm
  
  alloc = torch.cuda.memory_allocated() / toGb
  rsv = torch.cuda.memory_reserved() / toGb
  palloc = 100*alloc / totalm
  prsv = 100*rsv / totalm
  

  print(f'total GPU Mem: {totalm:.4}Gb')
  print(f'allocated percentage: {palloc:8.4}% --- usage: {alloc:.4}Gb')
  print(f'reserved percentage: {prsv:8.4}% --- usage: {rsv:.4}Gb')
  print(f'max allocated percentage: {pmaxalloc:8.4}% --- usage: {max_alloc:.4}Gb')
  print(f'max reserved percentage: {pmaxrsv:8.4}% --- usage: {max_rsv:.4}Gb')
  

def memReport():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
    import psutil
    # print(sys.version)
    print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
    
def getNumNans(tensor):
  return torch.isnan(tensor).sum()

def parse_files(fp):
  with open(fp, 'r') as f:
    lines = [line.rstrip() for line in f]
  return lines
