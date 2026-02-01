import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from models.srns import *
from models.modules import Siren
from models.baselines.explorable_inr import *
from models.baselines.kplane import KPlaneField
from models.baselines.fainr.attention_MoE_cloverleaf import KVMemoryModel
from models.modules import get_nested_attr
from models.ngp import NGPRadianceField, DRRGridRadianceField
from nerfacc_scripts.radiance_fields.mlp import VanillaNeRFRadianceField
from datasets.sf_dataset import *
from util.utils import print_model_parameters
import importlib

# parse config files; reuse for your own project
class Config():

    def __init__(self, cfg_dict) -> None:
        self.config = cfg_dict

    def get_dataset(self) -> Dataset:
        """
        Instantiates and returns a PyTorch Dataset based on the configuration.

        The dataset's type and parameters are specified in the 'dataset' section
        of the configuration file. The 'type' should correspond to a class name
        that is available in the current scope (e.g., 'FieldDataset', 'HDFieldDataset').

        Returns:
            torch.utils.data.Dataset: The instantiated dataset.

        Raises:
            ValueError: If the dataset configuration is missing, the type is not
                        specified, or the specified class cannot be found.
        """
        if 'dataset' not in self.config:
            raise ValueError("Dataset configuration 'dataset' not found.")

        cfg = self.config['dataset']
        name = cfg.get('type')
        param = cfg.get('param', {})

        if not name:
            raise ValueError("Dataset 'type' not specified in the config.")

        try:
            # Find the dataset class in the current module's scope
            dataset_class = getattr(sys.modules[__name__], name)
        except AttributeError:
            raise ValueError(f"Dataset class '{name}' not found. Ensure it is imported.")

        return dataset_class(**param)
    
    
    def get_optim(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Instantiates and returns a PyTorch optimizer based on the configuration.

        Supports creating optimizers with multiple parameter groups, allowing for
        different learning rates for different parts of the model.

        The configuration should be under the 'optim' key and specify the
        optimizer 'type' (e.g., 'Adam', 'AdamW') and its 'param' dictionary.

        For multiple parameter groups, a 'groups' list should be provided. Each
        item in the list should define a 'params' key (a string path to a
        submodule, e.g., 'encoder.net') and a 'lr'. Any parameters not
        explicitly assigned to a group will be added to a default group with
        the main learning rate.

        Example config for multiple parameter groups:
        optim:
          type: Adam
          param: {lr: 0.001}
          groups:
            - {params: "encoder", lr: 0.0001}
            - {params: "decoder", lr: 0.001}

        Args:
            model (torch.nn.Module): The model whose parameters will be optimized.

        Returns:
            torch.optim.Optimizer: The configured PyTorch optimizer.

        Raises:
            ValueError: If the optimizer configuration is missing or the
                        specified optimizer type is not found in `torch.optim`.
        """
        if 'optim' not in self.config:
            raise ValueError("Optimizer configuration 'optim' not found.")

        cfg = self.config['optim']
        name = cfg.get('type')
        base_params = cfg.get('param', {})

        if not name:
            raise ValueError("Optimizer 'type' not specified in the config.")

        param_groups = []
        if 'groups' in cfg:
            all_model_params = set(model.parameters())
            assigned_params = set()

            for group_cfg in cfg['groups']:
                sub_module = get_nested_attr(model, group_cfg['params'])
                
                # Collect parameters from the submodule
                group_params = list(sub_module.parameters())
                
                # Ensure parameters are not double-assigned
                unique_params = [p for p in group_params if p in all_model_params and p not in assigned_params]
                if not unique_params:
                    continue

                param_groups.append({"params": unique_params, "lr": group_cfg['lr']})
                assigned_params.update(unique_params)
            
            # Add remaining unassigned parameters to a default group
            unassigned_params = list(all_model_params - assigned_params)
            if unassigned_params:
                param_groups.append({"params": unassigned_params})
        else:
            # Use all model parameters in a single group
            param_groups = [{"params": model.parameters()}]

        try:
            optimizer_class = getattr(torch.optim, name)
        except AttributeError:
            raise ValueError(f"Optimizer '{name}' not found in torch.optim.")

        return optimizer_class(param_groups, **base_params)

    def get_model(self, verbose: bool = True, **kwargs) -> torch.nn.Module:
        """
        Instantiates and returns a PyTorch model based on the configuration.

        The model's type and parameters are specified in the 'model' section
        of the configuration file. The type should be a fully qualified class
        name (e.g., 'models.srns.SRN').

        Args:
            verbose (bool, optional): If True, prints the model's parameter
                count. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the model's
                constructor, overriding config file parameters.

        Returns:
            torch.nn.Module: The instantiated model.

        Raises:
            ValueError: If the model configuration is missing or the specified
                model class cannot be found.
        """
        if 'model' not in self.config:
            raise ValueError("Model configuration not found in the config file.")

        cfg = self.config['model']
        name = cfg.get('type')
        param = cfg.get('param', {})

        if not name:
            raise ValueError("Model 'type' not specified in the config.")

        # Update parameters with any runtime arguments
        param.update(kwargs)

        try:
            if '.' in name:
                # Dynamically import the model class from a module path
                module_path, class_name = name.rsplit('.', 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            else:
                # Look for the class in the current module's scope
                model_class = getattr(sys.modules[__name__], name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Failed to import model class '{name}'. Error: {e}")

        model = model_class(**param)

        if verbose:
            print(f"Instantiated model '{name}'.")
            print_model_parameters(model)

        return model


    def set_args(self, args):
        """
        Overrides argparse arguments with values from the configuration file.

        This allows config file settings to take precedence over command-line
        arguments or their defaults. It merges settings from the 'train' and
        'test' sections of the config into the args namespace.

        Args:
            args (argparse.Namespace): The command-line arguments object.
        """
        args_dict = vars(args)
        for section in ['train', 'test']:
            if section in self.config:
                args_dict.update(self.config[section])