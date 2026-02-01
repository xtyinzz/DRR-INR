from typing import List, Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import abc
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
import numpy as np

from .modules import *



class HDINRBase(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        encoder_kwargs: Dict,
        spatial_fdim: int,
        decoder_kwargs: Dict,
        latent_kwargs: Dict,
        default_act: str = 'ELU',
        out_act: str = None,
        use_out_proj: bool = False,
        spatial_trans_kwargs: Dict = None,
        cond_trans_kwargs: Dict = None,
        if_refine: bool = False,
        verbose: bool = True,
        **kwargs, # For backward compatibility and unused args
    ):
        super().__init__()
        self._save_init_params(locals())
        self._init_components()
        if self.verbose:
            print(self)

    def _save_init_params(self, locals_dict: Dict):
        """Saves relevant initialization parameters as attributes."""
        self.out_dim = locals_dict.get('out_dim')
        self.hidden_dim = locals_dict.get('hidden_dim')
        self.spatial_fdim = locals_dict.get('spatial_fdim')
        self.verbose = locals_dict.get('verbose', True)

        self.encoder_kwargs = locals_dict.get('encoder_kwargs')
        self.latent_kwargs = locals_dict.get('latent_kwargs')
        self.decoder_kwargs = locals_dict.get('decoder_kwargs')
        self.spatial_trans_kwargs = locals_dict.get('spatial_trans_kwargs')
        self.cond_trans_kwargs = locals_dict.get('cond_trans_kwargs')

        self.if_refine = locals_dict.get('if_refine')
        self.have_refined_spatial = False
        self.have_refined_cond = False

        self.use_out_proj = locals_dict.get('use_out_proj', False)
        self.default_act_name = locals_dict.get('default_act', 'ELU')
        self.out_act_name = locals_dict.get('out_act')

    def _init_components(self):
        """Initializes the neural network modules."""
        self.default_act = (
            getattr(nn, self.default_act_name)()
            if hasattr(nn, self.default_act_name)
            else globals()[self.default_act_name]()
        )
        if self.out_act_name:
            self.out_act = (
                getattr(nn, self.out_act_name)()
                if hasattr(nn, self.out_act_name)
                else globals()[self.out_act_name]()
            )

        self.latent = get_model_by_name(self.latent_kwargs["type"], self.latent_kwargs["param"])
        self.spatial_encoder = get_model_by_name(self.encoder_kwargs["type"], self.encoder_kwargs["param"])
        self.decoder = get_model_by_name(self.decoder_kwargs['type'], self.decoder_kwargs['param'])

        # Refiner modules
        if self.spatial_trans_kwargs:
            self.spatial_refiner = get_model_by_name(self.spatial_trans_kwargs["type"], self.spatial_trans_kwargs["param"])
        if self.cond_trans_kwargs:
            self.cond_refiner = get_model_by_name(self.cond_trans_kwargs["type"], self.cond_trans_kwargs["param"])

        self.cond_latent_fdim = self._get_cond_latent_fdim()

        self.fusion = nn.Sequential(
            nn.Linear(self.cond_latent_fdim + self.spatial_fdim, self.hidden_dim),
            self.default_act
        )

        if self.use_out_proj:
            self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)

    def _get_cond_latent_fdim(self) -> int:
        """Determines the output feature dimension of the conditional latent module."""
        if isinstance(self.latent, MultiCondLatentLines) or isinstance(self.latent, CondLatentLines):
            return self.latent.refine_fdim
        return self.latent.latent_shape[-1]

    def get_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[List[Dict], torch.Tensor]:
        """
        Calculates the training loss, simplified to Mean Squared Error.

        Args:
            pred (torch.Tensor): The predicted output from the model.
            gt (torch.Tensor): The ground truth values.

        Returns:
            A tuple containing:
            - A list of dictionaries, each with loss component name, value, and ratio.
            - The total loss tensor.
        """
        loss_list = []
        total_loss = torch.tensor(0.0, device=pred.device)

        likelihood_data_loss = F.mse_loss(pred, gt)
        loss_list.append({'name': 'mse', 'value': likelihood_data_loss})
        total_loss += likelihood_data_loss

        if total_loss > 0:
            for component in loss_list:
                component['ratio'] = (component['value'].detach() / total_loss.detach()).item()

        return loss_list, total_loss

    def refine_spatial(self):
        if not hasattr(self, 'spatial_refiner'):
            raise AttributeError("Model does not have a 'spatial_refiner' module to bake.")

        self.spatial_encoder.refine_transforms(self.spatial_refiner)
        self.have_refined_spatial = True

    def refine_cond(self):
        if hasattr(self, 'cond_refiner'):
            cond_refiner = self.cond_refiner
        else:
            raise AttributeError("Model does not have a 'cond_refiner' to bake.")
            
        self.latent.refine_transforms(cond_refiner)
        self.have_refined_cond = True

    @torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        CB, SB, _ = x.shape
        x_flat = x.view(CB * SB, -1)

        # Refine spatial encoder if configured and not already done (e.g., first inference pass)
        if hasattr(self, 'spatial_refiner') and self.if_refine and (self.training or not self.have_refined_spatial):
            self.refine_spatial()

        spatial_feat = self.spatial_encoder(x_flat).view(CB, SB, -1)

        # Refine conditional module if configured and not already done
        if hasattr(self, 'cond_refiner') and self.if_refine and (self.training or not self.have_refined_cond):
            self.refine_cond()

        latent_feat = self.latent(cond)
        latent_feat = latent_feat.view(CB, 1, self.cond_latent_fdim).expand(-1, SB, -1)

        combined_feat = torch.cat([spatial_feat, latent_feat], dim=-1)
        fused_feat = self.fusion(combined_feat)
        output = self.decoder(fused_feat)

        if hasattr(self, 'out_proj'):
            output = self.out_proj(output)
        if hasattr(self, 'out_act'):
            output = self.out_act(output)

        return output




class BaseINR(nn.Module, abc.ABC):
    """
    Abstract base class for an Implicit Neural Representation (INR) model.
    
    This model defines a shared decoder MLP and a common loss function, while
    requiring subclasses to implement their own method for creating the initial
    feature encoding MLP (`mlp_base`).
    """
    def __init__(
        self,
        feat_dim: int,
        out_dim: int,
        decoder_hidden_dim: int = 64,
        decoder_n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        self.mlp_decoder = tcnn.Network(
            n_input_dims=self.feat_dim,
            n_output_dims=self.out_dim,
            network_config={
                "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                "n_neurons": decoder_hidden_dim, "n_hidden_layers": decoder_n_hidden_layers,
            }
        )
        
        # This abstract method must be implemented by any subclass.
        self.mlp_base = self._create_mlp_base()

    @abc.abstractmethod
    def _create_mlp_base(self) -> nn.Module:
        """
        Subclasses must implement this to define how input coordinates are
        mapped to a feature vector of dimension `self.feat_dim`.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass: coordinates -> features -> output.
        """
        features = self.mlp_base(x)
        output = self.mlp_decoder(features)
        return output

    def get_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[List[Dict], torch.Tensor]:
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



class DRRINR(BaseINR):
    """
    A DRR INR implementation using a custom spatial encoder ('DRR' style).
    """
    def __init__(
        self,
        out_dim: int,
        spatial_fdim: int,
        decoder_hidden_dim: int = 64,
        decoder_n_hidden_layers: int = 2,
        encoder_kwargs: dict = None,
    ):
        self.encoder_kwargs = encoder_kwargs
        super().__init__(
            feat_dim=spatial_fdim,
            out_dim=out_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_n_hidden_layers=decoder_n_hidden_layers,
        )

    def _create_mlp_base(self) -> nn.Module:
        spatial_encoder = get_model_by_name(
            self.encoder_kwargs["type"], self.encoder_kwargs["param"]
        )
        return spatial_encoder
    

