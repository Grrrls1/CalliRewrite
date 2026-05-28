from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP, miniblock
from siren import Sine
from siren import SIREN

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]



def FFTblock(in_dim, fourier_dim, sigma, train_B, )->nn.Module:
    
    b_shape = (in_dim, fourier_dim // 2)
    B_ = nn.Parameter(torch.normal(torch.zeros(*b_shape), torch.full(b_shape, sigma))) # mu sigma
    B_.requires_grad = train_B
    return B_

class FourierMLP(nn.Module):
    """
    param in **kwargs:

    :param sigma: default to 1.0, initialize B matrix1.0,
    :param train_B=False,
    :param concatenate_fourier=False,
    """
    def __init__(self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device
        
        name_dict = ['sigma', 'train_B', 'fourier_dim','concatenate_fourier']
        # insurance
        if kwargs.keys() is not None:
            for name in name_dict:
                assert name in kwargs.keys()
        
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)

        self.concatenate_fourier = kwargs['concatenate_fourier']
        if self.concatenate_fourier:
            mlp_input_dim = kwargs['fourier_dim'] + input_dim
        else:
            mlp_input_dim = kwargs['fourier_dim']

        # shape lists
        hidden_sizes = [mlp_input_dim] + list(hidden_sizes)
        model = []  # rest of
        
        ## begin make self.B matrix:
        self.sigma = kwargs['sigma']
                
        self.B = FFTblock( 
                input_dim, kwargs['fourier_dim'], self.sigma, kwargs['train_B']
            )# make self matrix

        ## make rest networks
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, norm_args_list,
            activation_list, act_args_list
        ):
            model += miniblock(
                in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer
            )
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        
        ## fourier matrix
        proj = (2 * np.pi) * torch.matmul(obs, self.B)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.concatenate_fourier:
            ff = torch.cat([obs, ff], dim=-1)
        return self.model(ff)


class RBFMLP(nn.Module):
    """MLP preceded by Gaussian radial basis function features."""

    def __init__(self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        rbf_dim = kwargs['rbf_dim']
        rbf_sigma = kwargs['rbf_sigma']
        if rbf_dim <= 0:
            raise ValueError(f"rbf_dim must be > 0, got {rbf_dim}.")
        if rbf_sigma <= 0:
            raise ValueError(f"rbf_sigma must be > 0, got {rbf_sigma}.")

        self.device = device
        self.flatten_input = flatten_input
        self.rbf_sigma = rbf_sigma
        self.concatenate_rbf = kwargs['concatenate_rbf']
        self.centers = nn.Parameter(torch.empty(rbf_dim, input_dim).uniform_(-1.0, 1.0))
        self.centers.requires_grad = kwargs['train_rbf_centers']

        mlp_input_dim = rbf_dim + input_dim if self.concatenate_rbf else rbf_dim
        self.model = MLP(
            mlp_input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim

    @no_type_check
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)

        obs_norm = torch.sum(obs ** 2, dim=-1, keepdim=True)
        centers_norm = torch.sum(self.centers ** 2, dim=-1).unsqueeze(0)
        squared_dist = obs_norm + centers_norm - 2 * torch.matmul(obs, self.centers.t())
        squared_dist = squared_dist.clamp_min(0.0)
        features = torch.exp(-squared_dist / (2 * self.rbf_sigma ** 2))
        if self.concatenate_rbf:
            features = torch.cat([obs, features], dim=-1)
        return self.model(features)


class My_MLP(nn.Module):
    """My Wrapper of MLP to support more specific DRL usage. referring to Tisnshou wrapper

    Only to notify:
    :param learn_fourier: default False, decide whether to transfer state into a fourier-version feature.
        https://arxiv.org/pdf/2112.03257.pdf; dim of fourier layer is picked fromhidden_sizes[0]
    :param learn_rbf: default False, map state to Gaussian radial basis features before the MLP.
    """
    def __init__(self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        learn_fourier: bool = False,
        learn_rbf: bool = False,
        **kwargs,
    ) -> None:
        
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        
        if learn_fourier and learn_rbf:
            raise ValueError("learn_fourier and learn_rbf cannot both be enabled.")

        # modify
        if (learn_fourier):
            self.model = FourierMLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer, **kwargs
        )
        elif learn_rbf:
            self.model = RBFMLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer, **kwargs
        )
        else:
            self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            if (learn_fourier):
                self.Q, self.V = FourierMLP(**q_kwargs,**kwargs), FourierMLP(**v_kwargs,**kwargs)
            elif learn_rbf:
                self.Q, self.V = RBFMLP(**q_kwargs,**kwargs), RBFMLP(**v_kwargs,**kwargs)
            else:    
                self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state
    


class My_Siren(nn.Module):
    
    def __init__(self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        flatten_input: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.device = device
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        self.flatten_input = flatten_input
        output_dim = action_dim if not self.use_dueling and not concat else 0
        hidden_sizes = list(hidden_sizes)
        self.output_dim = output_dim if output_dim>0 else hidden_sizes[-1]
        initializer = 'siren'
        w0 = 1.0
        w0_initial = 30.0
        c = 6
        self.model =  SIREN(hidden_sizes, input_dim, self.output_dim, w0, w0_initial,
        initializer=initializer, c=c)

        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state
