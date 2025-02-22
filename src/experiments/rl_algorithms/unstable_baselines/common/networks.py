import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Sequence, Union, final

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gym
from gym.spaces import Discrete, Box, MultiBinary, space

from src.experiments.rl_algorithms.unstable_baselines.common import util
import torch.nn.functional as F
import warnings


def get_optimizer(optimizer_class: str, network: nn.Module, learning_rate: float, **kwargs):
    """
    Parameters
    ----------
    optimizer_class: ['adam', 'sgd'], optional
        The optimizer class.

    network: torch.nn.Module
        The network selected to optimize.

    learning_rate: float

    Return
    ------
    """
    optimizer_fn = optimizer_class.lower()
    if optimizer_fn == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_fn == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"Unimplemented optimizer {optimizer_class}.")
    return optimizer


def get_network(in_shape, net_param):
    """
    Parameters
    ----------
    in_shape:
        type: int or tuple
    net_param 
        type: tuple
        format: ("net type", *net_parameters)
    """
    (net_type, *net_args) = net_param
    if isinstance(in_shape, tuple) and len(in_shape) == 1:
        in_shape = in_shape[0] 
    if net_type == 'mlp':
        assert isinstance(in_shape, int) and len(net_args) == 1
        out_shape = net_args[0]    
        net = torch.nn.Linear(in_shape, out_shape)
    elif net_type == 'conv2d':
        assert isinstance(in_shape, tuple) and len(in_shape) == 3 and len(net_args) == 4
        out_channel, kernel_size, stride, padding = net_args
        assert padding >= 0 and stride >= 1 and kernel_size > 0
        in_channel, h, w = in_shape
        net = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        out_h = int((h + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_w = int((w + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_shape = (out_channel, out_h, out_w)
    elif net_type == "flatten":
        assert isinstance(in_shape, tuple) and len(in_shape) == 3
        net = torch.nn.Flatten()
        out_shape = int(np.prod(in_shape))
    elif net_type in ['maxpool2d', 'avgpool2d']:
        kernel_size, stride, padding = net_param
        assert padding >= 0 and stride >= 1 and kernel_size > 0 and len(in_shape) == 3
        c, h, w = in_shape
        if net_type == "maxpool2d":
            net = torch.nn.MaxPool2d(kernel_size, stride, padding)
        elif net_type == "avgpool2d":
            net = torch.nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise NotImplementedError
        out_h = int((h + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_w = int((w + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_shape = (c, out_h, out_w)
    else:
        raise ValueError(f"Network params {net_param} illegal.")

    return net, out_shape

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def get_act_cls(act_fn_name):
    act_fn_name = act_fn_name.lower()
    if act_fn_name == "tanh":
        act_cls = torch.nn.Tanh
    elif act_fn_name == "sigmoid":
        act_cls = torch.nn.Sigmoid
    elif act_fn_name == 'relu':
        act_cls = torch.nn.ReLU
    elif act_fn_name == 'leakyrelu':
        act_cls = torch.nn.LeakyReLU
    elif act_fn_name == 'identity':
        act_cls = torch.nn.Identity
    elif act_fn_name == 'swish':
        act_cls = Swish
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls



class JointNetwork(nn.Module): # joint networks for multiple inputs, then concatenate the outputs for each input, finally use mlp to produce output
    def __init__(
            self,
            in_shape_list: list,
            out_shape_list: list,
            network_params_list: list,
            act_fn_list: list,
            out_act_fn_list: list,
            joint_out_shape: int,
            joint_network_params: list,
            joint_act_fn: str,
            joint_out_act_fn: str,
            **kwargs
    ):
        super(JointNetwork, self).__init__()
        self.network_heads = []
        for in_shape, out_shape, network_params, act_fn, out_act_fn in zip(in_shape_list, out_shape_list, network_params_list, act_fn_list, out_act_fn_list):
            self.network_heads.apend(SequentialNetwork(in_shape, out_shape, network_params, act_fn, out_act_fn))

        joint_network_input_shape = sum(out_shape_list)
        self.joint_network = SequentialNetwork(joint_network_input_shape, joint_out_shape, joint_network_params, joint_act_fn, joint_out_act_fn)
         
    def forward(self, inputs: list):
        outputs = [net(x) for net, x in zip(self.network_heads, inputs)]
        joint_network_input = torch.cat(outputs, axis=1)
        final_output = self.joint_network(joint_network_input)
        return final_output

    @property
    def weights(self):
        all_weights = []
        for net in self.network_heads:
            all_weights += net.weights
        all_weights += self.joint_network.weights
        return all_weights


class SequentialNetwork(nn.Module):

    def __init__(
            self, in_shape: int,
            out_shape: int,
            network_params: list,
            act_fn="relu",
            out_act_fn="identity",
            **kwargs
    ):
        super(SequentialNetwork, self).__init__()
        if len(kwargs.keys()) > 0:
            warn_str = "Redundant parameters for SequentialNetwork {}.".format(kwargs)
            warnings.warn(warn_str)
        ''' network parameters:
            int: mlp hidden dim
            str: different kinds of pooling
            (in_channel, out_channel, stride, padding): conv2d
        ''' 
        self.networks = []
        curr_shape = in_shape
        if isinstance(act_fn, str):
            act_cls = get_act_cls(act_fn)
            act_cls_list = [act_cls for _ in network_params]
        else:
            act_cls_list = [get_act_cls(act_f) for act_f in act_fn]

        out_act_cls = get_act_cls(out_act_fn)

        for i, (net_param, act_cls) in enumerate(zip(network_params, act_cls_list)):
            curr_network, curr_shape = get_network(curr_shape, net_param)
            self.networks.extend([curr_network, act_cls()])

        #final network only support mlp
        final_net_params = ('mlp', out_shape)
        final_network, final_shape = get_network(curr_shape, final_net_params)

        self.networks.extend([final_network, out_act_cls()])
        
        self.networks = nn.Sequential(*self.networks)

    def forward(self, inputs: Union[torch.Tensor, list]): # takes two forms of input: 1. single tensor 2. multiple tensor to be concatenated (the same as joint network)
        
        if isinstance(inputs, torch.Tensor): # single tensor 
            return self.networks(inputs)
        elif isinstance(inputs, list):
            #concatenate the inputs, and forward
            input = torch.cat(inputs, dim=1)
            return self.networks(input)

    @property
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear) or isinstance(net, torch.nn.modules.Conv2d)]


class BasePolicyNetwork(ABC, nn.Module):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 *args, **kwargs
                 ):
        super(BasePolicyNetwork, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.kwargs = kwargs

        # if isinstance(hidden_dims, int):
        #     hidden_dims = [hidden_dims]
        # hidden_dims = [input_dim] + hidden_dims

        # # init hidden layers
        # self.hidden_layers = []
        # act_cls = get_act_cls(act_fn)
        # for i in range(len(hidden_dims) - 1):
        #     curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
        #     curr_network = get_network([curr_shape, next_shape])
        #     self.hidden_layers.extend([curr_network, act_cls()])

        # init output layer shape
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, MultiBinary):
            self.action_dim = action_space.shape[0]
        else:
            raise TypeError

    @abstractmethod
    def forward(self, obs):
        raise NotImplementedError

    @abstractmethod
    def sample(self, obs, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, obs, actions, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        return nn.Module.to(self, device)


class DeterministicPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 *args, **kwargs
                 ):
        super(DeterministicPolicyNetwork, self).__init__(observation_space, action_space, network_params, act_fn)

        self.deterministic = True
        self.policy_type = "deterministic"

        # get final layer
        # final_network = get_network([hidden_dims[-1], self.action_dim])
        # out_act_cls = get_act_cls(out_act_fn)
        # self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0], network_params, act_fn, out_act_fn)


        # set noise
        self.noise = torch.Tensor(self.action_dim)

        # set scaler
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1., dtype=torch.float, device=util.device))
            self.register_buffer("action_bias", torch.tensor(0., dtype=torch.float, device=util.device))
        elif not isinstance(action_space, Discrete):
            self.register_buffer("action_scale",
                                 torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))
            self.register_buffer("action_bias",
                                 torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))

    def forward(self, obs: Union[torch.Tensor, list]):
        out = self.networks(obs)
        return out

    def sample(self, obs: Union[torch.Tensor, list]):
        action_prev_tanh = self.networks(obs)
        action_raw = torch.tanh(action_prev_tanh)
        action = action_raw * self.action_scale + self.action_bias

        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw,
            "action": action,
        }

    # CHECK: I'm not sure about the reparameterization trick used in DDPG
    def evaluate_actions(self, obs: Union[torch.Tensor, list]):
        action_prev_tanh = self.networks(obs)
        action_raw = torch.tanh(action_prev_tanh)
        action = action_raw * self.action_scale + self.action_bias

        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw,
            "action": action,
        }

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicyNetwork, self).to(device)


class CategoricalPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                  **kwargs
                 ):
        super(CategoricalPolicyNetwork, self).__init__(observation_space, action_space, network_params, act_fn, **kwargs)

        self.networks = SequentialNetwork(observation_space.shape, action_space.n, network_params, act_fn, out_act_fn)


    def forward(self, obs: Union[torch.Tensor, list]):
        out = self.networks(obs)
        return out

    def sample(self, obs: Union[torch.Tensor, list], deterministic=False):
        logit = self.forward(obs)
        probs = torch.softmax(logit, dim=-1)
        if deterministic:
            return {
                "logit": logit,
                "probs": probs,
                "action": torch.argmax(probs, dim=-1, keepdim=True).view(-1, 1),
                "log_prob": torch.log(torch.max(probs, dim=-1, keepdim=True).values + 1e-8),
            }
        else:
            dist = Categorical(probs=probs)

            action = dist.sample().view(-1, 1)
            z = (probs == 0.0).float() * 1e-8
            
            log_prob = torch.log(probs + z)
            return {
                "logit": logit,
                "probs": probs,
                "action": action,
                "log_prob": log_prob
            }

    def evaluate_actions(self, obs, actions, **kwargs):
        logit = self.forward(obs)
        probs = torch.softmax(logit, dim=1)
        dist = Categorical(probs)
        return {
            "log_prob": torch.log(torch.gather(probs, 1, actions.unsqueeze(1))),
            "entropy": dist.entropy().sum(0, keepdim=True)
        }
        #return dist.log_prob(actions).view(-1, 1), dist.entropy().view(-1, 1)

    def to(self, device):
        return super(CategoricalPolicyNetwork, self).to(device)


class GaussianPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 re_parameterize: bool = True,
                 predicted_std: bool = True,
                 parameterized_std: bool = False,
                 log_std: float = None,
                 log_std_min: int = -20,
                 log_std_max: int = 2,
                 stablize_log_prob: bool = False,
                 **kwargs
                 ):
        super(GaussianPolicyNetwork, self).__init__(observation_space, action_space, network_params, act_fn)

        self.deterministic = False
        self.policy_type = "Gaussian"
        self.predicted_std = predicted_std
        self.re_parameterize = re_parameterize
        if self.predicted_std:
            self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0] * 2, network_params, act_fn, out_act_fn)
        else:
            self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0], network_params, act_fn, out_act_fn)


        # set scaler
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1., dtype=torch.float, device=util.device))
            self.register_buffer("action_bias", torch.tensor(0., dtype=torch.float, device=util.device))
        elif not isinstance(action_space, Discrete):
            self.register_buffer("action_scale",
                                 torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))
            self.register_buffer("action_bias",
                                 torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))

        # set log_std
        if log_std == None:
            self.log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
        else:
            self.log_std = log_std
        if parameterized_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor(self.log_std)).to(util.device)
        else:
            self.log_std = torch.tensor(self.log_std, dtype=torch.float, device=util.device)

        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float, device=util.device))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float, device=util.device))
        self.stablize_log_prob = stablize_log_prob

    def forward(self, obs: Union[torch.Tensor, list]):
        out = self.networks(obs)
        action_mean = out[:, :self.action_dim]
        # check whether the `log_std` is fixed in forward() to make the sample function
        # keep consistent
        if self.predicted_std:
            action_log_std = out[:, self.action_dim:]    
        else:   
            action_log_std = self.log_std
        return action_mean, action_log_std

    def sample(self, obs: Union[torch.Tensor, list], deterministic: bool = False):

        mean, log_std = self.forward(obs)
        # util.debug_print(type(log_std), info="Gaussian Policy sample")

        
        if deterministic:
            action_mean_raw = mean.detach()
            action = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
            info = {
                "action_mean_raw": mean,
                "action": action, 
                "log_prob": torch.ones_like(action_mean_raw),
                "log_std": torch.zeros_like(action_mean_raw)
            }
            return info
        if self.stablize_log_prob: # for sac like actor that require re-parameterization and 
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
            dist = Normal(mean, log_std.exp())   
            if self.re_parameterize:
                action_prev_tanh = dist.rsample()
            else:
                action_prev_tanh = dist.sample()
            action_raw = torch.tanh(action_prev_tanh)
            action = action_raw * self.action_scale + self.action_bias

            log_prob_prev_tanh = dist.log_prob(action_prev_tanh)
            # log_prob = log_prob_prev_tanh - torch.log(self.action_scale*(1-torch.tanh(action_prev_tanh).pow(2)) + 1e-6)
            if self.stablize_log_prob:
                log_prob = log_prob_prev_tanh - (
                        2 * (np.log(2) - action_prev_tanh - torch.nn.functional.softplus(-2 * action_prev_tanh)))
            else:
                log_prob = log_prob_prev_tanh
            log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
            info = {
                "action_mean_raw": mean,
                "action": action, 
                "log_prob": log_prob,
                "log_std": log_std
            }
        else:
            action_mean = torch.tanh(mean)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
            dist = Normal(action_mean, log_std.exp())
            action = dist.sample()
            action = torch.clip(action,  min=-1.0, max=1.0)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            action =  action * self.action_scale + self.action_bias
            info = {
                "action_mean_raw": action_mean * self.action_scale,
                "action": action, 
                "log_prob": log_prob,
                "log_std": log_std
            }
        return info
    
    def evaluate_actions(self, obs: Union[torch.Tensor, list], actions: torch.Tensor):
        """ Evaluate action to get log_prob and entropy.
        
        Note: This function should not be used by SAC because SAC only replay obs in buffer.
        """
        
        mean, log_std = self.forward(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
        action_mean = torch.tanh(mean)
        dist = Normal(action_mean, log_std.exp())

        
        actions = (actions - self.action_bias) / self.action_scale
        #actions = torch.atanh(actions)
      
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return {
            "log_prob": log_prob,
            "entropy": entropy
        }

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyNetwork, self).to(device)


class PolicyNetworkFactory():
    @staticmethod
    def get(
            observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
            action_space: gym.Space,
            network_params: Union[Sequence[int], int],
            act_fn: str = "relu",
            out_act_fn: str = "identity",
            deterministic: bool = False,
            distribution_type: str = None,
            *args, **kwargs
    ):
        cls = None
        if deterministic:
            cls = DeterministicPolicyNetwork
        elif not distribution_type is None:
            cls = {
                "deterministic": DeterministicPolicyNetwork,
                "gaussian": GaussianPolicyNetwork,
                "categorical": CategoricalPolicyNetwork
            }.get(distribution_type)
        elif isinstance(action_space, Discrete):
            cls = CategoricalPolicyNetwork
        elif isinstance(action_space, Box):
            cls = GaussianPolicyNetwork
        else:
            raise ArithmeticError(
                f"Cannot determine policy network type from arguments - deterministic: {deterministic}, distribution_type: {distribution_type}, action_space: {action_space}.")
        return cls(observation_space, action_space, network_params, act_fn, out_act_fn, *args,
                   **kwargs)
