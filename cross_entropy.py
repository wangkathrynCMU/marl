import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Normal
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from cross_entropy_rollouts import plot_CEM
from ma_state_dynamics_model import MAStateModel
from reward_model_transformer import RewardDynamicsTransformer
from CEM_rollouts_real_env import CEM_vis

import gym
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3

from comet_ml import Experiment

from datetime import datetime

import ipdb

def CEM(cost_fn,pop_size,frac_keep,n_iters, device, a_mean,a_sig=None):
    '''
    INPUTS:
    cost_fn: function that takes a pop_size x T x a_dim-sized tensor of potential action sequences and returns pop_size tensor of associated costs (negative rewards)
    pop_size: how many trajs to simulate
    n_iters: number of optimization iterations to perform
    a_mean: T x a_dim tensor of initial mean of actions (won't matter if a_sig is none)
    a_sig: T x a_dim tensor of initial mean of actions, or none (in which case initial actions will be sampled uniformly in interval [-1,+1])

    OUTPUTS:
    a_mean: T x a_dim tensor of optimized action sequence
    '''
    a_shape = [pop_size] + list(a_mean.shape)

    if a_sig is None:
        # sample from enironment action space, which I'm assuming is
        actions = torch.rand(a_shape,device=device)
        assert torch.all(actions <= 1)
        assert torch.all(actions >= -1)
    else:
        # sample actions form gaussian
        actions = a_mean + a_sig*torch.randn(a_shape,device=device)

    for i in range(n_iters):
        print('i: ', i)
        a_mean,a_sig,cost = CEM_update(actions,cost_fn,frac_keep)
        # print('cost: ', cost)
        # re-sample actions
        actions = a_mean + a_sig*torch.randn(a_shape,device=device)
        
        # clip
        actions = torch.clamp(actions,min=0,max=1)

    return a_mean,a_sig

def CEM_update(actions,cost_fn,frac_keep):
    N = actions.shape[0]
    k = int(N*frac_keep)

    costs = cost_fn(actions).squeeze()
    print('costs mean:',torch.mean(costs).item())

    inds = torch.argsort(costs)
    inds_keep = inds[:k]
    actions_topk = actions[inds_keep,...].squeeze(1)
    cost_topk = torch.mean(costs[inds_keep])

    print('cost_topk:',torch.mean(cost_topk).item())

    a_mean = torch.mean(actions_topk,dim=0)
    a_sig = torch.std(actions_topk, dim=0)

    print("a_mean shape", a_mean.shape)
    print("a_sig shape", a_sig.shape)

    # ipdb.set_trace()

    return a_mean,a_sig,cost_topk

class CustomMultiagentEnv:
    def __init__(self, config):
        # Initialize PettingZoo Environment
        self.start_states = config['start_states']
        self.device = config['device']
        
        STATE_PATH = config['STATE_PATH']
        REWARD_PATH = config['REWARD_PATH']

        s_dim, a_dim = self.get_sdim_adim(env)
        
        # INITIALIZE STATE MODEL
        s_d_model = 256
        s_h_dim = 256
        
        state_model = MAStateModel(s_dim, a_dim, s_d_model, s_h_dim)
        state_model.load_state_dict(torch.load(STATE_PATH)['model_state_dict'])
        self.state_model = state_model.cuda()

        # INITIALIZE REWARD MODEL
        r_d_model = 256
        r_dim_feedforward = 512
        r_nhead = 8
        r_num_layers = 1

        reward_model = RewardDynamicsTransformer(s_dim, a_dim, r_d_model, r_dim_feedforward, r_nhead, r_num_layers)
        reward_model.load_state_dict(torch.load(REWARD_PATH)['model_state_dict'])
        self.reward_model = reward_model.cuda()
    
    def get_sdim_adim(self, env):
        state, info = env.reset()
        s_dim = state['agent_0'].shape[0]

        action_space = env.action_space('agent_0')
        a_dim = action_space.shape[0]

        return s_dim, a_dim
        
    def step(self, states, actions):
        next_state = self.state_model.sample_next_state(states, actions)
        reward = self.reward_model.sample_next_reward(states, actions)

        return next_state, reward
    
    def CEM_cost_fn(self, actions):
        pop_size, T, n_agents, a_dim = actions.shape
        rewards = torch.zeros(pop_size, 1).to(self.device)
        curr_states = self.start_states

        for i in range(T):
            next_states, reward = self.step(curr_states, actions[:, i, :, :])
            curr_states = next_states
            reward = reward.mean(dim=1) #averaging over agents
            rewards += -reward
        
        return rewards / T
    
def dict_to_numpy(dictionary):
    keys = sorted(dictionary.keys())
    matrix = [dictionary[key] for key in keys]
    return np.array(matrix)

def dict_to_torch(dictionary):
    return torch.tensor(dict_to_numpy(dictionary))

def torch_to_dict(tensor, agents):
    output = dict()
    index = 0

    for agent in agents:
        output[agent] = tensor[index]
        index +=1
    
    return output

if __name__ == '__main__':
    pop_size = 1000
    frac_keep = 0.02
    n_iters = 50
    timesteps = 101
    n_agents = 3
    a_dim = 5

    STATE_PATH = "..\\model_checkpoints\\ma_state_dynamics\\ma_state_dynamics_10_31_2023_10_38_13_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.208150715827943.pth"
    REWARD_PATH = "..\\model_checkpoints\\reward_dynamics\\reward_dynamics_transformer_10_31_2023_10_39_02_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-2.2542934103572403.pth"

    env = simple_spread_v3.parallel_env(N = 3, 
                        local_ratio = 0.5, 
                        max_cycles = 100, 
                        continuous_actions = True,
                        render_mode='rgb_array')

    device = torch.device('cuda:0')

    start_state, _ = env.reset()
    start_state = dict_to_torch(start_state).cuda()
    start_states = start_state.unsqueeze(0).repeat(pop_size, 1, 1)

    print("start_states shape", start_states.shape)

    env_config = {
        'start_states': start_states,
        'device': device,
        'STATE_PATH': STATE_PATH,
        'REWARD_PATH': REWARD_PATH,
    }
    
    a_mean = torch.zeros(timesteps, n_agents, a_dim, device=device)
    # a_sig = torch.ones(timesteps, n_agents, a_dim, device=device)

    custom_env = CustomMultiagentEnv(env_config)
    final_a_mean, final_a_sig = CEM(custom_env.CEM_cost_fn, pop_size, frac_keep, n_iters, device, a_mean, a_sig = None)

    # ipdb.set_trace()
    print("Final Action Mean:", final_a_mean.shape)
    print("Final Action Sig:", final_a_sig.shape)
    # print("Final Action Mean:", final_a_mean)
    # print("Final Action Sig:", final_a_sig)

    final_a_mean = final_a_mean.cpu()
    final_a_sig = final_a_sig.cpu()

    states, actions = CEM_vis(env=env, a_mean=final_a_mean, a_sig=final_a_sig)
    actions = actions.cuda()
    states = states.cuda()
    plot_CEM(states, start_state, custom_env.state_model, actions)
    print("actions shape", actions.shape)
    env_config['start_states'] = start_state.unsqueeze(0)
    custom_env = CustomMultiagentEnv(env_config)
    print("Expected Reward: ", custom_env.CEM_cost_fn(actions.unsqueeze(0)))