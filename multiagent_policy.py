import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Normal
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

import gym
from pettingzoo.mpe import simple_spread_v2

from comet_ml import Experiment

from datetime import datetime

from simple_transformer import MultiAgentDynamicsModel

comet = True
experiment = None



class MultiAgentPolicyModel(nn.Module):
    def __init__(self, dynamics_model, s_dim, a_dim, d_model, h_dim):
        super(MultiAgentPolicyModel, self).__init__()

        self.dynamics_model = dynamics_model

        self.pre_transformer = nn.Sequential(
                        nn.Linear(s_dim, d_model),
                        nn.ReLU())

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, dim_feedforward = 256, nhead=1, batch_first=True)
        
        self.meanlayer = nn.Sequential(
                        nn.Linear(d_model, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, a_dim),
                        nn.Sigmoid()
        )
        # self.stdevlayer = nn.Sequential(
        #                 nn.Linear(d_model, h_dim),
        #                 nn.ReLU(),
        #                 nn.Linear(h_dim, a_dim),
        #                 nn.Softplus()
        # )
        
    def forward(self,state):

        pre_transformer = self.pre_transformer(state)
        feats = self.encoder_layer(pre_transformer)

        action = self.meanlayer(feats)
        # s_next_stdev = self.stdevlayer(feats)
        
        return action #,s_next_stdev

    def get_loss(self,states):
        '''
        Passes batch of states and actions through dynamics model, then evaluates log likelihood of predictions
        
        states: batch_size x s_dim
        actions: batch_size x a_dim
        next_states: batch_size x s_dim

        '''
        action = self.forward(states) #s_next_sig
        exp_reward = self.dynamics_model(states, action)[0][...,-1:] #TODO: take mean over agents, check loss value
        # print(exp_reward)
        # print(exp_reward[...,-1:])
        loss = -1*torch.mean(exp_reward)

        return loss, action


def train(model, dynamics_model, start_state, opt, exploration, epochs, device):
    losses = list()

    #TODO: reset state
    state = start_state

    for epoch in range(epochs):
        # clear the gradients
        opt.zero_grad()
        loss, action = model.get_loss(state) 
        # calculate loss
        loss.backward()
        # loss.backward() computes dloss/dx for every parameter x 

        opt.step()
        losses.append(loss.item())
        
        # if np.random.random_sample() < (1 - exploration):
        #     #random
        #     action = Box(0.0, 1.0, (5))

        next_state_mean, next_state_stdev = dynamics_model.forward(state, action)
        next_state = torch.normal(next_state_mean, next_state_stdev)
        state = next_state[...,:-1]

    return np.mean(losses)

def validate(policy_model, environment, agents, epochs):
    state_dict = env.reset()
    state_tensor = dict_to_torch(state)
    # use vectorized environment (GYM) make_vec_env function (similar to make_env)
    # so takes in integer for how many trajectories you want to do at same time (10)
    for epoch in range(epochs):
        actions = policy_model.forward(state_tensor)
        #convert actions to dictionary

def sample_action(env, agents):
    action = {}
    for agent in agents:
        action_space = env.action_space(agent)
        action[agent] = action_space.sample()
    return action

def dict_to_numpy(dictionary):
    matrix = []
    for key in dictionary:
        matrix.append(dictionary[key])
    return np.array(matrix)

def dict_to_torch(dictionary):
    return torch.tensor(dict_to_numpy(dictionary))

if __name__ == '__main__':
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\copy_transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_10_26_2022_07_51_40_simple_spread_v2_wd_0.001_lr_0.0001_hdim_512\\transformer_Decay_best_loss_-1.8287531244372266.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_18_2022_13_25_34_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.749979948712691.pth"
    PATH = ".\\all_checkpoints_transformer\\transformer_01_25_2023_02_10_05_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.724443092141849.pth"
    env_name = 'simple_spread_v2'
    env_config = {
        'N':3, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }

    env = simple_spread_v2.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])    
    
    deviceID = 'cuda:0'
    
    state_dict = env.reset()
    start_state = dict_to_torch(state_dict)

    agents = env.agents
    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)

    s_dim = start_state.shape[1]
    a_dim = action_tensor.shape[1]
    print("state dim", s_dim)
    print("action dim", a_dim)
    d_model = 64
    h_dim = 256
    
    dynamics_model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    dynamics_model.load_state_dict(torch.load(PATH)['model_state_dict'])
    device = torch.device(deviceID)

    training_epochs = 500
    n_epochs = 5000

    policy_model = MultiAgentPolicyModel(dynamics_model, s_dim, a_dim, d_model, h_dim)
    
    lr = 1e-4
    wd = 0.001 #weight decay
    opt = torch.optim.Adam(policy_model.parameters(), lr=lr, weight_decay=wd) #which opt to use?

    for i in range(n_epochs):
        mean_loss = train(policy_model, dynamics_model, start_state, opt, 0.1, training_epochs, device)
        if comet:
            experiment.log_metric("train_loss", mean_loss, step=i)