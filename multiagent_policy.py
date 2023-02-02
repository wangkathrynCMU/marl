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

from multiagent_dynamics_model import MultiAgentDynamicsModel

comet = True
if comet:
    experiment = Experiment(
        api_key="s8Nte4JHNGudFkNjXiJ9RpofM",
        project_name="marl-policy",
        workspace="wangkathryncmu",
    )

device = 'cpu'

if torch.cuda.is_available():
    device = torch.device('cuda:0')

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


class MultiAgentPolicyModel(nn.Module):
    def __init__(self, dynamics_model, agents, s_dim, a_dim, d_model, h_dim):
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
    
    def forward_to_dictionary(self, state):
        # i think this only works if only one state is passed in
        action = self.forward(state).cpu().numpy()

        output = dict()
        index = 0
        for agent in agents:
            output[agent] = action[index]
            index +=1

        return output

    # def get_loss(self,states):
    #     '''
    #     Passes batch of states and actions through dynamics model, then evaluates log likelihood of predictions
        
    #     states: batch_size x s_dim
    #     actions: batch_size x a_dim
    #     next_states: batch_size x s_dim

    #     '''
    #     action = self.forward(states) #s_next_sig
    #     exp_reward = self.dynamics_model(states, action)[0][...,-1:] #TODO: take mean over agents, check loss value
    #     # print(exp_reward)
    #     # print(exp_reward[...,-1:])
    #     loss = -1*torch.mean(exp_reward)

    #     return loss, action


def train(model, dynamics_model, start_states, opt, exploration, ep_len, epochs, prev_epochs, device):
    losses = list()

    #TODO: reset state
    states = start_states

    # clear the gradients
    
    losses = []
    for epoch in range(epochs):

        loss = None
        opt.zero_grad()

        for i in range(ep_len):
            actions = model.forward(states) 
            next_states, rewards = dynamics_model.sample_next_state_reward(states, actions) #shape (batch size, num agents, 1)
            rewards = torch.mean(rewards) # shape (batch size, 1)
            # print('mean_rewards', torch.mean(rewards))
            if loss == None:
                loss = rewards
            else:
                loss = loss + (-1 * rewards)

        mean_loss = loss/ep_len

        mean_loss.backward()
        opt.step()

        losses.append(mean_loss.item())

        if comet:
            experiment.log_metric("train_loss", mean_loss, step= prev_epochs + epoch)
    return np.mean(losses)

def validate(policy_model, env, ep_len, epochs):
    # use vectorized environment (GYM) make_vec_env function (similar to make_env)
    # so takes in integer for how many trajectories you want to do at same time (10)
    losses = []
    with torch.no_grad():
        for epoch in range(epochs):    
            state_dict = env.reset()
            total_loss = 0
            for i in range(ep_len):
                state_tensor = dict_to_torch(state_dict).cuda()
                action = policy_model.forward_to_dictionary(state_tensor)
                state_dict, reward, terminated, info = env.step(action)
                mean_loss = -1 * (np.mean(dict_to_numpy(reward)))
                total_loss += mean_loss
            epoch_loss = total_loss/ep_len
            losses.append(epoch_loss)
            #convert actions to dictionary
    return np.mean(losses)
    
if __name__ == '__main__':
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\copy_transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_10_26_2022_07_51_40_simple_spread_v2_wd_0.001_lr_0.0001_hdim_512\\transformer_Decay_best_loss_-1.8287531244372266.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_18_2022_13_25_34_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.749979948712691.pth"
    PATH = "..\\biorobotics\\all_checkpoints_transformer\\transformer_01_25_2023_02_10_05_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.724443092141849.pth"
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

    
    batch_size = 100
    
    state_dict = env.reset()
    agents = env.agents
    
    action_dict = sample_action(env, agents)
    print(action_dict)
    action_tensor = dict_to_torch(action_dict)

    s_dim = dict_to_torch(state_dict).shape[1]
    a_dim = action_tensor.shape[1]
    print("state dim", s_dim)
    print("action dim", a_dim)
    d_model = 64
    h_dim = 256
    
    dynamics_model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    dynamics_model.load_state_dict(torch.load(PATH)['model_state_dict'])
    dynamics_model = dynamics_model.cuda()

    training_epochs = 500
    n_epochs = 5000

    policy_model = MultiAgentPolicyModel(dynamics_model, agents, s_dim, a_dim, d_model, h_dim)
    policy_model = policy_model.cuda()

    lr = 1e-4
    wd = 0.001 #weight decay
    opt = torch.optim.Adam(policy_model.parameters(), lr=lr, weight_decay=wd) #which opt to use?
    ep_len = 100
    train_epochs = 50
    test_epochs = 25
    prev_epochs = 0

    for i in range(n_epochs):

        states = []
        for j in range(batch_size):
            state_dict = env.reset()
            states.append(dict_to_torch(state_dict))
        states = torch.stack(states)
        states = states.cuda()

        train_loss = train(policy_model, dynamics_model, states, opt, 0.1, ep_len, train_epochs, prev_epochs, device)
        prev_epochs += train_epochs
        
        test_loss = validate(policy_model=policy_model, env=env, ep_len=ep_len, epochs=test_epochs)

        if comet:
            experiment.log_metric("test_loss", test_loss, step=i)
        print("epoch", i, " | ", "train_loss:", train_loss, "test_loss:", test_loss)

        