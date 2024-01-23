

#### FILE INCOMPLETE, see plot_dynamics_vs_environment function in multiagent_policy.py for a working version ####

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from pettingzoo.mpe import simple_spread_v2
import sys
import copy
import matplotlib.pyplot as plt

from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action

import torch.distributions.normal as Normal
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

def sample_action(env, agents):
    action = {}
    for agent in agents:
        action_space = env.action_space(agent)
        action[agent] = action_space.sample()
        
    return action
def dict_to_array(dictionary):
    matrix = []
    for key in dictionary:
        matrix.append(dictionary[key])
    return matrix

def dict_to_numpy(dictionary):
    matrix = []
    for key in dictionary:
        value = dictionary[key]
        matrix.append(value)
    return np.array(matrix)

def dict_to_torch(dictionary):
    return torch.tensor(dict_to_numpy(dictionary))

def validate(model):
    N = 1
    ep_len = 100
    batch_size = 100

    states,actions,next_states, rewards = gather_data(N, ep_len, env_name, env_config)
    dataset = torch.utils.data.TensorDataset(states,actions,next_states, rewards)
    test_loader  = DataLoader(dataset,  batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for batch_i, (states, actions, next_states, rewards) in enumerate(test_loader):
            print("batch", batch_i)
            loss, mse_loss = model.get_loss(states, actions, next_states, rewards)
            print("LOSS, MSELOSS", loss, mse_loss)

def get_sdim_adim(env):
    state = env.reset()
    agents = env.agents
    
    s_dim = state['agent_0'].shape[0]

    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)
    a_dim = action_tensor.shape[1]
    
    print("s_dim", s_dim, ", a_dim", a_dim)

    return s_dim, a_dim


if __name__ == '__main__':
    PATH = "..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
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
    
    d_model = 64
    h_dim = 256

    s_dim, a_dim = get_sdim_adim(env)
    model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    device = torch.device(deviceID)
    model.eval()
    # validate(model)

    episodes = 50
    ep_len = 50
    
    actualx = []
    actualy = []

    predictedx = []
    predictedy = []

    predicted_state = copy.deepcopy(state)
    predicted_state = dict_to_torch(predicted_state)
    state_arr = dict_to_numpy(state)

    actualx = state_arr[:,2].reshape(-1, 1)
    actualy = state_arr[:,3].reshape(-1, 1)
    predictedx = state_arr[:,2].reshape(-1, 1)
    predictedy= state_arr[:,3].reshape(-1, 1)
    
    # predicted_state = predicted_state[None, :,:]

    with torch.no_grad():
        # for i in range(2):
        #     state = env.reset()
        #     predicted_state = copy.deepcopy(state)
        #     predicted_state = dict_to_torch(predicted_state)
        #     predicted_state = predicted_state[None, :,:]
            for j in range(100):    
                action = sample_action(env, agents)
                # print("action", action)
                action_tensor = dict_to_torch(action)
                # action_tensor = action_tensor.reshape(action_tensor.size(0), 1)
                # action_tensor = action_tensor[None, :]
                # print("action tensor", action_tensor)

                obs, reward, terminated, info = env.step(action)
                s_next_mean, s_next_sig = model(predicted_state, action_tensor)

                obs_arr = np.matrix(dict_to_numpy(obs))
                actualx = np.concatenate((actualx, obs_arr[:,2]), axis = 1)
                actualy = np.concatenate((actualy, obs_arr[:,3]), axis = 1)

                predicted_state = s_next_mean[:,:,:-1]
                # use squeeze
                s_next_mean = predicted_state[0]
                predictedx = np.concatenate((predictedx, s_next_mean[:,2].numpy().reshape(-1, 1)), axis = 1)
                predictedy = np.concatenate((predictedy, s_next_mean[:,3].numpy().reshape(-1, 1)), axis = 1)

                # reward_arr = dict_to_array(reward) 
                # rewards_actual.append(reward_arr)

                # pred_reward = s_next_mean[:,-1:].tolist()

    plt.plot(actualx[0].T, actualy[0].T, label = "agent 1 actual")
    plt.plot(actualx[1].T, actualy[1].T, label = "agent 2 actual")
    plt.plot(actualx[2].T, actualy[2].T, label = "agent 3 actual")
    plt.plot(predictedx[0].T, predictedy[0].T, label = "agent 1 predicted")
    plt.plot(predictedx[1].T, predictedy[1].T, label = "agent 2 predicted")
    plt.plot(predictedx[2].T, predictedy[2].T, label = "agent 3 predicted")
    plt.legend()
    plt.savefig('position_plot.png')
    plt.show()
    env.close()