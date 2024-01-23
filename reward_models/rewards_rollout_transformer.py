
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3
import sys
import copy
import matplotlib.pyplot as plt

from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action
from reward_model import RewardDynamicsModel
from reward_model_transformer import RewardDynamicsTransformer

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

def get_sdim_adim(env):
    state, info = env.reset()
    agents = env.agents
    
    s_dim = state['agent_0'].shape[0]

    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)
    a_dim = action_tensor.shape[1]
    
    print("s_dim", s_dim, ", a_dim", a_dim)

    return s_dim, a_dim

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


if __name__ == '__main__':
    PATH = "..\\..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
    R_PATH = "..\\..\\model_checkpoints\\reward_dynamics\\reward_dynamics_10_12_2023_12_17_35_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\reward_dynamics_Decay_best_loss_-1.666979865068348.pth"
    R_PATH_TRANSFORMER = "..\\..\\model_checkpoints\\reward_dynamics\\reward_dynamics_transformer_10_31_2023_10_39_02_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-2.2542934103572403.pth"
    env_name = 'simple_spread_v3'
    env_config = {
        'N':3, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }

    env = simple_spread_v3.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])    
    
    deviceID = 'cuda:0'
    device = torch.device(deviceID)
    
    d_model = 64
    h_dim = 256

    s_dim, a_dim = get_sdim_adim(env)
    model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    model.eval()
    
    d_model = 256
    dim_feedforward = 512
    nhead = 8
    num_layers = 1
    
    reward_model = RewardDynamicsTransformer(s_dim, a_dim, d_model, dim_feedforward, nhead, num_layers)
    reward_model.load_state_dict(torch.load(R_PATH_TRANSFORMER)['model_state_dict'])
    reward_model.eval()

    episodes = 50
    ep_len = 50
    
    actualr_model = [[],[],[]]
    predictedr_model = [[],[],[]]

    actualr = [[],[],[]]
    predictedr = [[],[],[]]
    
    actualx = [[],[],[]]
    predictedx = [[],[],[]]

    actualy = [[],[],[]]
    predictedy = [[],[],[]]

    state, info = env.reset()
    agents = env.agents
    state_tensor = dict_to_torch(state) 
    print(state_tensor.shape)

  
    losses = []
    with torch.no_grad():
        for j in range(100):    
            action = sample_action(env, agents)
            action_tensor = dict_to_torch(action)

            obs, reward, _, _, _ = env.step(action)
            state_tensor = dict_to_torch(state)
            reward_tensor = dict_to_torch(reward)
            obs_tensor = dict_to_torch(obs)
            next_state_mean, next_state_stdev = model.forward(state_tensor, action_tensor)
            r_next_state_mean, r_next_state_stdev = reward_model.forward(state_tensor, action_tensor)
            # predicted_state, predicted_reward = model.sample_next_state_reward(state_tensor, action_tensor)
            
            output_reward_model = r_next_state_mean
            output_reward = next_state_mean[...,-1:]
            output_x = next_state_mean[..., 2]
            output_y = next_state_mean[..., 3]

            for i in range(3):
                predictedr_model[i].append(output_reward_model[i].item())
                predictedr[i].append(output_reward[i].item())
                actualr[i].append(reward_tensor[i].item())
                
                predictedx[i].append(output_x[i].item())
                actualx[i].append(obs_tensor[i][2].item())
                
                predictedy[i].append(output_y[i].item())
                actualy[i].append(obs_tensor[i][3].item())


            state = obs
            # print("actual state", reward)
            # print("predict state", next_state_mean)
            # print("actual reward", reward)
            # print("predict reward", output_reward)
            s_next_distribution = Normal.Normal(next_state_mean, next_state_stdev) 
            # print("shapes",dict_to_torch(obs).shape,dict_to_torch(reward).shape)
            next_state_reward = torch.cat((dict_to_torch(obs),dict_to_torch(reward).reshape(3, 1)),dim=-1)

            loss = - torch.mean(s_next_distribution.log_prob(next_state_reward))
            mseloss = torch.nn.MSELoss()(next_state_reward, next_state_mean)
            losses.append(loss)
    
    x_axis = [x for x in range(len(predictedr[0]))]
    agent_colors = ['red', 'blue', 'orange']

    # Create an empty list to collect the legend handles
    legend_handles = []

    # Plot each agent's data
    for i in range(3):
        if i == 0:  # Add labels only for the first iteration to avoid repetition in the legend
            legend_handles.append(plt.plot(x_axis, predictedr[i], '--', color=agent_colors[i], label="Original Model")[0])
            legend_handles.append(plt.plot(x_axis, predictedr_model[i], 'o', color=agent_colors[i], label="Transformer Model")[0])
            legend_handles.append(plt.plot(x_axis, actualr[i], '-', color=agent_colors[i], label="Actual")[0])
        else:  # No labels for the subsequent iterations
            plt.plot(x_axis, predictedr[i], '--', color=agent_colors[i])
            plt.plot(x_axis, predictedr_model[i], 'o', color=agent_colors[i])
            plt.plot(x_axis, actualr[i], '-', color=agent_colors[i])

    # Add a custom legend entry
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='white', lw=2, label='Different colors = Different Agents')]
    legend_handles += custom_lines


    # Adding the labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward Predictions by Model")


    # Show the legend using the collected handles
    plt.legend(handles=legend_handles)
    plt.savefig('reward_transformer_plot.png')
    plt.show()


    # x and y coordinate plot
    # plt.figure()
    # for i in range(3):
    #     plt.plot(predictedx[i], predictedy[i], label = "predicted pos agent " + str(i))
    #     plt.plot(actualx[i], actualy[i], label = "actual pos agent " + str(i))
    # plt.legend()
    # plt.savefig('position_plot_1_step.png')
    # plt.show()


    env.close()