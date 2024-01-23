

#### FILE INCOMPLETE, see plot_dynamics_vs_environment function in multiagent_policy.py for a working version ####

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from pettingzoo.mpe import simple_spread_v3
import sys
import copy
import matplotlib.pyplot as plt

# from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action
from ma_state_dynamics_model import MAStateModel

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

def torch_to_dict(tensor, agents):
    output_dict = {}
    for i, agent in enumerate(agents):
        output_dict[agent] = tensor[i].detach().cpu().numpy()  # Convert to numpy array if needed
    return output_dict

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
    state, info = env.reset()
    agents = env.agents
    
    s_dim = state['agent_0'].shape[0]

    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)
    a_dim = action_tensor.shape[1]
    
    print("s_dim", s_dim, ", a_dim", a_dim)

    return s_dim, a_dim


if __name__ == '__main__':
    PATH = "..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
    PATH = "..\\model_checkpoints\\ma_state_dynamics\\ma_state_dynamics_10_31_2023_10_38_13_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.208150715827943.pth"
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
    
    s_d_model = 256
    s_h_dim = 256
    
    model = MAStateModel(s_dim, a_dim, s_d_model, s_h_dim)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    # model = model.cuda()
    model.eval()

    episodes = 50
    ep_len = 50
    
    actualr = [[],[],[]]
    predictedr = [[],[],[]]
    
    actualx = [[],[],[]]
    predictedx = [[],[],[]]

    actualy = [[],[],[]]
    predictedy = [[],[],[]]

    state, infos = env.reset()
    agents = env.agents
    state_tensor = dict_to_torch(state)
    print(state_tensor.shape)

  
    losses = []
    with torch.no_grad():
        for j in range(100):    
            action = sample_action(env, agents)
            action_tensor = dict_to_torch(action)
 
            obs, reward, terminations, truncations, infos= env.step(action)
            obs_tensor = dict_to_torch(obs)
            next_state_mean, next_state_stdev = model.forward(state_tensor, action_tensor)
            predicted_state = model.sample_next_state(state_tensor, action_tensor)
            state_tensor = predicted_state
            
            output_x = next_state_mean[..., 2]
            output_y = next_state_mean[..., 3]

            for i in range(3):
                predictedx[i].append(output_x[i].item())
                actualx[i].append(obs_tensor[i][2].item())
                
                predictedy[i].append(output_y[i].item())
                actualy[i].append(obs_tensor[i][3].item())


            state = obs
            next_state = dict_to_torch(obs)
            

    plt.figure()

    agent_colors = ['red', 'blue', 'orange']

    for i in range(3):
        # Plotting the predicted positions as dotted lines
        plt.plot(predictedx[i], predictedy[i], '--', color=agent_colors[i], label="Predicted Agent " + str(i))
        
        # Plotting the actual positions as straight lines
        plt.plot(actualx[i], actualy[i], '-', color=agent_colors[i], label="Actual Agent " + str(i))
        
        # Plotting the starting point as a large dot
        plt.scatter(predictedx[i][0], predictedy[i][0], s=100, color=agent_colors[i])
        
    plt.legend()
    plt.savefig('position_plot_continuous.png')
    plt.show()


    env.close()

    # plt.plot(actualx[0].T, actualy[0].T, label = "agent 1 actual")
    # plt.plot(actualx[1].T, actualy[1].T, label = "agent 2 actual")
    # plt.plot(actualx[2].T, actualy[2].T, label = "agent 3 actual")
    # plt.plot(predictedx[0].T, predictedy[0].T, label = "agent 1 predicted")
    # plt.plot(predictedx[1].T, predictedy[1].T, label = "agent 2 predicted")
    # plt.plot(predictedx[2].T, predictedy[2].T, label = "agent 3 predicted")
    # plt.legend()
    # plt.savefig('position_plot.png')
    # plt.show()
    # env.close()



    # states = []
    # actions = []
    # next_states = []
    # rewards = []
    
    # for i in range(100):
    #     state = env.reset()
    #     for j in range(100):
    #         states.append(dict_to_torch(state))

    #         action = sample_action(env, agents)
    #         action_tensor = dict_to_torch(action)

    #         # this is necessary because discrete action space has dimension 1 for each agent (0-4 for each action option)
    #         # but continuous action space has dimension 4 (each value represents velocity in each direction)
    #         if env_config['continuous_actions']:
    #             actions.append(action_tensor)
    #         else:
    #             actions.append(action_tensor.reshape(action_tensor.size(0), 1)) 
    #         obs, reward, terminated, info = env.step(action)

    #         next_states.append(dict_to_torch(obs))
    #         reward_tensor = dict_to_torch(reward) 
    #         rewards.append(reward_tensor.reshape(reward_tensor.size(0), 1))

    #         state = obs

    # states = torch.stack(states)
    # actions = torch.stack(actions)
    # next_states = torch.stack(next_states)
    # rewards = torch.stack(rewards)

    # for i in range(100):
