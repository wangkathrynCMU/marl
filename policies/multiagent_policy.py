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
from environment_to_gif import save_frames_as_gif

comet = False
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

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward = 256, nhead=1, batch_first=True)
        
        self.meanlayer = nn.Sequential(
                        nn.Linear(d_model, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, a_dim),
                        nn.Sigmoid()
        )
        
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

def dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def train(model, dynamics_model, start_states, opt, exploration, ep_len, epochs, device):
    losses = list()

    #TODO: reset state
    states = start_states
    batch_size = start_states.shape[0]
    rewards_buffer = []
    # clear the gradients
    losses = []
    for epoch in range(epochs):

        loss = 0
        opt.zero_grad()
        
        states = start_states
        for i in range(ep_len):
            actions = model.forward(states) 
            next_states, rewards = dynamics_model.sample_next_state_reward(states, actions) #shape (batch size, num agents, 1)
            states = next_states
            # print("next states, rewards", i, next_states, rewards)
            rewards_buffer.append(rewards)
            loss = loss + (-1 * torch.sum(rewards)/3/batch_size) #todo make num agents dynamic
            # print("loss", i, loss)

        mean_loss = loss/ep_len 
        print("meanloss", mean_loss)
        mean_loss.backward()
        opt.step()

        losses.append(mean_loss.item())
    if np.mean(losses) < 0.1:
        print(rewards_buffer)

    return np.mean(losses)

def validate(policy_model, env, ep_len, epochs):
    # use vectorized environment (GYM) make_vec_env function (similar to make_env)
    # so takes in integer for how many trajectories you want to do at same time (10)
    losses = []
    with torch.no_grad():
        for epoch in range(epochs):    
            state_dict = env.reset()
            total_loss = 0
            frames = []
            for i in range(ep_len):
                state_tensor = dict_to_torch(state_dict).cuda()
                action = policy_model.forward_to_dictionary(state_tensor)
                state_dict, reward, terminated, info = env.step(action)
                total_loss += -1 * (np.sum(dict_to_numpy(reward)))
                # print("validation state reward:", epoch, i, state_dict, reward)
            epoch_loss = total_loss/ep_len/3 #todo make num agents dynamic
            # print("epoch_loss", epoch, epoch_loss)
            losses.append(epoch_loss)
            #convert actions to dictionary
    return np.mean(losses)

def plot_dynamics_vs_environment(env, dynamics_model, policy_model):
    #looking at agent trajectory in dynamics model vs. real environment
    import matplotlib.pyplot as plt
    import copy

    state = env.reset()
    actualx = []
    actualy = []
    actualr = []

    predictedx = []
    predictedy = []
    predictedr = []

    predicted_state = copy.deepcopy(state)
    predicted_state = dict_to_torch(predicted_state)
    state_arr = dict_to_numpy(state)

    actualx = state_arr[:,2].reshape(-1, 1)
    actualy = state_arr[:,3].reshape(-1, 1)
    predictedx = state_arr[:,2].reshape(-1, 1)
    predictedy= state_arr[:,3].reshape(-1, 1)
    
    # predicted_state = predicted_state[None, :,:]
    frames = []
    
    policy_model = policy_model.cpu()
    dynamics_model = dynamics_model.cpu()
    reward_dynamics = 0
    reward_env = 0
    with torch.no_grad():
        for i in range(ep_len):
            # state_tensor = dict_to_torch(state_dict).cuda()
            action = policy_model.forward_to_dictionary(predicted_state)
            action_tensor = dict_to_torch(action)
            predicted_state, predicted_rewards = dynamics_model.sample_next_state_reward(predicted_state, action_tensor)
            
            # s_next_mean = predicted_state[0]
            predictedx = np.concatenate((predictedx, predicted_state[:,2].numpy().reshape(-1, 1)), axis = 1)
            predictedy = np.concatenate((predictedy, predicted_state[:,3].numpy().reshape(-1, 1)), axis = 1)
            # print(reward_dynamics)
            reward_dynamics += torch.mean(predicted_rewards).item()
            # print("dynamics reward", predicted_rewards, torch.mean(predicted_rewards).item())
            predictedr.append(reward_dynamics)

            obs, reward, terminated, info = env.step(action)
            env.observation_space(agent)

            obs_arr = np.matrix(dict_to_numpy(obs))
            actualx = np.concatenate((actualx, obs_arr[:,2]), axis = 1)
            actualy = np.concatenate((actualy, obs_arr[:,3]), axis = 1)

            reward_arr = dict_to_numpy(reward)
            # print("env reward", reward_arr, np.mean(reward_arr))
            reward_env += np.mean(reward_arr)
            actualr.append(reward_env)

    plt.plot(0)
    plt.plot(actualx[0].T, actualy[0].T, label = "agent 1 actual")
    plt.plot(actualx[1].T, actualy[1].T, label = "agent 2 actual")
    plt.plot(actualx[2].T, actualy[2].T, label = "agent 3 actual")
    plt.plot(predictedx[0].T, predictedy[0].T, label = "agent 1 predicted")
    plt.plot(predictedx[1].T, predictedy[1].T, label = "agent 2 predicted")
    plt.plot(predictedx[2].T, predictedy[2].T, label = "agent 3 predicted")
    plt.legend()
    plt.savefig('position_plot.png')
    plt.show()

    plt.plot(1)
    x1 = np.array([range(len(predictedr))]).transpose()
    plt.plot(x1, predictedr, label = "r predicted")
    plt.plot(x1, actualr, label = "r actual")
    plt.legend()
    plt.savefig('reward_plot.png')
    plt.show()

    env.close()

def make_model_dir():
    current_time = datetime.now()
    str_date_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")
    print("Current timestamp", str_date_time)
    folder = "..\\model_checkpoints\\multiagent_policy"
    run_dir = folder + "\\multiagent_policy_" + str_date_time + "_" + str(env_name) + '_wd_' +str(wd)+'_lr_'+str(lr)+'_hdim_' + str(h_dim)
    os.mkdir(run_dir)
    filepath = run_dir + '\\multiagent_policy_'
    return filepath


if __name__ == '__main__':
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\copy_transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_10_26_2022_07_51_40_simple_spread_v2_wd_0.001_lr_0.0001_hdim_512\\transformer_Decay_best_loss_-1.8287531244372266.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_18_2022_13_25_34_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.749979948712691.pth"
    # PATH = "..\\biorobotics\\all_checkpoints_transformer\\transformer_01_25_2023_02_10_05_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.724443092141849.pth"
    # PATH = "..\\biorobotics\\all_checkpoints_transformer\\transformer_01_25_2023_12_19_44_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-7.2226179268924895.pth"
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

    
    batch_size = 100
    
    state_dict = env.reset()
    agents = env.agents
    
    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)

    s_dim = dict_to_torch(state_dict).shape[1]
    a_dim = action_tensor.shape[1]
    # d_model = 64
    # h_dim = 256
    d_model = 256
    h_dim = 256
    n_epochs = 5000

    dynamics_model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    dynamics_model.load_state_dict(torch.load(PATH)['model_state_dict'])
    dynamics_model = dynamics_model.cuda()

    n_epochs = 5

    policy_model = MultiAgentPolicyModel(dynamics_model, agents, s_dim, a_dim, d_model, h_dim)
    policy_model = policy_model.cuda()

    lr = 1e-4
    wd = 0.001 #weight decay
    opt = torch.optim.Adam(policy_model.parameters(), lr=lr, weight_decay=wd) #which opt to use?

    ep_len = 100
    train_epochs = 10
    test_epochs = 10

    min_test_loss = 10**10
    filepath = make_model_dir()

    for i in range(n_epochs):

        states = []
        for j in range(batch_size):
            state_dict = env.reset()
            states.append(dict_to_torch(state_dict))
        states = torch.stack(states)
        states = states.cuda()

        train_loss = train(model=policy_model, dynamics_model=dynamics_model, start_states=states, opt=opt, 
                                exploration = 0.1, ep_len=ep_len, epochs=test_epochs, device=device)
        test_loss = validate(policy_model=policy_model, env=env, ep_len=ep_len, epochs=test_epochs)

        if comet:
            experiment.log_metric("train_loss", train_loss, step=i)
            experiment.log_metric("test_loss", test_loss, step=i)
        print("epoch", i, " | ", "train_loss:", train_loss, "test_loss:", test_loss)

        if i % 10 == 0:
            print("hello!")
            checkpoint_path = filepath + 'Decay.pth'
            torch.save({
                        'model_state_dict': policy_model.state_dict(),
                        'model_optimizer_state_dict': opt.state_dict(),
                        }, checkpoint_path)

        if test_loss < min_test_loss:
            # print('Min Test Loss:', min_test_loss)
            checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
            if min_test_loss != 10**10:
                os.rename(filepath + "Decay_best_loss_" + str(min_test_loss) +".pth", checkpoint_path)
            min_test_loss = test_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

            checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
            torch.save({'model_state_dict': policy_model.state_dict(),
                'model_optimizer_state_dict': opt.state_dict()}, checkpoint_path)
    
    for i,layer in enumerate(policy_model.children()):
        # if isinstance(layer, nn.Linear):
        #     print("linear layer", i, layer.state_dict())
        # if isinstance(layer, nn.TransformerEncoderLayer):
        print("layer", i, layer.state_dict())

    # plot_dynamics_vs_environment(env, dynamics_model, policy_model)

    # with torch.no_grad():
    #     for i in range(5):
    #         state_dict = env.reset()
    #         state = dict_to_torch(state_dict)
    #         agents = env.agents

    #         frames = []
    #         for j in range(100):
    #             env.render()
    #             frames.append(env.render(mode="rgb_array"))
    #             action = policy_model.forward_to_dictionary(state)
    #             obs, reward, terminated, info = env.step(action)
    #             state = dict_to_torch(obs)

    #         save_frames_as_gif(frames, filename='gym_animation_' + str(i) + '.gif')
    
    
    env.close()
    

    

        