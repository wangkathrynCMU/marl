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
from pettingzoo.mpe import simple_spread_v3

from comet_ml import Experiment

from datetime import datetime


folder = "..\\model_checkpoints\\single_agent_state_dynamics"

comet = True
saving = True
experiment = None

class MAStateModel(nn.Module):
    def __init__(self, s_dim, a_dim, d_model, h_dim):
        super(MAStateModel, self).__init__()


        # State Predictions
        self.pre_transformer = nn.Sequential(
                        nn.Linear(s_dim+a_dim, d_model),
                        nn.ReLU())

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward = 512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        
        self.meanlayer = nn.Sequential(
                        nn.Linear(d_model, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, s_dim)
        )
        self.stdevlayer = nn.Sequential(
                        nn.Linear(d_model, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, s_dim),
                        nn.Softplus()
        )
        
    def forward(self,state,action):
        '''
        Concatenates the state and action, passes it thru network to get a mean and stand dev vector
        What I would recommend is actually having s_next_mean be calculated as state+delta_mean, so that the model is learning deltas and they're always close to 0
        '''
        # directly return mean reward, stdev is factored into training for loss function -- can i just do this in the custom env?
        # have output be sdim+1 -- targets are next state + reward
        state_action = torch.cat((state,action),dim=-1)
        pre_transformer = self.pre_transformer(state_action)
        feats = self.transformer_encoder(pre_transformer)

        s_delta_mean = self.meanlayer(feats)
        s_next_mean = s_delta_mean + state

        s_next_stdev = self.stdevlayer(feats)
        
        return s_next_mean,s_next_stdev

    def get_loss(self,states,actions,next_states):
        '''
        Passes batch of states and actions through dynamics model, then evaluates log likelihood of predictions
        
        states: batch_size x s_dim
        actions: batch_size x a_dim
        next_states: batch_size x s_dim

        '''
        s_next_mean, s_next_sig = self.forward(states, actions)
        s_next_distribution = Normal.Normal(s_next_mean, s_next_sig)
        loss = - torch.mean(s_next_distribution.log_prob(next_states))
        mseloss = torch.nn.MSELoss()(next_states, s_next_mean)

        return loss, mseloss
    
    def sample_next_state(self, states, actions):
        s_next_mean, s_next_sig = self.forward(states, actions)
        next_state_distribution = Normal.Normal(s_next_mean, s_next_sig)
        next_state = next_state_distribution.sample()
        
        return next_state

    
    def sample_next_dict(self, states, actions, agents):
        next_state = self.sample_next_state(states, actions)
        
        output = dict()
        index = 0
        for agent in agents:
            output[agent] = next_state[index]
            index +=1

        return output

def train(model, opt, device):
    losses = []
    mse_losses = []
    
    for batch_i, (states, actions, next_states) in enumerate(train_loader):
        if torch.cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            next_states = next_states.cuda()
        
        # clear the gradients
        opt.zero_grad()

        # calculate loss
        loss, mseloss = model.get_loss(states, actions, next_states) 
        loss.backward()
        opt.step()

        losses.append(loss.item())
        mse_losses.append(mseloss.item())

    return np.mean(losses), np.mean(mse_losses)

def validate(model, device):
    losses = list()
    mselosses = list()

    with torch.no_grad():
        for batch_i, (states, actions, next_states, ) in enumerate(test_loader):
            if torch.cuda.is_available():
                states = states.cuda()
                actions = actions.cuda()
                next_states = next_states.cuda()
            loss, mseloss = model.get_loss(states, actions, next_states)
            losses.append(loss.item())
            mselosses.append(mseloss.item())
    return np.mean(losses), np.mean(mselosses)

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

def gather_data(N,ep_len, env_name, env_config, trainer=None, noise = 0):
    env = None
    if env_name == 'simple_spread_v2':
        env = simple_spread_v2.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])
    if env_name == 'simple_spread_v3':
        env = simple_spread_v3.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])

    state, info = env.reset()
    agents = env.agents
    action = sample_action(env, agents)
    
    states = []
    actions = []
    next_states = []
    rewards = []
    for i in range(N):
        state, info = env.reset()
        for j in range(ep_len):
            states.append(dict_to_torch(state))

            action = sample_action(env, agents)
            action_tensor = dict_to_torch(action)

            # this is necessary because discrete action space has dimension 1 for each agent (0-4 for each action option)
            # but continuous action space has dimension 4 (each value represents velocity in each direction)
            if env_config['continuous_actions']:
                actions.append(action_tensor)
            else:
                actions.append(action_tensor.reshape(action_tensor.size(0), 1)) 
            obs, reward, _, _, _ = env.step(action)
            next_states.append(dict_to_torch(obs))
            reward_tensor = dict_to_torch(reward) 
            rewards.append(reward_tensor.reshape(reward_tensor.size(0), 1))

            state = obs
            
    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)
    rewards = torch.stack(rewards)
    
    env.close()
    return states, actions, next_states, rewards

def checkpoints_path(folder):
    current_time = datetime.now()
    str_date_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")
    print("Current timestamp", str_date_time)

    run_dir = folder + "\\ma_state_dynamics_" + str_date_time + "_" + str(env_name)
    os.mkdir(run_dir)

    filepath = run_dir + '\\ma_state_dynamics_'

    return filepath

if __name__ == '__main__':
        
    if comet:
        experiment  = Experiment(
            api_key="s8Nte4JHNGudFkNjXiJ9RpofM",
            project_name="ma-state-dynamics",
            workspace="wangkathryncmu"
            )
    n_agents = 1
    N = 1000    
    deviceID = 'cuda:0'
    ep_len = 100
    env_name = 'simple_spread_v3'
    env_config = {
        'N':1, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }
    # QUESTION: Max Cycles?

    device = torch.device(deviceID)
    if torch.cuda.is_available():
        print("CUDA Available")

    states,actions,next_states, rewards = gather_data(N, ep_len, env_name, env_config)
    dataset = torch.utils.data.TensorDataset(states,actions,next_states)


    test_split = .2
    data_size = states.shape[0]
    N_train = int((1-test_split)*data_size)
    N_test = data_size - N_train
    batch_size = 100

    train_data, test_data = torch.utils.data.random_split(dataset, [N_train, N_test])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=True)

    d_model = 256
    h_dim = 256
    n_epochs = 5000
    s_dim = states.shape[2]
    if env_config['continuous_actions']:
        a_dim = actions.shape[2]
    else:
        a_dim = 1

    model = MAStateModel(s_dim,a_dim,d_model,h_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # defining optimizer
    lr = 1e-4
    wd = 0.001 #weight decay
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    decay = True

    lr_decay = 0.1
    lr_decay_epochs_interval = 100
    if decay == True:
    	lr_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, 1, lr_decay)
      
    if saving:
        filepath = checkpoints_path(folder)

        print("Saving " + filepath + "train_data.pt ...")
        torch.save(train_data, filepath + 'train_data.pt')
        print("Saved")

    min_test_loss = 10**10

    if comet:
        experiment.log_parameters({"learning_rate": lr,
                                    "learning_rate_decay": lr_decay,
                                    "weight decay": wd,
                                    "decay_bool": decay,
                                    "n_agent": n_agents,
                                    "num_epochs": n_epochs,
                                    "batch_size": batch_size,
                                    "hidden_layer_dim": h_dim,
                                    "environment": env_name,
                                    "data_N": N,
                                    "data_episode_len": ep_len,
                                    "cuda": deviceID,
                                    "filepath": filepath})

    
    for i in range(n_epochs):
        print("epoch:", i)
        if (i+1)%lr_decay_epochs_interval == 0 and decay == True:
            lr_scheduler.step()
            
        train_loss, train_mse = train(model, model_optimizer, device = device)
        print("train_loss, train_mse", train_loss, train_mse)
        if comet:
            experiment.log_metric("train_loss", train_loss, step=i)
            experiment.log_metric("train_mse", train_mse, step=i)
            

        if (i+1) % 5 == 0:
            test_loss, test_mse = validate(model, device = device)
            print("test_loss, test_mse", test_loss, test_mse)
            if comet:
                experiment.log_metric("test_loss", test_loss, step=i)
                experiment.log_metric("test_mse", test_mse, step=i)
        
        
        if saving:
            if i % 10 == 0:
                checkpoint_path = filepath + 'Decay.pth'
                torch.save({
                            'model_state_dict': model.state_dict(),
                            'model_optimizer_state_dict': model_optimizer.state_dict(),
                            }, checkpoint_path)

            if (i+1) % 5 == 0 and test_loss < min_test_loss:
                # print('Min Test Loss:', min_test_loss)
                checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
                if min_test_loss != 10**10:
                    os.rename(filepath + "Decay_best_loss_" + str(min_test_loss) +".pth", checkpoint_path)
                min_test_loss = test_loss

                checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
                torch.save({'model_state_dict': model.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict()}, checkpoint_path)

        
