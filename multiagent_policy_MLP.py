import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Normal
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


import gym
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3
from gym.vector import SyncVectorEnv

from comet_ml import Experiment

from datetime import datetime

# from multiagent_dynamics_model import MultiAgentDynamicsModel
from ma_state_dynamics_model import MAStateModel
from reward_model_transformer import RewardDynamicsTransformer


folder = "model_checkpoints\\multiagent_policy_MLP"

comet = True
if comet:
    experiment = Experiment(
    api_key="s8Nte4JHNGudFkNjXiJ9RpofM",
    project_name="multiagent-policy-mlp",
    workspace="wangkathryncmu"
    )

saving = True

device = 'cpu'

if torch.cuda.is_available():
    device = torch.device('cuda:0')

env_config = {
        'N':3, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }

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
    def __init__(self, s_dim, a_dim, d_model, dim_feedforward, nhead, num_layers, h_dim):
        super(MultiAgentPolicyModel, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Linear(s_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, h_dim),
                        nn.ReLU()
                    )
        self.meanlayer = nn.Sequential(
                        nn.Linear(h_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, a_dim)
        )
        self.stdevlayer = nn.Sequential(
                        nn.Linear(h_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, a_dim),
                        nn.Softplus()
        )
        
    def forward(self,state):
        feats = self.layer1(state)
        a_next_mean = self.meanlayer(feats)
        a_next_stdev = self.stdevlayer(feats)
        
        return a_next_mean, a_next_stdev
    
    def forward_to_dictionary(self, state, agents):
        # i think this only works if only one state is passed in
        
        a_next_mean, a_next_sig = self.forward(state)
        a_next_distribution = Normal.Normal(a_next_mean, a_next_sig)
        actions = a_next_distribution.sample().cpu().numpy()

        output = dict()
        index = 0
        for agent in agents:
            output[agent] = actions[index]
            index +=1

        return output

def train(model, state_model, reward_model, opt, env, batch_size, ep_len, episodes, n_agents, device):
    model.to(device)
    state_model.to(device)
    reward_model.to(device)
    model.train()
    losses = []

    for episode in range(episodes):

        states = []

        for j in range(batch_size):
            state_dict, _ = env.reset()
            states.append(dict_to_torch(state_dict))

        states = torch.stack(states)
        states = states.cuda()

        cumulative_loss = 0
        
        for t in range(ep_len):
            a_next_mean, a_next_sig = model.forward(states)
            a_next_distribution = Normal.Normal(a_next_mean, a_next_sig)
            actions = a_next_distribution.rsample()

            next_states = state_model.sample_next_state(states, actions)
            rewards = reward_model.sample_next_reward(states, actions)
            
            loss_t = -rewards.sum()/ (n_agents * batch_size)
            cumulative_loss += loss_t
            states = next_states
        # Normalize the loss by the number of steps
        normalized_loss = cumulative_loss/ ep_len
        # print("normalized loss", normalized_loss)
        opt.zero_grad()
        normalized_loss.backward()
        opt.step()
        
        # Recording mean loss per epoch
        epoch_loss = normalized_loss.item()
        print(f"Episode {episode+1}/{episodes} - Loss: {epoch_loss}")
        losses.append(epoch_loss)

    return np.mean(losses)

def validate(model, env, ep_len, n_epochs, device, n_agents, agents):
    model.to(device)
    losses = []

    with torch.no_grad():
        for epoch in range(n_epochs):    
            state_dict, _ = env.reset()
            total_loss = 0
            for i in range(ep_len):
                # Convert the state dictionary to a tensor and pass it to the model
                state_tensor = dict_to_torch(state_dict).float().to(device)
                action_dict = model.forward_to_dictionary(state_tensor, agents)

                # Take a step in the environment using the action dictionary
                next_state_dict, reward_dict, terminated, _, _ = env.step(action_dict)

                # Compute loss; rewards are negative of the loss
                total_loss += -sum(reward_dict.values())

                # Check if the episode is terminated
                if all(terminated.values()):
                    break

                state_dict = next_state_dict

            # Calculate the average loss per step for the epoch
            epoch_loss = total_loss / (i+1) / n_agents  # Ensure this is the correct loss calculation for your scenario
            losses.append(epoch_loss)

    return np.mean(losses)


def plot_dynamics_vs_environment(env, dynamics_model, policy_model):
    #looking at agent trajectory in dynamics model vs. real environment
    import matplotlib.pyplot as plt
    import copy

    state, info = env.reset()
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
    
    policy = policy.cpu()
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

def checkpoints_path(folder):
    current_time = datetime.now()
    str_date_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")
    print("Current timestamp", str_date_time)

    run_dir = folder + "\\policy_MLP_" + str_date_time + "_" + str(env_name)
    os.mkdir(run_dir)

    filepath = run_dir + '\\policy_MLP_'

    return run_dir, filepath


if __name__ == '__main__':
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_04_2022_12_26_39_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\copy_transformer_Decay_best_loss_-1.799452094718725.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_10_26_2022_07_51_40_simple_spread_v2_wd_0.001_lr_0.0001_hdim_512\\transformer_Decay_best_loss_-1.8287531244372266.pth"
    # PATH = ".\\all_checkpoints_transformer\\transformer_11_18_2022_13_25_34_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.749979948712691.pth"
    # PATH = "..\\biorobotics\\all_checkpoints_transformer\\transformer_01_25_2023_02_10_05_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-5.724443092141849.pth"
    # PATH = "..\\biorobotics\\all_checkpoints_transformer\\transformer_01_25_2023_12_19_44_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\transformer_Decay_best_loss_-7.2226179268924895.pth"
    # PATH = "..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
    
    # MA PATHS
    PATH = "model_checkpoints\\ma_state_dynamics\\ma_state_dynamics_10_31_2023_10_38_13_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.208150715827943.pth"
    R_PATH = "model_checkpoints\\reward_dynamics\\reward_dynamics_transformer_10_31_2023_10_39_02_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-2.2542934103572403.pth"
    
    env_name = 'simple_spread_v3'
    

    env = simple_spread_v3.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])    

    state_dict, _ = env.reset()
    agents = env.agents
    
    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)

    s_dim = dict_to_torch(state_dict).shape[1]
    a_dim = action_tensor.shape[1]
    print("action dim", a_dim)
    n_agents = len(state_dict.keys())
    print("n_agents: ", n_agents)

    batch_size = 128

    # STATE MODEL
    s_d_model = 256
    s_h_dim = 256
    
    state_model = MAStateModel(s_dim, a_dim, s_d_model, s_h_dim)
    state_model.load_state_dict(torch.load(PATH)['model_state_dict'])
    state_model = state_model.cuda()

    # REWARD MODEL
    r_d_model = 256
    r_dim_feedforward = 512
    r_nhead = 8
    r_num_layers = 1

    reward_model = RewardDynamicsTransformer(s_dim, a_dim, r_d_model, r_dim_feedforward, r_nhead, r_num_layers)
    reward_model.load_state_dict(torch.load(R_PATH)['model_state_dict'])
    reward_model = reward_model.cuda()

    state_model.eval()
    for param in state_model.parameters():
        param.requires_grad = False

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    d_model = 256
    dim_feedforward = 256
    nhead = 4
    num_layers = 2
    h_dim = 512

    policy = MultiAgentPolicyModel(s_dim, a_dim, d_model, dim_feedforward, nhead, num_layers, h_dim)
    policy = policy.cuda()

    lr = 1e-6
    wd = 0. #0.001 #weight decay
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=wd) #which opt to use?

    ep_len = 25
    train_episodes = 10
    test_episodes = 10

    min_test_loss = 10**10

    if saving:
        runpath, filepath = checkpoints_path(folder)
        if comet:
            writer = SummaryWriter(runpath + '\\tensorboard_log')

        # print("Saving " + filepath + "train_data.pt ...")
        # torch.save(train_data, filepath + 'train_data.pt')
        # print("Saved")

    n_epochs = 5000

    try:
        for i in range(n_epochs):
            train_loss = train(model=policy, state_model=state_model, reward_model=reward_model, opt=opt, 
                    env=env, batch_size=batch_size, ep_len=ep_len, episodes=train_episodes, 
                    n_agents=n_agents, device=device)
            
            print("epoch", i, " | ", "train_loss:", train_loss)

            if i%10 == 0:
                test_loss = validate(model=policy, env=env, ep_len=ep_len, n_epochs=test_episodes, device=device, n_agents=n_agents, agents=agents)
                print("test_loss:", test_loss)

            for name, param in policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, i)
                else:
                    print(f'{name}.requires_grad and param.grad are None')

            if comet:
                print("writing", i)
                writer.add_scalar('training loss', train_loss, i)
                writer.add_scalar('test loss', test_loss, i)
                experiment.log_metric("train_loss", train_loss, step=i)
                experiment.log_metric("test_loss", test_loss, step=i)

            if saving:
                if i % 10 == 0:
                    checkpoint_path = filepath + 'Decay.pth'
                    torch.save({
                                'model_state_dict': policy.state_dict(),
                                'model_optimizer_state_dict': opt.state_dict(),
                                }, checkpoint_path)

                if test_loss < min_test_loss:
                    checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
                    if min_test_loss != 10**10:
                        os.rename(filepath + "Decay_best_loss_" + str(min_test_loss) +".pth", checkpoint_path)
                    min_test_loss = test_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

                    checkpoint_path = filepath + "Decay_best_loss_" + str(test_loss) +".pth"
                    torch.save({'model_state_dict': policy.state_dict(),
                        'model_optimizer_state_dict': opt.state_dict()}, checkpoint_path)
    except KeyboardInterrupt:
        print('Training interrupted, closing TensorBoard writer...')
    finally:
        writer.close()
        print('TensorBoard writer closed.')
    
    env.close()