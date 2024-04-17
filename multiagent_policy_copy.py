# compare trajectories during training
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


folder = "model_checkpoints\\single_agent_policy"

comet = True
if comet:
    experiment = Experiment(
        api_key="s8Nte4JHNGudFkNjXiJ9RpofM",
        project_name="single-agent-policy-transformer",
        workspace="wangkathryncmu"
        )
    # experiment = Experiment(
    #     api_key="s8Nte4JHNGudFkNjXiJ9RpofM",
    #     project_name="marl-policy",
    #     workspace="wangkathryncmu",
    # )

saving = True

device = 'cpu'

if torch.cuda.is_available():
    device = torch.device('cuda:0')

env_config = {
        'N':1, 
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

        self.pre_transformer = nn.Sequential(
                        nn.Linear(s_dim, d_model),
                        nn.ReLU())

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward = dim_feedforward, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.meanlayer = nn.Sequential(
                        nn.Linear(d_model, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, a_dim),
                        nn.Sigmoid()
        )

        # self.stdevlayer = nn.Sequential(
        #                 nn.Linear(h_dim, 256),
        #                 nn.ReLU(),
        #                 nn.Linear(256, 256),
        #                 nn.Softplus()
        # )
        
    def forward(self,state):
        pre_transformer = self.pre_transformer(state)
        feats = self.encoder_layer(pre_transformer)
        feats = self.transformer_encoder(feats)

        mean = self.meanlayer(feats)
        # stdev = self.stdevlayer(feats)
        
        return mean #, stdev
    
    def forward_to_dictionary(self, state, agents):
        # i think this only works if only one state is passed in
        action = self.forward(state).cpu().numpy()

        output = dict()
        index = 0
        for agent in agents:
            output[agent] = action[index]
            index +=1

        return output

def log_trajectories_to_wandb(real_traj, predicted_traj, episode, phase):
    # Convert trajectories to numpy arrays
    real_traj = np.array(real_traj)
    predicted_traj = np.array(predicted_traj)

    # Plot trajectories
    fig, ax = plt.subplots()
    ax.plot(real_traj[:, 0], real_traj[:, 1], label='Real')
    ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], label='Predicted')
    ax.legend()

    # Log the plot to WandB
    wandb.log({f"{phase}_episode_{episode}": [wandb.Image(plt, caption="Trajectories")]})
    plt.close(fig)
    
def train(model, state_model, reward_model, opt, env, batch_size, ep_len, episodes, n_agents, device):
    model.to(device)
    state_model.to(device)
    reward_model.to(device)
    model.train()
    losses = []

    for episode in range(episodes):

        states = []

        real_traj = []
        predicted_traj = []

        for j in range(batch_size):
            state_dict, _ = env.reset()
            states.append(dict_to_torch(state_dict))

        states = torch.stack(states)
        states = states.cuda()

        cumulative_loss = 0

        real_traj = []
        predicted_traj = []
        real_traj.append(states.cpu().numpy())
        predicted_traj.append(states.cpu().numpy())
        
        for t in range(ep_len):
            actions = model(states)
            next_states = state_model.sample_next_state(states, actions)
            rewards = reward_model.sample_next_reward(states, actions)
            
            real_next_states, rewards, done, _, _ = env.step(torch_to_dict(actions))

            loss_t = -rewards.sum()/ (n_agents * batch_size)
            cumulative_loss += loss_t
            states = next_states
            
            real_traj.append(states.cpu().numpy())
            predicted_traj.append(dict_to_numpy(real_next_states))
        
        log_trajectories_to_wandb(real_traj, predicted_traj, episode, 'train')

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

def checkpoints_path(folder):
    current_time = datetime.now()
    str_date_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")
    print("Current timestamp", str_date_time)

    run_dir = folder + "\\policy_transformer_" + str_date_time + "_" + str(env_name)
    os.mkdir(run_dir)

    filepath = run_dir + '\\policy_transformer_'

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
    # PATH = "..\\model_checkpoints\\ma_state_dynamics\\ma_state_dynamics_10_31_2023_10_38_13_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.208150715827943.pth"
    # R_PATH = "..\\model_checkpoints\\reward_dynamics\\reward_dynamics_transformer_10_31_2023_10_39_02_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-2.2542934103572403.pth"
    
    # SINGLE AGENT PATHS
    PATH = "model_checkpoints\\single_agent_state_dynamics\\ma_state_dynamics_12_05_2023_09_56_54_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.023130450248718.pth"
    R_PATH = "model_checkpoints\\single_agent_reward_dynamics\\reward_dynamics_transformer_12_05_2023_14_04_59_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-3.798559585164234.pth"
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
    h_dim = 128

    policy = MultiAgentPolicyModel(s_dim, a_dim, d_model, dim_feedforward, nhead, num_layers, h_dim)
    policy = policy.cuda()

    lr = 1e-4
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


# def create_env(env_config):
#     return simple_spread_v3.parallel_env(N=env_config['N'], 
#                                          local_ratio=env_config['local_ratio'], 
#                                          max_cycles=env_config['max_cycles'], 
#                                          continuous_actions=env_config['continuous_actions'])


# def validate(model, ep_len, device, n_agents, n_envs=50):
#     # Create the vectorized environment using the create_env function
#     envs = [create_env for _ in range(n_envs)]
#     vec_env = SyncVectorEnv(envs)

#     model.to(device)
#     model.eval()  # Set the model to evaluation mode
#     losses = []

#     with torch.no_grad():
#         # Reset the vectorized environment
#         obs = vec_env.reset()
#         total_rewards = np.zeros(n_envs)

#         for step in range(ep_len):
#             # Convert observations to tensor
#             obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
            
#             # Generate actions for each environment instance
#             actions = model(obs_tensor)
#             actions = actions.cpu().numpy()
            
#             # Step in the vectorized environments
#             next_obs, rewards, dones, _, _ = vec_env.step(actions)
            
#             # Update total rewards
#             total_rewards += rewards

#             # Handle environments where the episode has ended
#             for i in range(n_envs):
#                 if dones[i]:
#                     total_rewards[i] = 0  # Reset the rewards for the next episode
#                     obs[i] = create_env().reset()  # Reset the environment
            
#             obs = next_obs

#         # Calculate average reward across all environments
#         average_reward = total_rewards.mean() / n_agents
#         losses.append(average_reward)

#     vec_env.close()  # Close the vectorized environment when done
#     return np.mean(losses)

# PLOTTING    