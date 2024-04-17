from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3
import torch
import imageio
import numpy as np

from PIL import Image
from datetime import datetime

def initialize_env():
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
                    continuous_actions = env_config['continuous_actions'],
                    render_mode='rgb_array')    
    return env

def dist(a, b):
    return math.sqrt(a*a + b*b)

def save_frames_as_gif(frames, filename='output.gif', duration=5):
    """Save a list of frames as a gif"""
    imageio.mimsave(filename, frames, duration=duration)

def torch_to_dict(tensor, agents):
    output = dict()
    index = 0

    for agent in agents:
        output[agent] = tensor[index]
        index +=1
    
    return output


def dict_to_numpy(dictionary):
    matrix = []
    for key in dictionary:
        # print("key, dict[key]", key, dictionary[key])
        matrix.append(dictionary[key])
    return np.array(matrix)

def dict_to_torch(dictionary):
    return torch.tensor(dict_to_numpy(dictionary))

def CEM_vis(env, a_mean, a_sig):
    frame_list = []
    reward_sum = 0
    t = 0
    dones = {agent: False for agent in env.agents}
    actions = []
    env_states = []

    while not all(dones.values()):
        a_shape = a_mean[t].shape
        action = a_mean[t] + a_sig[t]*torch.randn(a_shape)
        action = torch.clamp(action,min=0,max=1)
        
        obs, rewards, dones, _, _ = env.step(torch_to_dict(action, env.agents))
        if all(dones.values()):
            break
        reward_sum += sum(rewards.values())
        env_states.append(dict_to_torch(obs))
        # Render the frame and append to the list
        frame = env.render()
        frame_list.append(Image.fromarray(frame))
        actions.append(action)
        t +=1
    
    print(f"Total Reward: {reward_sum}")
    print(f"Total Steps: {t}")
    print(f"Reward Avg: {reward_sum/3/t}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CEM_vis_{timestamp}.gif"

    frame_list[0].save(filename, save_all=True, append_images=frame_list[1:], duration=3, loop=0)
    return  torch.stack(env_states), torch.stack(actions)


if __name__ == '__main__':
    PATH = "C:\\Users\\kkwan\\devDir\\model_checkpoints\\multiagent_policy\\multiagent_policy_04_12_2023_12_38_48_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_policy_Decay_best_loss_1.9817438017410454.pth"
    
    env = initialize_env()
    obs, infos  = env.reset()
    agents = env.agents

    env_name = "simple_spread_v3"

    frame_list = []
    reward_sum = 0
    # print(len(env.reset()))
    obs, _ = env.reset()
    dones = {agent: False for agent in env.agents}
    while not all(dones.values()):
        # actions = {agentID: agent.compute_single_action(obs[agent], policy_id=agentID) for agentID in env.agents if not dones[agent]}
        a_mean = torch.tensor([[0.7455, 0.2588, 0.1484, 0.2024, 0.2805],
        [0.3118, 0.4567, 0.5087, 0.2351, 0.6696],
        [0.3131, 0.7137, 0.1214, 0.2107, 0.1445]])
        a_sig = torch.tensor([[5.5041e-02, 1.1223e-04, 2.0688e-03, 9.6651e-03, 2.0562e-02],
        [1.1288e-01, 2.9873e-03, 3.0426e-03, 1.5107e-03, 9.4772e-04],
        [2.7333e-03, 2.9553e-02, 6.3297e-03, 6.3907e-03, 3.4539e-03]])
        a_shape = a_mean.shape
        
        actions = a_mean + a_sig*torch.randn(a_shape)
        actions = torch.clamp(actions,min=0,max=1)
        
        obs, rewards, dones, _, _ = env.step(torch_to_dict(actions, env.agents))
        reward_sum += sum(rewards.values())

        # Render the frame and append to the list
        frame = env.render()
        frame_list.append(Image.fromarray(frame))

    print(f"Total Reward: {reward_sum}")

    # Save the frames as a GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CEM_vis_{timestamp}.gif"

    frame_list[0].save(filename, save_all=True, append_images=frame_list[1:], duration=3, loop=0)