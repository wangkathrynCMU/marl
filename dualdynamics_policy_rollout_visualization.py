from multiagent_policy import MultiAgentPolicyModel
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

if __name__ == '__main__':
    PATH = "C:\\Users\\kkwan\\devDir\\model_checkpoints\\policy_transformer\\policy_transformer_11_14_2023_13_01_50_simple_spread_v3\policy_transformer_Decay_best_loss_1.1950670533457337.pth"
    PATH = "C:\\Users\\kkwan\\devDir\\model_checkpoints\\policy_transformer\\policy_transformer_11_07_2023_15_32_24_simple_spread_v3\policy_transformer_Decay_best_loss_0.7637799317784137.pth"
    
    env = initialize_env()
    s_dim = 18
    a_dim = 5
    
    d_model = 256
    dim_feedforward = 256
    nhead = 4
    num_layers = 2
    h_dim = 128

    policy = MultiAgentPolicyModel(s_dim, a_dim, d_model, dim_feedforward, nhead, num_layers, h_dim)
    policy.load_state_dict(torch.load(PATH)['model_state_dict'])
    policy.eval()
    
    # state_tensor = dict_to_torch(state)
    # state_tensor = state_tensor[None, :]

    frame_list = []
    obs, _ = env.reset()
    agents = obs.keys()
    dones = {agent: False for agent in env.agents}

    with torch.no_grad():
        while not all(dones.values()):
            actions = policy.forward(dict_to_torch(obs))
            action_dict = torch_to_dict(actions, agents)

            obs, rewards, dones, _, _ = env.step(action_dict)
            frame = env.render()
            frame_list.append(Image.fromarray(frame))

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"customPolicy_trajectory_visualization_{timestamp}.gif"

    frame_list[0].save(filename, save_all=True, append_images=frame_list[1:], duration=3, loop=0)

