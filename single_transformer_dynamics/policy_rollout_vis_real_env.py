from multiagent_policy import MultiAgentPolicyModel
from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3
import torch
import imageio
import numpy as np

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
    D_PATH = "..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
    # PATH = "..\\model_checkpoints\\multiagent_policy\\multiagent_policy_02_26_2023_20_41_06_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_policy_Decay.pth"
    # PATH = "..\\model_checkpoints\\multiagent_policy\\multiagent_policy_02_26_2023_20_41_06_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_policy_Decay_best_loss_1.3465408047713745.pth"
    PATH = "C:\\Users\\kkwan\\devDir\\model_checkpoints\\multiagent_policy\\multiagent_policy_04_12_2023_12_38_48_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_policy_Decay_best_loss_1.9817438017410454.pth"
    # device = torch.device(deviceID)
    

    env = initialize_env()
    state = env.reset()    
    state, infos  = env.reset()

    agents = env.agents

    d_model = 64
    h_dim = 256

    s_dim = 18
    a_dim = 5

    dynamics_model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    dynamics_model.load_state_dict(torch.load(D_PATH)['model_state_dict'])
    dynamics_model.eval()
    
    model = MultiAgentPolicyModel(dynamics_model, agents, s_dim, a_dim, d_model, h_dim)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    model.eval()
    

    num_eps = 10
    ep_len = 500
    buffer = []
    frames = []
    print(state)
    state_tensor = dict_to_torch(state)
    print("STATE_TENSOR SHAPE", state_tensor.shape)
    state_tensor = state_tensor[None, :]
    print("STATE_TENSOR SHAPE", state_tensor.shape)
    with torch.no_grad():
        for i in range(ep_len):
            actions = model.forward(state_tensor)
            print("------ STEP", i, "------")
            # print("state before:", state_tensor)
            print("action", actions)
            next_state_mean, next_state_stdev = dynamics_model.forward(state_tensor, actions)
            # print("next_state_mean", next_state_mean)
            next_state_reward = torch.normal(next_state_mean, next_state_stdev)
            

            state_tensor = next_state_reward[...,:-1]
            predicted_reward = next_state_mean[...,-1:]

            action_dict = torch_to_dict(actions[0], agents)
            env.step(action_dict)
            frame = env.render()
            frames.append(Image.fromarray(frame))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"policy_rollout_{timestamp}.gif"

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=3, loop=0)

