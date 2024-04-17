from multiagent_policy import MultiAgentPolicyModel
from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_spread_v3
import torch
import imageio
import numpy as np
import ray
# import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from PIL import Image
from datetime import datetime


def env_creator(args):
    env_config = {
    'N': 3, 
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
    PATH = "C:\\Users\\kkwan\\devDir\\model_checkpoints\\multiagent_policy\\multiagent_policy_04_12_2023_12_38_48_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_policy_Decay_best_loss_1.9817438017410454.pth"
    
    env = initialize_env()
    obs, infos  = env.reset()
    agents = env.agents

    ray.init()

    # Load the trained model
    checkpoint_path = "C:\\Users\\kkwan\\ray_results\\simple_spread_v3\\PPO\\PPO_simple_spread_v3_1beab_00000_0_2024-01-23_12-44-51\\checkpoint_000100"
    # agent = PPOTrainer(env="simple_spread_v3")
    # ray.init()
    env_name = "simple_spread_v3"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    agent = PPO.from_checkpoint(checkpoint_path)

    frame_list = []
    reward_sum = 0
    # print(len(env.reset()))
    obs, _ = env.reset()
    dones = {agent: False for agent in env.agents}
    while not all(dones.values()):
        # actions = {agentID: agent.compute_single_action(obs[agent], policy_id=agentID) for agentID in env.agents if not dones[agent]}
        actions = agent.compute_single_action(dict_to_torch(obs))
        obs, rewards, dones, _, _ = env.step(torch_to_dict(actions, env.agents))
        reward_sum += sum(rewards.values())

        # Render the frame and append to the list
        frame = env.render()
        frame_list.append(Image.fromarray(frame))

    print(f"Total Reward: {reward_sum}")

    # Save the frames as a GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_visualization_{timestamp}.gif"

    frame_list[0].save(filename, save_all=True, append_images=frame_list[1:], duration=3, loop=0)