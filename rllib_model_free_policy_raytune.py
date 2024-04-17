import os
import csv
from datetime import datetime
import ray
# import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from torch import nn
import wandb

from ray.air.integrations.wandb import WandbLoggerCallback


from pettingzoo.mpe import simple_spread_v3

class MyCustomCallbacks(DefaultCallbacks):
    pass
    # def __init__(self):
    #     super().__init__()
    #     # Define the file path
    #     self.file_path = os.path.join('logs', 'obs_actions_rewards_log.csv')
    #     os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
    #     # Open the file and write the headers
    #     with open(self.file_path, "w", newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["Episode ID", "Step", "Agent ID", "Observation", "Action", "Reward"])

    # def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
    #     # Log observations, actions, and rewards for each agent
    #     with open(self.file_path, "a", newline='') as file:
    #         writer = csv.writer(file)
    #         for agent_id in episode.agent_rewards.keys():
    #         # Fetch last observation, action, and reward for each agent
    #             last_obs = episode.last_obs_for(agent_id)
    #             last_action = episode.last_action_for(agent_id)
    #             last_reward = episode.last_reward_for(agent_id)
    #             writer.writerow([episode.episode_id, episode.length, agent_id, last_obs, last_action, last_reward])

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
                    continuous_actions = env_config['continuous_actions'])
    return env

if __name__ == "__main__":
    wandb.init(project="rllib-MAPPO-simplespread", entity="biorobotics-marl")

    ray.init()

    env_name = "simple_spread_v3"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-4,
            gamma=0.9,
            # lambda_=0.9,
            use_gae=True,
            # clip_param=0.4,
            grad_clip=0.5,
            # entropy_coeff=0.1,
            # vf_loss_coeff=0.25,
            # sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        # .callbacks(MyCustomCallbacks)
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # local_dir = os.path.expanduser("~/ray_results/simple_spread_v3")
    local_dir = "C:\\Users\\kkwan\\ray_results\\simple_spread_v3"

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir= local_dir,
        config=config.to_dict(),
        callbacks=[WandbLoggerCallback(
        project="rllib-MAPPO-simplespread",
        entity= "biorobotics-marl",
        api_key="5c8d2be2d372d7685da285d9477c9f2e90577628",
        log_config=True
        )]
    )