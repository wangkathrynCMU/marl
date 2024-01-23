import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import csv
import gymnasium as gym

# Custom callback for logging actions and rewards
class MyCustomCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # Define the file path
        self.file_path = os.path.join('logs', 'obs_actions_rewards_log.csv')
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Open the file and write the headers
        with open(self.file_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode ID", "Step", "Agent ID", "Observation", "Action", "Reward"])

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Log observations, actions, and rewards for each agent
        with open(self.file_path, "a", newline='') as file:
            writer = csv.writer(file)
            for agent_id in episode.last_info_for().keys():
                last_obs = episode.last_observation_for(agent_id)
                last_action = episode.last_action_for(agent_id)
                last_reward = episode.last_reward_for(agent_id)
                writer.writerow([episode.episode_id, episode.length, agent_id, last_obs, last_action, last_reward])

# Register the PettingZoo environment
def env_creator(env_config):
    env = simple_spread_v3.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])
    print("reset!!", env.reset())
    return ParallelPettingZooEnv(env)

register_env("simple_spread_v3", env_creator)

env_config = {
    'N': 3, 
    'local_ratio': 0.5, 
    'max_cycles': 100, 
    'continuous_actions': True
}

# Configure the PPO algorithm using PPOConfig
# disable_env_checking=True) \
config = PPOConfig() \
    .environment(env="simple_spread_v3", env_config=env_config) \
    .training(
        use_gae=True,
        lambda_=0.95,
        kl_coeff=0.2,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lr=5e-4,
        entropy_coeff=0.01,
        clip_param=0.2,
        vf_loss_coeff=1.0,
        checkpoint_freq=10,
        checkpoint_at_end=True
    ) \
    .resources(num_gpus=1) \
    .rollouts(num_rollout_workers=4) \
    .callbacks(MyCustomCallbacks) \
    .debugging(log_level="INFO") 

# Initialize Ray
ray.init()

# Build and train the PPO algorithm
NUM_TRAINING_ITERATIONS = 100
algo = config.build()
for i in range(NUM_TRAINING_ITERATIONS):
    print("Iteration", i)
    algo.train()

# Shut down Ray
ray.shutdown()
