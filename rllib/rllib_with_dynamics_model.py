import os
import torch
import numpy as np

from ray import tune
from ray.tune.registry import register_env
# import the pettingzoo environment
from pettingzoo.mpe import simple_spread_v2
# import rllib pettingzoo interface
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from multiagent_custom_env import MultiagentEnvWrapper
from multiagent_dynamics_model import MultiAgentDynamicsModel, gather_data, sample_action


def dict_to_numpy(dictionary):
    matrix = []
    for key in dictionary:
        matrix.append(dictionary[key])
    return np.array(matrix)

def dict_to_torch(dictionary):  
    return torch.tensor(dict_to_numpy(dictionary))

def get_sdim_adim(env):
    state = env.reset()
    agents = env.agents
    
    s_dim = state['agent_0'].shape[0]

    action_dict = sample_action(env, agents)
    action_tensor = dict_to_torch(action_dict)
    a_dim = action_tensor.shape[1]
    
    print("s_dim", s_dim, ", a_dim", a_dim)

    return s_dim, a_dim


# define how to make the environment.
def env_creator(env_config):
    dynamics_env = MultiagentEnvWrapper()
    dynamics_env.model = env_config['model']
    dynamics_env.env = env_config['env']

    dynamics_env.N = env_config['N']
    dynamics_env.local_ratio = env_config['local_ratio'] 
    dynamics_env.max_cycles = env_config['max_cycles']
    dynamics_env.continuous_actions = env_config['continuous_actions']
    
    
    for i in range(dynamics_env.N):
        dynamics_env._agent_ids.add("agent_" + str(i))
    
    dynamics_env.observation_space = dict()
    dynamics_env.action_space = dict()
    for agent in dynamics_env._agent_ids:
        dynamics_env.observation_space[agent] = dynamics_env.env.observation_space(agent)
        dynamics_env.action_space[agent] = dynamics_env.env.action_space(agent)
    print("observation space", dynamics_env.observation_space)
    print("action_space", dynamics_env.action_space)
    dynamics_env.state = dict_to_torch(dynamics_env.env.reset())
    return dynamics_env

if __name__ == "__main__":
    # register that way to make the environment under an rllib name
    env_name = "simple_spread_dynamics"
    register_env(env_name, env_creator)
    # now you can use `simple_spread` as an environment
    PATH = "..\\model_checkpoints\\multiagent_dynamics\\multiagent_dynamics_02_02_2023_03_51_37_simple_spread_v2_wd_0.001_lr_0.0001_hdim_256\\multiagent_dynamics_Decay_best_loss_-8.795983451431288.pth"
    N = 3
    local_ratio = 0.5
    max_cycles = 100
    continuous_actions = True
    env = simple_spread_v2.parallel_env(N = N, 
                    local_ratio = local_ratio, 
                    max_cycles = max_cycles, 
                    continuous_actions = continuous_actions)    
    deviceID = 'cuda:0'

    ## SET UP MODEL ##    
    d_model = 64
    h_dim = 256

    s_dim, a_dim = get_sdim_adim(env)
    model = MultiAgentDynamicsModel(s_dim, a_dim, d_model, h_dim)
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
    device = torch.device(deviceID)
    model.eval()
    model = model.cuda()

    env_config = {
        'N':N, 
        'local_ratio': local_ratio, 
        'max_cycles': max_cycles, 
        'continuous_actions': continuous_actions,
        'env' : env,
        'model': model
    }

    observation_space = dict()
    action_space = dict()
    for agent in env.agents:
        observation_space[agent] = env.observation_space(agent)
        action_space[agent] = env.action_space(agent)

    print("main obs space", observation_space)
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config, clip_actions=True)
        .framework(framework="torch")
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )

    # config = (
    #     PPOConfig()
    #     .rollouts(num_rollout_workers=4, rollout_fragment_length=100)
    #     .training(
    #         train_batch_size=512,
    #         lr=1e-4,
    #         gamma=0.99,
    #         lambda_=0.9,
    #         use_gae=True,
    #         clip_param=0.4,
    #         grad_clip=None,
    #         entropy_coeff=0.1,
    #         vf_loss_coeff=0.25,
    #         sgd_minibatch_size=64,
    #         num_sgd_iter=10,
    #     )
    #     .environment(env=env_name, clip_actions=True)
    #     .debugging(log_level="ERROR")
    #     .framework(framework="torch")
    #     .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    # )

    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 5000000},
    #     checkpoint_freq=10,
    #     local_dir="~/ray_results/" + env_name,
    #     config=config.to_dict(),
    # )

    tune.run( 
    "PPO",
    stop={"training_iteration": 5000},
    config={
        "env": env_name,
        "env_config": env_config,
        "num_gpus":1,
        "multiagent": {
            "policies": {
                "agent_0": (None, env.observation_space("agent_0"), env.action_space("agent_0"), {}),
                "agent_1": (None, env.observation_space("agent_1"), env.action_space("agent_1"), {}),
                "agent_2": (None, env.observation_space("agent_2"), env.action_space("agent_2"), {})
                # "agent_0": (None, None, None, {}),
                # "agent_1": (None, None, None, {}),
                # "agent_2": (None, None, None, {}),
                # "random": (RandomPolicy, obs_space, act_space, {}),
            },
            "policy_mapping_fn": (
                lambda agent_id: agent_id)
        },
        "framework":"torch"
    },
)
