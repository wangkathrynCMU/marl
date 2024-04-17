# need to look into how to vectorize custom environments
import os
import wandb
import ray
from ray.rllib.algorithms import ppo
from pettingzoo.mpe import simple_spread_v3
import torch
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ma_state_dynamics_model import MAStateModel
from reward_model_transformer import RewardDynamicsTransformer
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.air.integrations.wandb import WandbLoggerCallback

class CustomMultiagentEnv(MultiAgentEnv):
    def __init__(self, env_config):
        print("INITTTT", torch.cuda.is_available())
        # Initialize PettingZoo Environment
        if env_config['pz_env_name'] == "simple_spread_v3":
            pz_env_config = env_config['pz_env_config'] #petting zoo environment config
            env = simple_spread_v3.parallel_env(N = pz_env_config['N'], 
                        local_ratio = pz_env_config['local_ratio'], 
                        max_cycles = pz_env_config['max_cycles'], 
                        continuous_actions = pz_env_config['continuous_actions'])
            self.env = env
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.curr_state, self.info = self.env.reset()
        self.truncated = {}
        self.info = {}
        self.agents = self.curr_state.keys()
        self.terminated = {agentID:False for agentID in self.agents}
        print("agents", self.agents)
        self.max_cycles = pz_env_config['max_cycles']
        self.num_cycles = 0

        STATE_PATH = env_config['STATE_PATH']
        REWARD_PATH = env_config['REWARD_PATH']

        s_dim, a_dim = self.get_sdim_adim(env)
        
        # INITIALIZE STATE MODEL
        s_d_model = 256
        s_h_dim = 256
        
        state_model = MAStateModel(s_dim, a_dim, s_d_model, s_h_dim)
        state_model.load_state_dict(torch.load(STATE_PATH)['model_state_dict'], map_location=torch.device('cpu'))
        self.state_model = state_model.cuda()

        # INITIALIZE REWARD MODEL
        r_d_model = 256
        r_dim_feedforward = 512
        r_nhead = 8
        r_num_layers = 1

        reward_model = RewardDynamicsTransformer(s_dim, a_dim, r_d_model, r_dim_feedforward, r_nhead, r_num_layers)
        reward_model.load_state_dict(torch.load(REWARD_PATH)['model_state_dict'], map_location=torch.device('cpu'))
        self.reward_model = reward_model.cuda()

        self.curr_state, self.info = self.env.reset()
        self.terminated = {agentID:False for agentID in self.agents}
        self.truncated = {}
        self.info = {}
    
    def get_sdim_adim(self, env):
        state, info = env.reset()
        s_dim = state['agent_0'].shape[0]

        action_space = env.action_space('agent_0')
        a_dim = action_space.shape[0]

        return s_dim, a_dim

    def reset(self, seed, options):
        self.curr_state, self.info = self.env.reset()
        self.terminated = {agentID:False for agentID in self.agents}
        self.truncated = {}
        self.info = {}
        return self.curr_state, self.info
        
    def step(self, action):
        if all(self.terminated.values()):
            return self.curr_state, self.reward, self.terminated, self.truncated, self.info
        
        next_state = self.state_model.sample_next_dict(self.curr_state, self.action)
        reward = self.reward_model.sample_next_dict(self.curr_state, self.action)
        self.curr_state = next_state
        self.reward = reward

        self.num_cycles +=1

        if self.num_cycles >= self.max_cycles:
            self.terminated = {agentID:True for agent in agents}
        return self.curr_state, self.reward, self.terminated, self.truncated, self.info
    
    def dict_to_numpy(self, dictionary):
        matrix = []
        for key in dictionary:
            matrix.append(dictionary[key])
        return np.array(matrix)

    def dict_to_torch(self, dictionary):
        return torch.tensor(dict_to_numpy(dictionary))

    def torch_to_dict(self, tensor, agents):
        output = dict()
        index = 0

        for agent in agents:
            output[agent] = tensor[index]
            index +=1
        
        return output

def env_creator(env_config):
    return CustomMultiagentEnv(env_config)


if __name__ == '__main__':
    wandb.init(project="rllib-MAPPO-customenv_simplespread", entity="biorobotics-marl")
    
    print("HEREEEE!")
    ray.init()

    print("HEREEEE!!")
    STATE_PATH = "model_checkpoints\\ma_state_dynamics\\ma_state_dynamics_10_31_2023_10_38_13_simple_spread_v3\\ma_state_dynamics_Decay_best_loss_-8.208150715827943.pth"
    REWARD_PATH = "model_checkpoints\\reward_dynamics\\reward_dynamics_transformer_10_31_2023_10_39_02_simple_spread_v3\\reward_dynamics_transformer_Decay_best_loss_-2.2542934103572403.pth"
    
    pz_env_config = {
        'N':3, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }

    pz_env_name = "simple_spread_v3"

    env_config = {}  
    env_config['STATE_PATH'] = STATE_PATH
    env_config['REWARD_PATH'] = REWARD_PATH
    env_config['pz_env_config'] = pz_env_config
    env_config['pz_env_name'] = pz_env_name

    custom_env_name = "custom_simple_spread"
    register_env(custom_env_name, lambda config: env_creator(config))
    # algo = ppo.PPO(env="my_env")

    print("HEREEEE!!!")
    config = (
        PPOConfig()
        .environment(env=custom_env_name, clip_actions=True, env_config = env_config)
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

    local_dir = "C:\\Users\\kkwan\\ray_results\\simple_spread_v3"
    print("HEREEEE!!!!")
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir= local_dir,
        config=config.to_dict(),
        callbacks=[WandbLoggerCallback(
        project="rllib-MAPPO-custom-simplespread",
        entity= "biorobotics-marl",
        api_key="5c8d2be2d372d7685da285d9477c9f2e90577628",
        log_config=True
        )]
    )