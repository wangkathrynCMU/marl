import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dict_to_numpy(self, dictionary):
        matrix = []
        for key in dictionary:
            matrix.append(dictionary[key])
        return np.array(matrix)

def dict_to_torch(self, dictionary):
    return torch.tensor(dict_to_numpy(dictionary))

class CustomMultiagentEnv:
    def __init__(self, env_config):
        # Initialize PettingZoo Environment
        if env_config['pz_env_name'] == "simple_spread_v3":
            pz_env_config = env_config['pz_env_config'] #petting zoo environment config
            env = simple_spread_v3.parallel_env(N = pz_env_config['N'], 
                        local_ratio = pz_env_config['local_ratio'], 
                        max_cycles = pz_env_config['max_cycles'], 
                        continuous_actions = pz_env_config['continuous_actions'])
            self.env = env
        
        self.curr_state, self.info = self.env.reset()
        self.curr_state = self.dict_to_torch(self.curr_state)

        self.agents = self.curr_state.keys()
        
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
    
    def traj_costs(actions):
        costs = []
        for action in actions:
            _, reward, _, _, _ = self.step(action)
            costs.append(-reward)
    
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

def CEM(cost_fn,pop_size,frac_keep,n_iters,a_mean,a_sig=None):
    '''
    Ben's CEM Function
    INPUTS:
    cost_fn: function that takes a pop_size x T x a_dim-sized tensor of potential action sequences and returns pop_size tensor of associated costs (negative rewards)
    pop_size: how many trajs to simulate
    n_iters: number of optimization iterations to perform
    a_mean: T x a_dim tensor of initial mean of actions (won't matter if a_sig is none)
    a_sig: T x a_dim tensor of initial mean of actions, or none (in which case initial actions will be sampled uniformly in interval [-1,+1])

    OUTPUTS:
    a_mean: T x a_dim tensor of optimized action sequence
    '''
    a_shape = [pop_size] + list(a_mean.shape)

    if a_sig is None:
        # sample from enironment action space, which I'm assuming is
        actions = 2*torch.rand(a_shape,device=device) - 1
        assert torch.all(actions <= 1)
        assert torch.all(actions >= -1)
    else:
        # sample actions form gaussian
        actions = a_mean + a_sig*torch.randn(a_shape,device=device)

    for i in range(n_iters):
        # print('i: ', i)
        a_mean,a_sig,cost = CEM_update(actions,cost_fn,frac_keep)
        # print('cost: ', cost)
        # re-sample actions
        actions = a_mean + a_sig*torch.randn(a_shape,device=device)
        # clip
        actions = torch.clamp(actions,min=-1,max=1)

    return a_mean,a_sig



def CEM_update(actions,cost_fn,frac_keep):
    N = actions.shape[0]
    k = int(N*frac_keep)


    costs = cost_fn(actions)
    # print('costs:',torch.mean(costs).item())

    inds = torch.argsort(costs)

    inds_keep = inds[:k]
    actions_topk = actions[inds_keep,...]
    cost_topk = torch.mean(costs[inds_keep])

    # print('cost_topk:',torch.mean(cost_topk).item())

    a_mean = torch.mean(actions_topk,dim=0)
    a_sig = torch.std(actions_topk,dim=0)

    return a_mean,a_sig,cost_topk



def optimize_actions(env, initial_obs, batch_size, n_opt_iters, H, a_dim):
    obs_torch = torch.tensor(initial_obs, dtype=torch.float32, device=device)
    obs_tiled = torch.stack(batch_size * [obs_torch])

    cost_fn = lambda actions: env.get_cost_from_a_seq(obs_tiled, actions)
    a_mean = torch.zeros(H, a_dim, device=device)
    actions, _ = CEM(env, cost_fn, batch_size, frac_keep, n_opt_iters, a_mean)

    return actions

env = MyEnv(env_config)
initial_obs = env.reset()[0]
optimized_actions = optimize_actions(env, initial_obs, batch_size, n_opt_iters, H, a_dim)
