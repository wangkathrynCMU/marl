from ray.rllib.env.multi_agent_env import MultiAgentEnv
import torch
import numpy as np

def dict_to_numpy(dictionary):
    matrix = []
    for key in dictionary:
        matrix.append(dictionary[key])
    return np.array(matrix)

def dict_to_torch(dictionary):  
    return torch.tensor(dict_to_numpy(dictionary))

def torch_to_dict(tensor, agents):
    output = dict()
    index = 0
    for agent in agents:
        output[agent] = tensor[index]
    return output

class MultiagentEnvWrapper(MultiAgentEnv):
    def dict_to_torch(dictionary):
        return torch.tensor(dict_to_numpy(dictionary))
    
    def init(self):
        self.ep_len = 0
        self._agent_ids = set()
        self.observation_space = dict()
        self.action_space = dict()

    def fillDict(self, agents, val):
        output = dict()
        for agent in self._agent_ids:
            output[agent] = val
        return output
    
    
    def reset(self):
        self.ep_len = 0
        state_dict = self.env.reset()
        self.state = dict_to_torch(state_dict).cuda()
        return state_dict
        

    def step(self, action_dict):
        # print("STEP!!!", action_dict, self._agent_ids, self.observation_space, self.action_space)
        if action_dict == {}:
            neg_inf = -10**5
            return torch_to_dict(self.state, self._agent_ids), self.fillDict(self._agent_ids, neg_inf), self.fillDict(self._agent_ids, False), {}
        actions = dict_to_torch(action_dict).cuda()
        # print("ACTION!!", action_dict, actions)
        next_state_mean, next_state_stdev = self.model.forward(self.state, actions)
        next_state_reward = torch.normal(next_state_mean, next_state_stdev)

        state = next_state_reward[...,:-1].detach().cpu()
        self.state = state.cuda()
        reward = next_state_reward[...,-1:].detach().cpu()
        
        obs = dict()
        rew = dict()
        done = dict()
        index = 0
        doneBool = False
        if self.ep_len < self.max_cycles:
            doneBool = True

        for agent in self._agent_ids:
            obs[agent] = state[index].numpy()
            rew[agent] = reward[index].item()
            done[agent] = doneBool
            index +=1
        
        done["__all__"] = doneBool
        self.ep_len += 1
        # print("obs", obs)
        # print("rew", rew)
        # print("done", done)
        
        return obs, rew, done, {}