import os

from ray import tune
from ray.tune.registry import register_env
# import the pettingzoo environment
from pettingzoo.mpe import simple_spread_v2
# import rllib pettingzoo interface
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# define how to make the environment.
def env_creator(args):
    env = simple_spread_v2.parallel_env(N = 3, 
                    local_ratio = 0.5, 
                    max_cycles = 100, 
                    continuous_actions = True)    

    return env

if __name__ == "__main__":
    # register that way to make the environment under an rllib name
    env_name = "simple_spread"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # now you can use `simple_spread` as an environment

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
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

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
