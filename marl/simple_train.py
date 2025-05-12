# import ray
# from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env import PettingZooEnv
# from pettingzoo.utils import parallel_to_aec
# import os
# import numpy as np
# from gym.spaces import Box

# from simple_env import FireFightingEnvSimple
# from model import CentralizedCriticModel
# from ray.rllib.models import ModelCatalog

# # Register the custom model
# # ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

# # Initialize Ray
# ray.init()

# # Create wrapper to properly expose state space to RLlib
# class RLlibFireEnv(PettingZooEnv):
#     def __init__(self, env):
#         super().__init__(env)
        
#         # Add state space for centralized critic
#         # Get sample state to determine dimensions
#         sample_state = self.env.unwrapped.get_global_state()
#         self.state_space = Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=sample_state.shape,
#             dtype=np.float32
#         )
    
#     def reset(self, seed=None, options=None):
#         obs, info = super().reset(seed=seed, options=options)
#         state = self.env.unwrapped.get_global_state()

#         for agent in self.env.agents:
#             info[agent] = {"state": state}
            
#         return obs, info
    
#     def step(self, action):
#         obs, rewards, terminations, truncations, infos = super().step(action)
#         state = self.env.unwrapped.get_global_state()
            
#         for agent_id in self.env.agents:
#             if agent_id in infos:  # Only for active agents
#                 infos[agent_id] = { "state": state }
                
#         return obs, rewards, terminations, truncations, infos

# # Wrap and register environment
# def env_creator(_):
#     env = FireFightingEnvSimple(grid_size=(80, 80), num_drones=1)
#     env = parallel_to_aec(env)
#     # return RLlibFireEnv(env)
#     return env

# tune.register_env("FireFightingEnv", env_creator)

# # Instantiate temp env to get agent IDs and state space
# temp_env = env_creator({})
# agent_ids = temp_env.possible_agents
# # state_space = temp_env.state_space
# temp_env.close()

# # Define shared policy
# def policy_mapping_fn(agent_id, episode, **kwargs):
#     return "shared_policy"

# policies = {
#     "shared_policy": (
#         None,  # Use default policy class (PPO)
#         temp_env.observation_space,
#         temp_env.action_space,
#         {
#             # "model": {
#             #     "custom_model": "cc_model",
#             #     "custom_model_config": {
#             #         "use_centralized_critic": True,
#             #         "use_obs_before_centralizing": True,
#             #         "state_space": state_space,
#             #         "fcnet_hiddens": [256, 256],
#             #         "fcnet_activation": "relu",
#             #     },
#             # },
#             # PPO specific hyperparameters for this policy
#             "gamma": 0.99,
#             "lambda": 0.95,
#             "entropy_coeff": 0.01,
#             "vf_loss_coeff": 1.0,
#         },
#     )
# }

# # PPO config with centralized critic (MAPPO style)
# config = (
#     PPOConfig()
#     .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
#     .environment("FireFightingEnv", env_config={})
#     .framework("torch")
#     .training(
#         gamma=0.99,
#         lr=3e-4,
#         train_batch_size=4096,
#         lambda_=0.95,
#         clip_param=0.2,
#         vf_clip_param=10.0,
#         entropy_coeff=0.01,
#         vf_loss_coeff=1.0,
#         use_critic=True,
#         use_gae=True,
#     )
#     .multi_agent(
#         policies=policies,
#         policy_mapping_fn=policy_mapping_fn,
#         policies_to_train=["shared_policy"],
#     )
#     .env_runners(num_envs_per_env_runner=1)
#     .resources(num_gpus=0)  # Set to 1 if using GPU
#     .debugging(
#         log_level="INFO",
#         # Enable if needed
#         # log_sys_usage=True,
#     )
# )

# # Run training
# results_path = os.path.join(os.path.dirname(__file__), "ray_results")
# tune.run(
#     "PPO",
#     name="MAPPO_CTDE",
#     config=config.to_dict(),
#     stop={"training_iteration": 200},
#     storage_path=results_path,
#     checkpoint_at_end=True,
#     checkpoint_freq=10,
#     keep_checkpoints_num=5,
#     verbose=2
# )


# ray.shutdown()


import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import PettingZooEnv
import supersuit as ss
from pettingzoo.utils import parallel_to_aec

from simple_env import FireFightingEnvSimple
import os

# Initialize Ray
ray.init()

# Wrap Environment for RLlib
def env_creator(_):
    env = FireFightingEnvSimple(grid_size=(80, 80), num_drones=10)
    env = parallel_to_aec(env)
    return PettingZooEnv(env)

# Register Custom Environment
tune.register_env("FireFightingEnv", env_creator)

temp_env = env_creator({})
agent_ids = temp_env.possible_agents
temp_env.close()


# Define PPO Configuration for MAPPO
config = (
    PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment("FireFightingEnv", env_config={})
    .framework("torch")  # Use PyTorch
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=2048,
        model={
            "custom_model_config": {
                "use_centralized_critic": True,
                "use_obs_before_centralizing": True
            }
        }
    )
    .env_runners(
        num_envs_per_env_runner=4,
    )
    .resources(num_gpus=0)  # Set to 1 if using GPU
    .multi_agent(
        policies={
            "shared_policy": PolicySpec(),
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
)

# Train MAPPO
results_path = os.path.join(os.path.dirname(__file__), "ray_results")
tune.run(
    "PPO",
    name="MadWorld",
    config=config.to_dict(),
    stop={"training_iteration": 500},
    storage_path=results_path,
    checkpoint_at_end=True
)

ray.shutdown()
