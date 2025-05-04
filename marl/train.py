import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import PettingZooEnv
import supersuit as ss
from pettingzoo.utils import parallel_to_aec

from env import FireFightingEnv
import os

# Initialize Ray
ray.init()

# Wrap Environment for RLlib
def env_creator(_):
    env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5)
    env = parallel_to_aec(env)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.multiagent_wrappers.pad_action_space_v0(env)

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
        policies={agent_id: PolicySpec() for agent_id in agent_ids},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
    )
)

# Train MAPPO
results_path = os.path.join(os.path.dirname(__file__), "ray_results")
tune.run(
    "PPO",
    name="Mad",
    config=config.to_dict(),
    stop={"training_iteration": 200},
    storage_path=results_path,
    checkpoint_at_end=True
)

ray.shutdown()
