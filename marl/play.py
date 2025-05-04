# import ray
# from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.env import PettingZooEnv
# import supersuit as ss
# from pettingzoo.utils import parallel_to_aec

# from env import FireFightingEnv  # your custom env
# from pathlib import Path
# import time

# # Initialize Ray
# ray.init(ignore_reinit_error=True)

# # === Register the custom environment ===
# def env_creator(_):
#     env = FireFightingEnv()
#     env = parallel_to_aec(env)
#     env = ss.pad_observations_v0(env)
#     env = ss.pad_action_space_v0(env)
#     return PettingZooEnv(env)

# from ray import tune
# tune.register_env("FireFightingEnv", env_creator)

# # === Load trained model ===
# checkpoint_path = Path("/Users/mash/Projects/Drone/hawk-tuah/marl/ray_results/PPO_2025-05-03_16-57-12/PPO_FireFightingEnv_8f28c_00000_0_2025-05-03_16-57-13/checkpoint_000000").expanduser().resolve()
# algo = PPO.from_checkpoint(str(checkpoint_path))

# # === Run the environment with rendering ===
# env = FireFightingEnv(render_mode="human")
# env = parallel_to_aec(env)
# env = ss.pad_observations_v0(env)
# env = ss.pad_action_space_v0(env)

# env.reset()
# obs = {agent: env.observe(agent) for agent in env.agents}
# # obs, infos = env.reset()

# done = {agent: False for agent in env.agents}


# while not all(done.values()):
#     actions = {}
#     for agent_id, agent_obs in obs.items():
#         actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=agent_id)
    
#     obs, rewards, dones, truncated, infos = env.step(actions)
#     done.update(dones)

#     # You may want to sleep to make the render smoother
#     env.render()
#     time.sleep(0.2)
# env.close()

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import PettingZooEnv
import supersuit as ss
from pettingzoo.utils import parallel_to_aec

from env import FireFightingEnv  # your custom env
from pathlib import Path
import time
import numpy as np
import random

# === Initialize Ray ===
np.random.seed(1)
random.seed(1)
ray.init(ignore_reinit_error=True)

# === Environment creator ===
def env_creator(_):
    env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5)
    env = parallel_to_aec(env)  # convert to AEC API
    env = ss.pad_observations_v0(env)  # pad obs
    env = ss.pad_action_space_v0(env)  # pad actions
    return PettingZooEnv(env)  # wrap for RLlib

# === Register with Ray Tune ===
from ray import tune
tune.register_env("FireFightingEnv", env_creator)

# === Load trained PPO checkpoint ===
log_dir = "~/ray_results" 
checkpoint_path = Path(f"/Users/mash/Projects/Drone/hawk-tuah/marl/ray_results/PPO_2025-05-03_16-57-12/PPO_FireFightingEnv_8f28c_00000_0_2025-05-03_16-57-13/checkpoint_000000").expanduser().resolve()
# checkpoint_path = Path(f"/Users/mash/ray_results/PPO_2025-04-22_23-30-17/PPO_FireFightingEnv_a634e_00000_0_2025-04-22_23-30-17/checkpoint_000000").expanduser().resolve()
algo = PPO.from_checkpoint(str(checkpoint_path))

# === Create environment ===
env = FireFightingEnv(grid_size=(80, 80), num_scouts=5, num_suppresors=5, render_mode='human')  # your custom env (parallel API)
env = parallel_to_aec(env)  # convert to AEC API
env = ss.pad_observations_v0(env)  # pad obs
env = ss.pad_action_space_v0(env)  # pad actions

# === Reset the environment ===
print(env.reset())

# === Run using AEC agent iteration loop ===
mean_reward = []
for agent in env.agent_iter():
    obs, reward, done, truncated, info = env.last()

    mean_reward.append(reward)
    if done:
        action = None  # must send None to move to next agent if done
    else:
        # action = algo.compute_single_action(obs, policy_id="default_policy")
        action = algo.compute_single_action(obs, policy_id=agent)

    env.step(action)
    env.render()
print(f"Mean reward: {sum(mean_reward) / len(mean_reward)}")
print(env.unwrapped.metrics_logger.calculate_metrics())  # print metrics
env.close()
