from pettingzoo.utils.env import ParallelEnv
from pettingzoo.test import parallel_api_test
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

"""
Multi-Agent Reinforcement Learning Environment for Firefighting
- The environment consists of multiple drones (scouts and suppressor) that need to extinguish a fire.
- The drones have to navigate to the fire location and extinguish it.
- The environment is partially observable, and the drones have to cooperate to extinguish the fire effectively.

The environment is inspired by the Firefighting Multi-Agent Reinforcement Learning (MARL) problem.

==========================================
    The Firefighting MARL Environment
==========================================

Scout: The scout drone is responsible for exploring the environment and finding the fire location.
Suppressor: The suppressor drone is responsible for extinguishing the fire once the fire location is found.
The environment consists of the following components:

Fire Grid:
- The fire grid is a 2D grid representing the intensity of the fire at each location.
- The fire intensity ranges from 0 to 1, where 0 represents no fire and 1 represents maximum fire intensity.
- The fire grid is partially observable to the drones, and they have to explore the environment to find the fire location.

Scout Drone:
- The scout drone has a sensing radius of 10 units.
- The scout drone has a speed of 1 unit per step.
- Observation Space: (x, y) position of the scout drone + battery level (normalized to [0, 1]) + fire intensity grid within the sensing radius.
- Action Space: Discrete(9) representing the actions: {8 Directions + Stay}.

Suppressor Drone:
- The suppressor drone has a speed of 1 unit per step.
- Observation Space: (x, y) position of the suppressor drone + battery level (normalized to [0, 1]) + fire intensity grid at the current location.
- Action Space: Discrete(10) representing the actions: {8 Directions + Stay + Extinguish}.
- The suppressor drone can extinguish the fire at its current location by taking the Extinguish action.

==================================
    Comparison Metrics
==================================

- Fire Extinguished %:      	Percentage of total fire cells extinguished by the end of an episode.
- Response Time: 	            Time taken to detect and start suppressing a fire after it starts.
- Average Extinguishing Time: 	Average time taken to extinguish a hotspot from detection.
- Total Time/Episode Length: 	Total time taken to extinguish all fires (or reach terminal condition).
- Energy/Battery Usage: 	    Total energy consumed across agents (battery-based cost).
- Number of Collisions: 	    For systems where agents can collide, track safety/efficiency.
- Idle Time: 	                Time agents spent doing nothing (e.g., no detection, no suppression).

==================================
"""

from drone import Drone, FireGrid
from metrics import MetricsLogger


class FireFightingEnvSimple(ParallelEnv):
    metadata = {
        "name": "FireFightingEnv"
    }

    def __init__(self, grid_size=(100, 100), num_drones=5, render_mode=None, initial_fire_count=3):
        self.num_drones = num_drones
        self.render_mode = render_mode

        self.fire_grid = FireGrid(grid_size, fire_count=initial_fire_count, fire_range=15)
        self.drones = [Drone(i, "drone", grid_size) for i in range(num_drones)]
        self.metrics_logger = MetricsLogger()

        self.agent_map = {drone.id: drone for drone in self.drones}
        self.agents = list(self.agent_map.keys())
        self.possible_agents = self.agents.copy()

        # Define the observation and action spaces for each agent
        self.observation_spaces = {
            drone.id: drone.observation_space() for drone in self.drones
        }

        self.action_spaces = {
            drone.id: spaces.Discrete(10) for drone in self.drones
        }

        if self.render_mode != None:
            self.fig, self.ax = None, None
        
        # Store episode step counter
        self.episode_step = 0
        self.max_steps = 1000  # Configure this based on your needs

    def _get_obs(self, agent=None):
        obs = {}
        for agent_id in self.agents:
            drone = self.agent_map[agent_id]
            x, y = drone.position
            fire_intensity = drone.sense_fire(self.fire_grid)

            # Normalize position
            x = x / self.fire_grid.grid.shape[0]
            y = y / self.fire_grid.grid.shape[1]

            obs[agent_id] = np.concatenate((
                np.array([x, y, drone.battery_level], dtype=np.float32),
                fire_intensity.flatten().astype(np.float32)
            ))

        return obs
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_global_state(self):
        """
        Returns the global state for centralized critic training in CTDE.
        This includes the entire fire grid state and all drone states.
        """
        fire_state = self.fire_grid.grid.flatten()
        drone_states = []
        for drone in self.drones:
            x, y = drone.position

            x_norm = x / self.fire_grid.grid.shape[0]
            y_norm = y / self.fire_grid.grid.shape[1]
            drone_states.extend([x_norm, y_norm, drone.battery_level])
            drone_states.append(float(drone.fire_extinguishing))
        
        return np.concatenate([fire_state, np.array(drone_states)]).astype(np.float32)
    
    def get_state_size(self):
        # Calculate based on fire grid size and drone state size
        fire_grid_size = self.fire_grid.grid.size  # Total cells in fire grid
        drone_state_size = 4 * self.num_drones  # x, y, battery, extinguishing state per drone
        return fire_grid_size + drone_state_size

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fire_grid.reset()
        for drone in self.agent_map.values():
            drone.reset()

        self.metrics_logger.reset()
        self.metrics_logger.log_initial_battery(self.agent_map.values())
        
        # Reset episode step counter
        self.episode_step = 0

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        # Add global state to info dictionary for CTDE
        # state = self.get_global_state()
        # for agent in self.agents:
        #     infos[agent]['state'] = state

        return obs, infos

    def set_reward(self):
        # Initialize base reward
        rewards = {agent: -0.05 for agent in self.agents}

        fire_locations = np.argwhere(self.fire_grid.grid > 0)  # locations of active fire
        for agent_id in self.agents:
            drone = self.agent_map[agent_id]

            # Penalty for dead battery
            if drone.battery_level <= 0:
                rewards[agent_id] -= 1.0

            # Bonus for suppressing fire
            local_fire_intensity = drone.sense_fire(self.fire_grid)
            fires_found = np.argwhere(local_fire_intensity > 0)
            x, y = int(drone.position[0]), int(drone.position[1])
            fire_in_radius = np.any(self.fire_grid.grid[max(0, x-1):min(self.fire_grid.grid.shape[0], x+1),
                                                        max(0, y-1):min(self.fire_grid.grid.shape[1], y+1)] > 0)
            if fire_in_radius or drone.fire_extinguishing:
                rewards[agent_id] += 2.0
            elif drone.fire_extinguishing:
                rewards[agent_id] += -0.1

            rewards[agent_id] += 0.05 * len(fires_found)

            # Negative reward for being too close to other agents
            for other_agent_id, other_drone in self.agent_map.items():
                if agent_id != other_agent_id:
                    distance = np.linalg.norm(np.array(drone.position) - np.array(other_drone.position))
                    if distance < 5:
                        rewards[agent_id] -= (5 - distance) * 0.1

        return rewards
    
    def get_team_reward(self, individual_rewards):
        """
        Calculate a shared team reward for cooperative learning
        """
        # Simple average of individual rewards
        if not individual_rewards:
            return 0
        team_reward = sum(individual_rewards.values()) / len(individual_rewards)
        
        # Add team-level bonuses
        fire_coverage = self._calculate_fire_coverage()
        team_reward += 0.1 * fire_coverage
        
        # Bonus for complete fire extinguishing
        if np.sum(self.fire_grid.grid > 0) == 0:
            team_reward += 10.0
            
        return team_reward
    
    def _calculate_fire_coverage(self):
        """Helper to calculate how much of the fire is being monitored"""
        fire_cells = set()
        monitored_cells = set()
        
        # Find all fire cells
        fire_locations = np.argwhere(self.fire_grid.grid > 0)
        for x, y in fire_locations:
            fire_cells.add((x, y))
        
        # Check which fire cells are being monitored by drones
        for drone in self.drones:
            x, y = drone.position
            # Assuming drones can monitor in a radius
            radius = 5  # Example sensing radius
            for i in range(max(0, x-radius), min(self.fire_grid.grid.shape[0], x+radius+1)):
                for j in range(max(0, y-radius), min(self.fire_grid.grid.shape[1], y+radius+1)):
                    if (i, j) in fire_cells:
                        monitored_cells.add((i, j))
        
        # Calculate percentage of fire being monitored
        if not fire_cells:
            return 1.0  # No fire, perfect coverage
        return len(monitored_cells) / len(fire_cells)

    def step(self, actions):
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Increment episode step counter
        self.episode_step += 1
        
        fire_cells = np.sum(self.fire_grid.grid > 0)

        # Step each agent
        rewards = self.set_reward()

        for agent_id, action in actions.items():
            drone = self.agent_map[agent_id]
            drone.step(action)

        for drone in self.drones:
            x, y = int(drone.position[0]), int(drone.position[1])

            local_fire_grid = drone.sense_fire(self.fire_grid)
            hotspot_indices = np.where(local_fire_grid > 0)

            offset_x = max(0, x - drone.sensing_radius)
            offset_y = max(0, y - drone.sensing_radius)
                
            # Convert local hotspot coordinates to global coordinates
            visible_hotspots = np.array([
                [hotspot_indices[0][i] + offset_x, hotspot_indices[1][i] + offset_y]
                for i in range(len(hotspot_indices[0]))
            ])

            self.fire_grid.detect_fires(visible_hotspots)

        suppressing = 0
        for drone in self.drones:
            x, y = int(drone.position[0]), int(drone.position[1])
            fire_in_radius = np.any(self.fire_grid.grid[max(0, x-1):min(self.fire_grid.grid.shape[0], x+1),
                                                        max(0, y-1):min(self.fire_grid.grid.shape[1], y+1)] > 0)
            
            if drone.fire_extinguishing or fire_in_radius:
                drone.suppress_fire(self.fire_grid)
                suppressing += 1
            

        # Check for collisions
        positions = {(drone.position[0], drone.position[1]) for drone in self.agent_map.values()}
        if len(positions) < len(self.agent_map):
            self.metrics_logger.log_collision()

        extinguished_cells = fire_cells - np.sum(self.fire_grid.grid > 0)

        self.metrics_logger.episode_length += 1
        self.metrics_logger.log_idle_time(self.agent_map.values())
        self.metrics_logger.log_extinguished_fire(extinguished_cells)
        self.metrics_logger.log_extinguishing_time(suppressing)

        # Check termination conditions
        env_done = np.argwhere(self.fire_grid.grid > 0).size == 0
        max_steps_reached = self.episode_step >= self.max_steps
        
        for agent_id, drone in self.agent_map.items():
            terminations[agent_id] = drone.battery_level <= 0 or env_done
            truncations[agent_id] = max_steps_reached
            infos[agent_id] = {}

        if env_done or all(terminations.values()) or max_steps_reached:
            self.metrics_logger.log_final_battery(self.agent_map.values())
            self.metrics_logger.log_fire_detection_time(self.fire_grid)
            self.metrics_logger.total_fire_cells = self.fire_grid.total_fire_count

        # Update fire grid
        self.fire_grid.spread_fire()

        # Update rewards, terminations, truncations, and infos for active agents
        obs = self._get_obs()
        terminations = {agent: terminations[agent] for agent in self.agents}
        truncations = {agent: truncations[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        
        # Calculate team reward for CTDE
        team_reward = self.get_team_reward(rewards)
        
        # Add global state and available actions to info
        state = self.get_global_state()
        
        for agent in self.agents:
            infos[agent]['state'] = state  # Global state for centralized critic
            infos[agent]['team_reward'] = team_reward  # Team reward for CTDE
            infos[agent]['individual_reward'] = rewards[agent]  # Store individual reward
            
            # Optionally replace individual rewards with team reward for full cooperation
            # rewards[agent] = team_reward

        # Remove terminated agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        return obs, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if self.render_mode is None:
            raise ValueError("Render mode is not set. Please set render_mode to 'human'.")
        
        if self.render_mode == "human":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots()

            self.ax.clear()
        else:
            self.fig, self.ax = plt.subplots()

        self.ax.imshow(self.fire_grid.grid.T, cmap="hot", interpolation="nearest")
        drone_positions = [drone.position for drone in self.drones]

        if drone_positions:
            scout_x, scout_y = zip(*drone_positions)
            self.ax.scatter(scout_x, scout_y, color="blue", label="Scout", s=20)

        self.ax.legend()
        self.ax.set_xlim(0, self.fire_grid.grid.shape[0])
        self.ax.set_ylim(0, self.fire_grid.grid.shape[1])
        self.ax.set_title("Firefighting Environment")

        if self.render_mode == "human":
            plt.pause(0.01)
            plt.draw()
        else:
            plt.show(block=False)
            plt.close(self.fig)

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None