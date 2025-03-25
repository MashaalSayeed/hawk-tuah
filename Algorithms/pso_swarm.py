import numpy as np
import random

from drone import Drone

# PSO CONFIGURATION
W = 1
C1 = 0.75
C2 = 0.75


class PSOSwarm:
    def __init__(self, drones: list[Drone]):
        self.drones = drones
        self.best_score_idx = 0
        self.best_position = self.drones[self.best_score_idx].position

    def run(self, fire_grid, target_positions):
        target_positions = np.array(list(target_positions))
        if len(target_positions) > 0:
            self.best_position = target_positions[0]

        for drone in self.drones:
            if len(target_positions) == 0:
                drone.velocity = np.zeros(2)
                continue

            r1, r2 = random.random(), random.random()
            drone.velocity = (
                W * drone.velocity + 
                C1 * r1 * (drone.best_position - drone.position) + 
                C2 * r2 * (self.best_position - drone.position)
            )

            pos = int(drone.position[0]), int(drone.position[1])
            radius = 2
            fire_intensity = np.sum(fire_grid.grid[pos[0]-radius:pos[0]+radius, pos[1]-radius:pos[1]+radius])
            distance = np.linalg.norm(drone.position - target_positions, axis=1)
            score = np.min(distance) / (fire_intensity + 1e-6)
            if score < drone.best_score:
                drone.best_score = score
                drone.best_position = drone.position

            if distance.min() < 1:
                # Extinguish fire
                # drone.velocity = np.zeros(2)
                drone.best_position = drone.position
                drone.best_score = np.inf

        best_scores = np.array([drone.best_score for drone in self.drones])
        self.best_score_idx = np.argmin(best_scores)
        self.best_position = self.drones[self.best_score_idx].position
        return True
    

class APFSwarm:
    def __init__(self, drones: list[Drone], ka=10.0, kr=1.0, d0=5.0):
        self.drones = drones
        self.ka = ka  # Attractive force coefficient
        self.kr = kr  # Repulsive force coefficient
        self.d0 = d0  # Threshold distance for repulsive force

    def run(self, fire_grid, target_positions):
        target_positions = np.array(list(target_positions))

        for drone in self.drones:
            if len(target_positions) == 0:
                drone.velocity = np.zeros(2)
                continue

            # Calculate Attractive Force
            nearest_target = min(target_positions, key=lambda t: np.linalg.norm(drone.position - t))
            attractive_force = self.calculate_attractive_force(drone, nearest_target)

            # Calculate Repulsive Force from obstacles (fire zones) and drones
            repulsive_force = self.calculate_repulsive_force(drone, fire_grid)

            # Resultant force is the sum of attractive and repulsive forces
            resultant_force = attractive_force + repulsive_force

            # Update drone velocity and position based on resultant force
            drone.velocity = resultant_force
            # Stop drone if near target or extinguish fire at the target
            distance = np.linalg.norm(drone.position - target_positions, axis=1)
            if distance.min() < 1:
                # Fire extinguished - stop drone near the target
                drone.velocity = np.zeros(2)

        return True

    def calculate_attractive_force(self, drone: Drone, target_position):
        """ Calculate attractive force pulling drone toward the nearest target. """
        direction = target_position - drone.position
        magnitude = self.ka * np.linalg.norm(direction)
        return (direction / np.linalg.norm(direction)) * magnitude if np.linalg.norm(direction) > 0 else np.zeros(2)

    def calculate_repulsive_force(self, drone: Drone, fire_grid):
        """ Calculate repulsive force from fire zones and nearby drones. """
        repulsive_force = np.zeros(2)

        # Repulsive force from obstacles (e.g., fire)
        pos_x, pos_y = int(drone.position[0]), int(drone.position[1])
        radius = 2
        if 0 <= pos_x < fire_grid.grid.shape[0] and 0 <= pos_y < fire_grid.grid.shape[1]:
            fire_intensity = np.sum(fire_grid.grid[pos_x - radius:pos_x + radius, pos_y - radius:pos_y + radius])
            if fire_intensity > 0:  # Strong repulsive force if near fire zone
                repulsive_force += self.kr * (1 / (fire_intensity + 1e-6))

        # Repulsion from other drones
        for other_drone in self.drones:
            if other_drone != drone:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < self.d0:
                    repulsive_force += self.kr * (1 / distance - 1 / self.d0) * (drone.position - other_drone.position)

        return repulsive_force


# MO_W = 0.9
# MO_C1 = 1.5
# MO_C2 = 3

# class MO_PSOSwarm:
#     def __init__(self, drones: list[Drone], archive_size=50):
#         self.drones = drones
#         self.archive = []  # Pareto front archive of non-dominated solutions
#         self.archive_size = archive_size
#         self.best_position = None

#     def run(self, fire_grid, target_positions):
#         target_positions = np.array(list(target_positions))
#         if len(self.archive) > 0:
#             self.best_position = random.choice([sol[0] for sol in self.archive])  # Random leader from Pareto front

#         for drone in self.drones:
#             if len(target_positions) == 0:
#                 drone.velocity = np.zeros(2)
#                 continue

#             # Velocity update based on MOPSO strategy
#             r1, r2 = random.random(), random.random()
#             leader_position = self.best_position if self.best_position is not None else drone.position

#             drone.velocity = (
#                 MO_W * drone.velocity +
#                 MO_C1 * r1 * (drone.best_position - drone.position) +
#                 MO_C2 * r2 * (leader_position - drone.position)
#             )

#             # Evaluate multi-objective fitness
#             # fire_intensity = self.evaluate_fire_intensity(drone.position, fire_grid)
#             distance_score = self.evaluate_distance(drone.position, target_positions)
#             dispersion_score = self.evaluate_dispersion(drone.position)

#             # Multi-objective fitness (fire suppression and dispersion)
#             scores = [distance_score, -dispersion_score]  # Negate dispersion to maximize it
#             self.update_pareto_archive(drone.position, scores)

#             # Update personal best for drone if dominated by new scores
#             if self.dominates(scores, drone.best_scores):
#                 drone.best_position = drone.position
#                 drone.best_scores = scores

#             # If a drone reaches a fire, it stops and updates the fire grid
#             if distance_score < 1:  # Close enough to extinguish fire
#                 drone.velocity = np.zeros(2)
#                 drone.best_scores = [float('inf'), float('inf')]

#         return True

#     def evaluate_fire_intensity(self, position, fire_grid, radius=2):
#         pos = int(position[0]), int(position[1])
#         return np.sum(fire_grid.grid[pos[0]-radius:pos[0]+radius, pos[1]-radius:pos[1]+radius])

#     def evaluate_distance(self, position, target_positions):
#         if len(target_positions) == 0:
#             return float('inf')
#         return np.min(np.linalg.norm(target_positions - position, axis=1))

#     def evaluate_dispersion(self, position):
#         swarm_positions = np.array([drone.position for drone in self.drones])
#         distances = np.linalg.norm(swarm_positions - position, axis=1)
#         distances = distances[distances > 0]  # Avoid zero distance to self
#         return np.min(distances) if len(distances) > 0 else 0

#     def dominates(self, a, b):
#         """ Check if solution `a` dominates solution `b`. """
#         return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

#     def update_pareto_archive(self, position, scores):
#         """ Update the Pareto front archive with the given position and scores. """
#         # Add new solution and remove dominated solutions
#         self.archive.append((position, scores))
#         # Filter Pareto archive correctly using a list comprehension
#         # self.archive = [p for p in self.archive if not (self.dominates(p[1], q[1]) for q in self.archive if p != q)]
#         self.archive = [p for p in self.archive if not any(self.dominates(p[1], q[1]) for q in self.archive if not np.array_equal(p, q))]

#         # If archive exceeds size limit, use crowding distance to prune solutions
#         if len(self.archive) > self.archive_size:
#             self.archive = self.prune_archive(self.archive)

#     def prune_archive(self, archive):
#         """ Prune Pareto archive using crowding distance. """
#         archive_scores = np.array([sol[1] for sol in self.archive])
#         crowding_distances = np.zeros(len(self.archive))

#         # Sort by each objective and calculate crowding distances
#         for i in range(archive_scores.shape[1]):  # Iterate over objectives
#             sorted_indices = np.argsort(archive_scores[:, i])
#             crowding_distances[sorted_indices[0]] = float('inf')  # Boundaries get max crowding distance
#             crowding_distances[sorted_indices[-1]] = float('inf')

#             for j in range(1, len(self.archive) - 1):
#                 prev_score = archive_scores[sorted_indices[j - 1], i]
#                 next_score = archive_scores[sorted_indices[j + 1], i]
#                 crowding_distances[sorted_indices[j]] += (next_score - prev_score)

#         # Sort solutions by crowding distance (descending) and keep the top archive_size
#         sorted_indices = np.argsort(-crowding_distances)
#         return [self.archive[i] for i in sorted_indices[:self.archive_size]]