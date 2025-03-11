import numpy as np
import matplotlib.pyplot as plt
import random

# PSO Parameters
NUM_DRONES = 20     # Number of drones (particles)
DIMENSIONS = 2      # 2D space
ITERATIONS = 500    # Max number of iterations
NUM_FIRES = 10       # Number of fire sources
FIRE_SPREAD_RATE = 0.5  # Rate at which fire spreads outward per iteration

# Initialize fire locations randomly
fires = np.random.uniform(10, 90, (NUM_FIRES, DIMENSIONS))
print(fires)

# PSO Hyperparameters
W = 1    # Inertia weight
C1 = 0.75   # Cognitive (personal best) coefficient
C2 = 0.75   # Social (global best) coefficient

# Initialize drone positions and velocities
positions = np.random.uniform(0, 30, (NUM_DRONES, DIMENSIONS))  # Random start positions
velocities = np.random.uniform(-1, 1, (NUM_DRONES, DIMENSIONS))  # Random initial velocities
p_best_positions = np.copy(positions)  # Personal best positions
p_best_scores = np.min(np.linalg.norm(p_best_positions[:, None] - fires, axis=2), axis=1)  # Distance to nearest fire

# Global best position (initialized with the best p_best found)
g_best_idx = np.argmin(p_best_scores)
g_best_position = p_best_positions[g_best_idx]

# Visualization setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

for iter in range(ITERATIONS):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Fire spread simulation
    fires += np.random.uniform(-FIRE_SPREAD_RATE, FIRE_SPREAD_RATE, fires.shape)
    fires = np.clip(fires, 0, 100)  # Keep fires within bounds

    fire_found = False

    # Update drone positions and velocities
    for i in range(NUM_DRONES):
        r1, r2 = random.random(), random.random()
        
        # Update velocity
        velocities[i] = (
            W * velocities[i] + 
            C1 * r1 * (p_best_positions[i] - positions[i]) + 
            C2 * r2 * (g_best_position - positions[i])
        )

        velocities[i] = np.clip(velocities[i], -1, 1)
        # Update position
        positions[i] += velocities[i]

        # Boundary conditions
        positions[i] = np.clip(positions[i], 0, 99)

        # Evaluate new position (distance to the closest fire)
        score = np.min(np.linalg.norm(positions[i] - fires, axis=1))

        # Update personal best
        if score < p_best_scores[i]:
            p_best_scores[i] = score
            p_best_positions[i] = positions[i]

        # Remove fire if drone is in the fire spot
        if score < 1.0:
            print(f"Fire found by drone {i} at iteration {iter + 1}!")
            fires = np.delete(fires, np.argmin(np.linalg.norm(positions[i] - fires, axis=1)), axis=0)
            fire_found = True

            if len(fires) == 0:
                print("All fires extinguished!")
                break

    if len(fires) == 0:
        break

    if fire_found:
        # Recalculate best scores
        p_best_positions = np.copy(positions)
        p_best_scores = np.min(np.linalg.norm(p_best_positions[:, None] - fires, axis=2), axis=1)
        fire_found = False

    # Update global best
    g_best_idx = np.argmin(p_best_scores)
    g_best_position = p_best_positions[g_best_idx]

    # Visualization
    ax.scatter(positions[:, 0], positions[:, 1], color='blue', label="Drones")
    ax.scatter(fires[:, 0], fires[:, 1], color='red', marker='*', s=200, label="Fires (Targets)")
    ax.scatter(g_best_position[0], g_best_position[1], color='green', marker='P', s=100, label="Global Best")
    
    ax.set_title(f"Iteration {iter+1}/{ITERATIONS}")
    ax.legend()
    plt.draw()
    plt.pause(0.1)

    # # Stop if a drone reaches any fire
    # if np.min(p_best_scores) < 1.0:
    #     print(f"Fire found by a drone at iteration {iter+1}!")
    #     break

plt.ioff()
plt.show()
