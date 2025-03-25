import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_LEADERS = 3
NUM_FOLLOWERS = 10
TARGET = np.array([100, 80])
SPEED = 0.5
SEPARATION_DISTANCE = 2.0
DRONE_RADIUS = 20

ATTRACTION_STRENGTH = 0.05
SEPARATION_STRENGTH = 0.1

class Drone:
    def __init__(self, position, role):
        self.position = position
        self.velocity = np.zeros(2)
        self.role = role
    
    def update_velocity(self, neighbours, target):
        separation_force = np.zeros(2)
        attraction_force = np.zeros(2)

        for neighbour in neighbours:
            diff = self.position - neighbour.position
            dist = np.linalg.norm(diff)
            if dist < SEPARATION_DISTANCE and dist > 0:
                separation_force += SEPARATION_STRENGTH * (diff / dist) / dist

        diff = target - self.position
        dist = np.linalg.norm(diff)
        if dist > 0:
            attraction_force += ATTRACTION_STRENGTH * diff / dist

        self.velocity = separation_force + attraction_force
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * SPEED

    def update_position(self):
        self.position += self.velocity

leaders = [Drone(np.random.rand(2) * 20, 'leader') for _ in range(NUM_LEADERS)]
followers = [Drone(np.random.rand(2) * 20, 'follower') for _ in range(NUM_FOLLOWERS)]

def update(frame):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for leader in leaders:
        leader.update_velocity(leaders, TARGET)
        leader.update_position()
        ax.plot(*leader.position, 'ro', label='Leader' if frame == 0 else "")

    # Update followers
    for follower in followers:
        # Find nearest leader
        leader_positions = np.array([leader.position for leader in leaders])
        distances = np.linalg.norm(leader_positions - follower.position, axis=1)
        target_leader = leader_positions[np.argmin(distances)]
        
        follower.update_velocity(followers + leaders, target_leader)
        follower.update_position()
        ax.plot(*follower.position, 'bo', label='Follower' if frame == 0 else "")

    ax.plot(*TARGET, 'gx', markersize=10, label='Target')
    ax.legend()
    ax.set_title('Hierarchical Leader-Follower Drone Swarm')

fig, ax = plt.subplots(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()
