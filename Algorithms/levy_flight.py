import numpy as np
import matplotlib.pyplot as plt
import random

def levy_flight(beta=1.5, size=2):
    sigma = (np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    step = np.random.normal(0, sigma, size)
    return step

def bounded_levy_flight(position, beta=1.5, size=2, bounds=(99, 99)):
    """ Generates a Levy flight step that remains within bounds. """
    for _ in range(10):  # Try up to 10 times to generate a valid step
        step = levy_flight(beta, size)
        new_position = position + step
        if np.all((new_position >= 0) & (new_position < bounds)):  
            return step  
    return np.zeros(2) 

class Drone:
    def __init__(self, id, position, search_radius=5):
        self.id = id
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.search_radius = search_radius
        self.detected_fires = []
    
    def move(self, new_position):
        self.position = np.clip(new_position, 0, 99)
    
    def detect_fire(self, fires: list['Fire']):
        for fire in fires:
            if np.linalg.norm(self.position - fire.position) <= self.search_radius:
                if fire not in self.detected_fires:
                    self.detected_fires.append(fire)
                    print(f"Drone {self.id} detected fire at {fire.position}")

class Fire:
    def __init__(self, position):
        self.position = np.array(position)
    
    def spread(self):
        self.position += np.random.uniform(-1, 1, 2)
        self.position = np.clip(self.position, 0, 99)

def update_potential_map(potential_map, drones, fires):
    potential_map *= 0.99  # Decay over time
    for fire in fires:
        potential_map[int(fire.position[0]), int(fire.position[1])] = 1  # Fire location
    for drone in drones:
        potential_map[int(drone.position[0]), int(drone.position[1])] = -1  # Visited area

def main():
    NUM_DRONES = 10
    NUM_FIRES = 3
    ITERATIONS = 200
    SEARCH_SPACE = (100, 100)
    
    drones = [Drone(i, np.random.uniform(0, 99, 2)) for i in range(NUM_DRONES)]
    fires = [Fire(np.random.uniform(0, 99, 2)) for _ in range(NUM_FIRES)]
    
    potential_map = np.zeros(SEARCH_SPACE)
    
    plt.ion()
    fig, ax = plt.subplots()
    
    for iter in range(ITERATIONS):
        ax.clear()
        
        for fire in fires:
            fire.spread()
        
        for drone in drones:
            drone.detect_fire(fires)
            [fires.remove(f) for f in drone.detected_fires if f in fires]
            
            # if drone.detected_fires:
            #     attraction = np.sum([(fire.position - drone.position) for fire in drone.detected_fires], axis=0)
            #     drone.velocity = attraction / np.linalg.norm(attraction) if np.linalg.norm(attraction) > 0 else np.zeros(2)
            # else:
            while True:
                velocity = bounded_levy_flight(drone.position)
                new_position = drone.position + velocity
                # new_position = np.clip(new_position, 0, 99)
                if potential_map[int(new_position[0]), int(new_position[1])] >= -0.25:
                    drone.velocity = velocity
                    break
            
            drone.move(drone.position + drone.velocity)
        
        update_potential_map(potential_map, drones, fires)
        
        ax.imshow(potential_map.T, origin='lower', cmap='hot', alpha=0.5)
        ax.scatter([fire.position[0] for fire in fires], [fire.position[1] for fire in fires], color='red', marker='*', s=200, label='Fire')
        ax.scatter([drone.position[0] for drone in drones], [drone.position[1] for drone in drones], color='blue', label='Drones')
        
        ax.set_title(f"Iteration {iter+1}/{ITERATIONS}")
        ax.legend()
        plt.draw()
        plt.pause(0.1)
        
        all_detected = all(any(tuple(fire.position) in drone.detected_fires for drone in drones) for fire in fires)
        if all_detected:
            print("All fires detected!")
            break
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
