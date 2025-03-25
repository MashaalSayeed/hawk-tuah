from dronev import Drone, Location
import random
import time
import csv
import math


class DroneSwarmACO:
    def __init__(self, connection_strings: list[str], home_position: Location, target_position: Location, swarm_size=3, max_iterations=30, evaporation_rate=0.1):
        self.drones = [Drone(conn) for conn in connection_strings]
        self.home_position = home_position
        self.target_position = target_position
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.evaporation_rate = evaporation_rate
        self.pheromone_map = {}  # Dictionary to store pheromone levels on paths
        self.alpha = 1  # Pheromone influence
        self.beta = 2   # Heuristic influence

    def initialize_pheromone_map(self, waypoints: list[Location]):
        """Initialize pheromones for all paths."""
        for wp1 in waypoints:
            for wp2 in waypoints:
                if wp1 != wp2:
                    self.pheromone_map[(wp1, wp2)] = 1.0  # Initial pheromone level
    
    def heuristic(self, position: Location):
        """Inverse distance to target as heuristic."""
        return 1 / (0.0001 + ((position.lat - self.target_position.lat)**2 +
                              (position.lon - self.target_position.lon)**2) ** 0.5)

    def select_next_position(self, drone: Drone, waypoints: list[Location]):
        """Select the next waypoint based on pheromone levels and heuristic information."""
        probabilities = []
        total = 0
        
        for wp in waypoints:
            if wp != drone.position:
                pheromone = self.pheromone_map.get((drone.position, wp), 1.0)
                heuristic_value = self.heuristic(wp)
                probability = (pheromone ** self.alpha) * (heuristic_value ** self.beta)
                probabilities.append((wp, probability))
                total += probability
        
        # Normalize probabilities
        probabilities = [(wp, prob / total) for wp, prob in probabilities]
        
        # Choose next waypoint based on probability
        rand_value = random.uniform(0, 1)
        cumulative = 0
        for wp, prob in probabilities:
            cumulative += prob
            if rand_value <= cumulative:
                return wp
        return waypoints[-1]  # Fallback in case of rounding errors
    
    def update_pheromones(self):
        """Update pheromone levels based on the paths taken by drones."""
        for drone in self.drones:
            for i in range(len(drone.path) - 1):
                wp1, wp2 = drone.path[i], drone.path[i+1]
                self.pheromone_map[(wp1, wp2)] = (1 - self.evaporation_rate) * self.pheromone_map.get((wp1, wp2), 1.0)
                self.pheromone_map[(wp1, wp2)] += 1 / self.heuristic(wp2)  # Deposit pheromone based on fitness
    
    def run_aco(self):
        """Run ACO swarm optimization."""
        waypoints = []
        for drone in self.drones:
            drone.connect()
            drone.arm_and_takeoff(self.target_position.alt)
            drone.update_position(self.home_position)
            waypoints.append(drone.position)

        print([str(w) for w in waypoints])
        self.initialize_pheromone_map(waypoints)
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            for drone in self.drones:
                next_position = self.select_next_position(drone, waypoints)
                drone.move_to_position(next_position)
                
                # If a drone reaches the target, stop
                if next_position == self.target_position:
                    print("Target found!")
                    self.return_home()
                    return
            
            self.update_pheromones()
            time.sleep(2)
        
        self.return_home()
    
    def return_home(self):
        for drone in self.drones:
            drone.return_to_launch()
            drone.disconnect()
        print("Mission Complete. Drones returned.")

# Define the home position
home_position = Location(26.8616042, 75.8124822, 0)

if __name__ == "__main__":
    connection_strings = [
        "tcp:127.0.0.1:5762",
        "tcp:127.0.0.1:5773",
        "tcp:127.0.0.1:5782"
    ]

    target_position = Location(-35.3622810, 149.1650623, 10, home_position)
    static_obstacles = [
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
    ]

    swarm = DroneSwarmACO(connection_strings, home_position, target_position)
    swarm.run_aco()