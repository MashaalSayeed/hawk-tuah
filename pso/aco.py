import random
import time
import math
import csv
from drone import Drone

class DroneSwarmACO:
    def __init__(self, connection_strings, target_position, max_iterations=200, evaporation_rate=0.1, alpha=1, beta=2, logging=False):
        self.drones = [Drone(conn) for conn in connection_strings]
        self.target_position = target_position  # Fire hotspot location
        self.max_iterations = max_iterations
        self.evaporation_rate = evaporation_rate  # Pheromone evaporation rate
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Heuristic influence
        self.pheromone_map = {}  # Dictionary to store pheromone levels on paths
        self.logging = logging
        self.log_file = "aco_log.csv"

        if self.logging:
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "Drone ID", "Latitude", "Longitude", "Altitude", "Pheromone", "Heuristic"])

    def heuristic(self, position):
        """Inverse distance to target as heuristic."""
        return 1 / (0.0001 + math.sqrt((position['lat'] - self.target_position['lat'])**2 +
                                      (position['lon'] - self.target_position['lon'])**2))

    def initialize_pheromone_map(self):
        """Initialize pheromone levels for drones."""
        for drone in self.drones:
            self.pheromone_map[drone] = 1.0  # Initial pheromone level for each drone

    def select_next_position(self, drone):
        """Select the next position based on pheromone levels and heuristic information."""
        pheromone = self.pheromone_map.get(drone, 1.0)
        heuristic_value = self.heuristic(drone.position)
        probability = (pheromone ** self.alpha) * (heuristic_value ** self.beta)
        
        # Randomized exploration-exploitation strategy
        if random.random() < probability:
            lat_offset = random.uniform(-0.0001, 0.0001)
            lon_offset = random.uniform(-0.0001, 0.0001)
            return {'lat': drone.position['lat'] + lat_offset, 'lon': drone.position['lon'] + lon_offset, 'alt': drone.position['alt']}
        else:
            return self.target_position  # Move towards target if probability is low

    def update_pheromones(self):
        """Update pheromone levels based on the paths taken by drones."""
        for drone in self.drones:
            pheromone = self.pheromone_map.get(drone, 1.0)
            self.pheromone_map[drone] = (1 - self.evaporation_rate) * pheromone
            self.pheromone_map[drone] += 1 / self.heuristic(drone.position)

    def log_positions(self, iteration):
        """Logs all drone positions and pheromone levels to a CSV file."""
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            for i, drone in enumerate(self.drones):
                writer.writerow([iteration, i, drone.position['lat'], drone.position['lon'], drone.position['alt'], self.pheromone_map.get(drone, 1.0), self.heuristic(drone.position)])

    def run_aco(self):
        """Run the ACO algorithm for fire hotspot detection."""
        self.initialize_pheromone_map()
        
        for drone in self.drones:
            drone.connect()
            drone.arm_and_takeoff(self.target_position['alt'])
            drone.update_position()
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            for drone in self.drones:
                next_position = self.select_next_position(drone)
                drone.move_to_position(next_position)
                drone.update_position()
                
                if math.isclose(next_position['lat'], self.target_position['lat'], abs_tol=0.0001) and \
                   math.isclose(next_position['lon'], self.target_position['lon'], abs_tol=0.0001):
                    print("Fire hotspot found!")
                    self.return_home()
                    return
            
            self.update_pheromones()
            self.log_positions(iteration)
            time.sleep(2)
        
        self.return_home()
    
    def return_home(self):
        """Command all drones to return home."""
        for drone in self.drones:
            drone.return_to_launch()
            drone.disconnect()
        print("Mission Complete. Drones returned.")


if __name__ == "__main__":
    connection_strings = [
        "tcp:127.0.0.1:5762",
        "tcp:127.0.0.1:5773",
        "tcp:127.0.0.1:5782"
    ]

    target_position = {"lat": -35.3622191, "lon": 149.1650770, "alt": 10}
    static_obstacles = [
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
    ]

    swarm = DroneSwarmACO(connection_strings, target_position, logging=True)
    swarm.run_aco()