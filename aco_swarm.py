import time
import math
import threading
import random

from drone import Drone

class ACOSwarm:
    def __init__(self, drones: list[Drone], evaporation_rate=0.01, pheromone_deposit=-1.0):
        self.drones = drones
        self.pheromone_map = {}  # Stores pheromone levels at (lat, lon) coordinates
        self.evaporation_rate = evaporation_rate  # Rate at which pheromones evaporate
        self.pheromone_deposit = pheromone_deposit  # Pheromone amount deposited by each drone
        self.max_iterations = 100

    def initialize_swarm(self, altitude):
        def initialize_drone(drone):
            drone.connect()
            drone.arm_and_takeoff(altitude)
            print(f"Drone {drone.connection_string} at position {drone.position}")

        threads = []
        for drone in self.drones:
            t = threading.Thread(target=initialize_drone, args=(drone,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print("Swarm initialized.")

    def update_pheromones(self):
        """Evaporate and update pheromone levels."""
        for pos in list(self.pheromone_map.keys()):
            self.pheromone_map[pos] *= (1 - self.evaporation_rate)
            if self.pheromone_map[pos] < 0.01:
                del self.pheromone_map[pos]  # Clean up negligible pheromones

    def deposit_pheromone(self, position, amount):
        """Deposit pheromones at the given position."""
        pos_key = (round(position["lat"], 5), round(position["lon"], 5))
        if pos_key not in self.pheromone_map:
            self.pheromone_map[pos_key] = 0.0
        self.pheromone_map[pos_key] += amount

    def choose_next_position(self, current_position):
        """Choose the next position based on pheromone concentration."""
        candidates = []
        step_size = 0.0001  # Adjust step size as needed
        for lat_offset in [-step_size, 0, step_size]:
            for lon_offset in [-step_size, 0, step_size]:
                if lat_offset == 0 and lon_offset == 0:
                    continue
                candidate = (
                    round(current_position["lat"] + lat_offset, 6),
                    round(current_position["lon"] + lon_offset, 6)
                )
                pheromone_level = self.pheromone_map.get(candidate, 0.01)  # Minimum pheromone
                candidates.append((candidate, pheromone_level))

        # Probabilistic selection based on pheromone levels
        total_pheromone = sum(p for _, p in candidates)
        if total_pheromone == 0:
            return current_position  # Stay if no pheromones

        # Weighted random choice
        r = random.uniform(0, total_pheromone)
        cumulative = 0
        for candidate, pheromone in candidates:
            cumulative += pheromone
            if r <= cumulative:
                return {"lat": candidate[0], "lon": candidate[1], "alt": current_position["alt"]}
        return current_position

    def return_to_launch(self):
        for drone in self.drones:
            drone.return_to_launch()
        print("Swarm returning to launch.")

    def disconnect_all(self):
        for drone in self.drones:
            drone.disconnect()
        print("All drones disconnected.")

    def run_swarm(self, target_position):
        self.initialize_swarm(target_position["alt"])

        for i in range(self.max_iterations):
            self.update_pheromones()
            for drone in self.drones:
                drone.update_position()
                self.deposit_pheromone(drone.position, self.pheromone_deposit)
                next_position = self.choose_next_position(drone.position)
                drone.move_to_position(next_position)
            time.sleep(1)

            # Check if all drones have reached the target
            all_reached = all(
                math.sqrt((drone.position["lat"] - target_position["lat"]) ** 2 + (drone.position["lon"] - target_position["lon"]) ** 2) < 0.00001
                for drone in self.drones
            )
            if all_reached:
                break
        print("Swarm reached the destination.")

if __name__ == "__main__":
    drones = [
        Drone("tcp:127.0.0.1:5762"),
        Drone("tcp:127.0.0.1:5772"),
        Drone("tcp:127.0.0.1:5782")
    ]

    swarm = ACOSwarm(drones)
    swarm.run_swarm({"lat": -35.3622810, "lon": 149.1650623, "alt": 10})
    swarm.return_to_launch()
    swarm.disconnect_all()