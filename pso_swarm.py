from drone import Drone
import random
import time
import csv
import threading

class DroneSwarm:
    def __init__(self, drones: list[Drone], target_position, max_iterations=30, static_obstacles=None, log_file="swarm_log.csv", logging=False):
        self.drones = drones
        self.target_position = target_position
        self.max_iterations = max_iterations
        self.logging = logging
        self.log_file = log_file
        self.static_obstacles = static_obstacles

        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.collision_avoidance_distance = 2  # Minimum safe distance in meters

        if self.logging:
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "Drone ID", "Latitude", "Longitude", "Altitude", "Fitness", "Global Best Fitness"])

    def initialize_swarm(self, altitude):
        def initialize_drone(drone: Drone):
            drone.connect()
            drone.arm_and_takeoff(altitude)
            # drone.get_status()
            print(f"Drone {drone.connection_string} at {drone.position}")

        threads = []
        for drone in self.drones:
            t = threading.Thread(target=initialize_drone, args=(drone,))
            threads.append(t)
            t.start()

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("Swarm initialization interrupted.")
            self.return_home()

        # Initialize random positions and personal bests
        for drone in self.drones:
            drone.update_position()
            # drone.position = Location(random.uniform(self.target_position.lat - 0.0005, self.target_position.lat + 0.0005),
            #                           random.uniform(self.target_position.lon - 0.0005, self.target_position.lon + 0.0005),
            #                           self.target_position.alt, )

            drone.pbest = drone.position
            drone.pbest_fitness = self.fitness(drone.pbest)

        # Initialize global best position
        best_drone = min(self.drones, key=lambda d: d.pbest_fitness)
        # self.global_best_position = best_drone.pbest.copy()
        # self.global_best_fitness = best_drone.pbest_fitness
        self.global_best_position = target_position.copy()
        self.global_best_fitness = self.fitness(target_position)
        print("Swarm initialized.")

    def fitness(self, position):
        """Calculate how close a drone is to the target position."""
        return ((position["lat"] - self.target_position["lat"]) ** 2 +
                (position["lon"] - self.target_position["lon"]) ** 2) ** 0.5
    
    def log_positions(self, iteration):
        """Logs all drone positions and fitness values to a CSV file."""
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            for i, drone in enumerate(self.drones):
                fitness = self.fitness(drone.position)
                writer.writerow([iteration, i, drone.position["lat"], drone.position["lon"], drone.position["alt"], fitness, self.global_best_fitness])

    def update_velocities_and_positions(self):
        """Apply PSO algorithm with obstacle avoidance."""
        for drone in self.drones:
            # Update best positions
            drone.update_position()
            drone.current_fitness = self.fitness(drone.position)
            if drone.current_fitness < drone.pbest_fitness:
                drone.pbest = drone.position
                drone.pbest_fitness = drone.current_fitness

            # Update global best if needed
            if drone.current_fitness < self.global_best_fitness:
                self.global_best_position = drone.position.copy()
                self.global_best_fitness = drone.current_fitness

            # Calculate new velocity
            r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
            velocity_lat = self.w * (drone.pbest["lat"] - drone.position["lat"]) + \
                           self.c1 * r1 * (self.global_best_position["lat"] - drone.position["lat"])
            velocity_lon = self.w * (drone.pbest["lon"] - drone.position["lon"]) + \
                           self.c2 * r2 * (self.global_best_position["lon"] - drone.position["lon"])

            # # Collision Avoidance: Adjust movement if too close to another drone
            # for other_drone in self.drones:
            #     if other_drone != drone and self.distance_between(drone.position, other_drone.position) < self.collision_avoidance_distance:
            #         print(f"Collision Avoidance: Adjusting {drone.connection_string}")
            #         velocity_lat += random.uniform(-0.00001, 0.00001) * 5
            #         velocity_lon += random.uniform(-0.00001, 0.00001) * 5

            # Move drone
            new_position = {
                "lat": drone.position["lat"] + velocity_lat, 
                "lon": drone.position["lon"] + velocity_lon, 
                "alt": drone.position["alt"]
            }
            drone.move_to_position(new_position)

    def run_pso(self):
        """Run PSO swarm optimization."""
        self.initialize_swarm(self.target_position["alt"])
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.update_velocities_and_positions()
            if self.logging:
                self.log_positions(iteration)

            current_best = min(self.drones, key=lambda d: d.pbest_fitness)
            print(f"Global Best Fitness: {current_best.pbest_fitness}")
            time.sleep(2)
        self.return_home()

    def return_home(self):
        """Command all drones to return to launch."""
        for drone in self.drones:
            drone.return_to_launch()
            drone.disconnect()
        print("Mission Complete. Drones returned.")



if __name__ == "__main__":
    drones = [
        Drone("tcp:127.0.0.1:5762"),
        Drone("tcp:127.0.0.1:5773"),
        Drone("tcp:127.0.0.1:5782"),
    ]

    target_position = {"lat": -35.3622810, "lon": 149.1650623, "alt": 10}
    static_obstacles = [
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
        {"lat": -35.3622810, "lon": 149.1650623, "alt": 10},
    ]

    swarm = DroneSwarm(drones, target_position, static_obstacles=static_obstacles, logging=True)
    swarm.run_pso()