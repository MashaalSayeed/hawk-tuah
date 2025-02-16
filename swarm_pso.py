from drone import Drone, Location
import random
import time
import csv


class DroneSwarm:
    def __init__(self, connection_strings: list[str], home_position: Location, target_position: Location, max_iterations=30, static_obstacles=None, log_file="swarm_log.csv", logging=False):
        self.drones = [Drone(conn) for conn in connection_strings]
        self.target_position = target_position
        self.home_position = home_position
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

    def initialize_swarm(self):
        """Connect and take off all drones."""
        for drone in self.drones:
            drone.connect()
            drone.arm_and_takeoff(self.target_position.alt)

        # Initialize random positions and personal bests
        for drone in self.drones:
            drone.update_position(self.home_position)
            print(f"Drone {drone.connection_string} at {drone.position}")

            drone.pbest = drone.position
            drone.pbest_fitness = self.fitness(drone.pbest)

        # Initialize global best position
        best_drone = min(self.drones, key=lambda d: d.pbest_fitness)
        self.global_best_position = best_drone.pbest
        self.global_best_fitness = best_drone.pbest_fitness

    def fitness(self, position: Location):
        """Calculate how close a drone is to the target position."""
        return ((position.lat - self.target_position.lat) ** 2 +
                (position.lon - self.target_position.lon) ** 2) ** 0.5

    def distance_between(self, d1: Location, d2: Location):
        """Calculate the distance between two drones."""
        return ((d1.lat - d2.lat) ** 2 + (d1.lon - d2.lon) ** 2) ** 0.5
    
    def log_positions(self, iteration):
        """Logs all drone positions and fitness values to a CSV file."""
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            for i, drone in enumerate(self.drones):
                fitness = self.fitness(drone.position)
                writer.writerow([iteration, i, drone.position.lat, drone.position.lon, drone.position.alt, fitness, self.global_best_fitness])

    def update_velocities_and_positions(self):
        """Apply PSO algorithm with obstacle avoidance."""
        for drone in self.drones:
            # Update best positions
            drone.update_position(self.home_position)
            drone.current_fitness = self.fitness(drone.position)
            if drone.current_fitness < drone.pbest_fitness:
                drone.pbest = drone.position
                drone.pbest_fitness = drone.current_fitness

            # Update global best if needed
            if drone.current_fitness < self.global_best_fitness:
                self.global_best_position = drone.position
                self.global_best_fitness = drone.current_fitness

            # Calculate new velocity
            r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
            velocity_lat = self.w * (drone.pbest.lat - drone.position.lat) + \
                           self.c1 * r1 * (self.global_best_position.lat - drone.position.lat)
            velocity_lon = self.w * (drone.pbest.lon - drone.position.lon) + \
                           self.c2 * r2 * (self.global_best_position.lon - drone.position.lon)

            # # Collision Avoidance: Adjust movement if too close to another drone
            # for other_drone in self.drones:
            #     if other_drone != drone and self.distance_between(drone.position, other_drone.position) < self.collision_avoidance_distance:
            #         print(f"Collision Avoidance: Adjusting {drone.connection_string}")
            #         velocity_lat += random.uniform(-0.00001, 0.00001) * 5
            #         velocity_lon += random.uniform(-0.00001, 0.00001) * 5

            # Move drone
            new_position = Location(drone.position.lat + velocity_lat, drone.position.lon + velocity_lon, drone.position.alt, drone.position.home_position)
            drone.move_to_position(new_position)

    def run_pso(self):
        """Run PSO swarm optimization."""
        self.initialize_swarm()
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.update_velocities_and_positions()
            if self.logging:
                self.log_positions(iteration)

            print(f"Global Best Fitness: {self.global_best_fitness}")
            time.sleep(2)

        # Return all drones to launch
        self.return_home()

    def return_home(self):
        """Command all drones to return to launch."""
        for drone in self.drones:
            drone.return_to_launch()
            drone.disconnect()
        print("Mission Complete. Drones returned.")


home_position = Location(-35.3633156, 149.1652412, 0)
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

    swarm = DroneSwarm(connection_strings, home_position, target_position, static_obstacles=static_obstacles, logging=True)
    swarm.run_pso()