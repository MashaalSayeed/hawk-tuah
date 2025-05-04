import time
import math
import threading

from drone import Drone

class LFSwarm:
    def __init__(self, drones: list[Drone], leader_index=0):
        self.drones = drones
        self.leader: Drone = drones[leader_index] if drones else None
        self.leader.role = "leader"

    def initialize_swarm(self, altitude):
        def initialize_drone(drone):
            drone.connect()
            drone.arm_and_takeoff(altitude)
            # drone.get_status()
            print(drone.position)

        threads = []
        for drone in self.drones:
            t = threading.Thread(target=initialize_drone, args=(drone,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print("Swarm initialized.")

    def move_leader_to(self, target_position):
        if self.leader:
            print(f"Leader {self.leader.connection_string} moving to {target_position}.")
            self.leader.move_to_position(target_position)

    def update_follower_positions(self):
        if not self.leader:
            print("No leader assigned.")
            return

        leader_position = self.leader.position
        for i, drone in enumerate(self.drones):
            if drone != self.leader:
                side = -1 if i % 2 == 0 else 1
                row = (i + 1) // 2

                offset_lat = row  * 0.0001
                offset_lon = side * row * 0.0001
                follower_target = {
                    "lat": leader_position["lat"] + offset_lat,
                    "lon": leader_position["lon"] + offset_lon,
                    "alt": leader_position["alt"]
                }

                print(f"Follower {drone.connection_string} moving to formation position {follower_target}.")
                drone.move_to_position(follower_target)

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
        self.move_leader_to(target_position)
        while True:
            self.leader.update_position()
            self.update_follower_positions()
            leader_dist = math.sqrt((self.leader.position["lat"] - target_position["lat"]) ** 2 + (self.leader.position["lon"] - target_position["lon"]) ** 2)
            print(f"Leader reached: {leader_dist}")
            if leader_dist < 0.00001:
                print("Leader has reached the target position.")
                break
            time.sleep(1)
        print("Swarm reached the destination.")

if __name__ == "__main__":
    drones = [
        Drone("tcp:127.0.0.1:5762"),
        Drone("tcp:127.0.0.1:5772"),
        Drone("tcp:127.0.0.1:5782")
    ]

    swarm = LFSwarm(drones)
    swarm.run_swarm({"lat": -35.3622810, "lon": 149.1650623, "alt": 10})
    swarm.return_to_launch()
    swarm.disconnect_all()
