import math
import random
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative


class Drone:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.vehicle = None
        self.position = None  # Current position (lat, lon, alt)
        self.pbest = None  # Personal best position (lat, lon, alt)
        self.pbest_fitness = float("inf")  # Personal best fitness
        self.current_fitness = float("inf")  # Current fitness

    def connect(self):
        print(f"Connecting to vehicle on {self.connection_string}...")
        self.vehicle = connect(self.connection_string, wait_ready=True)
        print(f"Connected to vehicle on {self.connection_string}.")

    def arm_and_takeoff(self, altitude):
        print(f"Waiting for position estimate...... ({self.connection_string})")
        while not self.vehicle.gps_0.fix_type >= 3 or not self.vehicle.ekf_ok:
            time.sleep(1)

        print(f"Arming motors for {self.connection_string}...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print(f" Waiting for arming... ({self.connection_string})")
            time.sleep(1)
        print(f"Vehicle armed ({self.connection_string}).")

        print(f"Taking off to {altitude} meters... ({self.connection_string})")
        self.vehicle.simple_takeoff(altitude)
        while True:
            if self.vehicle.location.global_relative_frame.alt >= altitude * 0.95:
                print(f"Reached target altitude. ({self.connection_string})")
                break
            time.sleep(1)

    def update_position(self):
        """Update the current position of the drone."""
        location = self.vehicle.location.global_relative_frame
        self.position = {"lat": location.lat, "lon": location.lon, "alt": location.alt}
        return self.position

    def move_to_position(self, target_position):
        """Moves the drone to the given target position."""
        waypoint = LocationGlobalRelative(target_position["lat"], target_position["lon"], target_position["alt"])
        self.vehicle.simple_goto(waypoint)

    def get_distance_meters(self, location1, location2):
        """Calculate the ground distance in meters between two locations."""
        dlat = location2["lat"] - location1["lat"]
        dlon = location2["lon"] - location1["lon"]
        return math.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * math.cos(math.radians(location1["lat"]))) ** 2)
    
    def return_to_launch(self):
        print(f"Returning to Launch for {self.connection_string}...")
        self.vehicle.mode = VehicleMode("RTL")
        while self.vehicle.mode.name != "RTL":
            time.sleep(1)
        print(f"Vehicle is returning to launch. ({self.connection_string})")

    def disconnect(self):
        self.vehicle.close()
        print(f"Vehicle disconnected from {self.connection_string}.")