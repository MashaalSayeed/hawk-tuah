import math
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative

class Location:
    def __init__(self, lat, lon, alt, home_position=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.home_position = home_position or self
        self.x, self.y = self.gps_to_relative()
    
    def gps_to_relative(self):
        dlat = (self.lat - self.home_position.lat) * 111320
        dlon = (self.lon - self.home_position.lon) * 111320 * math.cos(math.radians(self.home_position.lat))
        return dlat, dlon

    def to_global(self):
        lat_offset = self.x / 111320 + self.home_position.lat
        lon_offset = self.y / (111320 * math.cos(math.radians(self.home_position.lat))) + self.home_position.lon
        return LocationGlobalRelative(lat_offset, lon_offset, self.alt)
    
    def __str__(self):
        return f"Location(lat={self.lat}, lon={self.lon}, alt={self.alt})"
    
    def __eq__(self, other):
        """Check if two locations are equal."""
        if isinstance(other, Location):
            return (self.lat, self.lon, self.alt) == (other.lat, other.lon, other.alt)
        return False

    def __hash__(self):
        """Make Location instances hashable."""
        return hash((self.lat, self.lon, self.alt))


class Drone:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.vehicle = None
        self.position = None
        self.velocity = {"x": 0, "y": 0}  # Velocity in meters
        self.pbest = None  # Personal best position
        self.pbest_fitness = float("inf")  # Personal best fitness
        self.current_fitness = float("inf")  # Current fitness
        self.path = []  # List of waypoints to follow

    def connect(self):
        print(f"Connecting to vehicle on {self.connection_string}...")
        self.vehicle = connect(self.connection_string, wait_ready=True)
        print(f"Connected to vehicle on {self.connection_string}.")

    def get_status(self):
        print(f"Vehicle status: {self.vehicle.system_status.state} ({self.connection_string})")
        print(f"GPS: {self.vehicle.gps_0.fix_type}")
        print(f"Battery: {self.vehicle.battery}")
        print(f"Mode: {self.vehicle.mode.name}")
        print(f"Attitude: {self.vehicle.attitude}")

    def get_parameters(self):
        print(f"Vehicle parameters: {list(self.vehicle.parameters.items())}")

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

    def update_position(self, home_position: Location):
        """Update the current position of the drone in relative coordinates."""
        location = self.vehicle.location.global_relative_frame
        self.position = Location(location.lat, location.lon, location.alt, home_position)
        self.path.append(self.position)
        return self.position

    def move_to_position(self, target_position: Location):
        """Moves the drone to the given target position."""
        waypoint = target_position.to_global()
        self.vehicle.simple_goto(waypoint)
    
    def return_to_launch(self):
        print(f"Returning to Launch for {self.connection_string}...")
        self.vehicle.mode = VehicleMode("RTL")
        while self.vehicle.mode.name != "RTL":
            time.sleep(1)
        print(f"Vehicle is returning to launch. ({self.connection_string})")

    def disconnect(self):
        self.vehicle.close()
        print(f"Vehicle disconnected from {self.connection_string}.")