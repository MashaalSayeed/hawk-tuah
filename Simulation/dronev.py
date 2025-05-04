import time
from dronekit import connect, VehicleMode, LocationGlobalRelative

class Drone:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.vehicle = None
        self.position = {"lat": 0, "lon": 0, "alt": 0}
        self.velocity = {"x": 0, "y": 0}
        self.path = []

    def connect(self):
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
            time.sleep(1)
        print(f"Vehicle armed ({self.connection_string}).")

        print(f"Taking off to {altitude} meters... ({self.connection_string})")
        self.vehicle.simple_takeoff(altitude)
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            if current_alt >= altitude * 0.95:
                print(f"Reached target altitude. ({self.connection_string})")
                break
            time.sleep(1)
        
        self.update_position()

    def update_position(self):
        """Update the current position of the drone in relative coordinates."""
        location = self.vehicle.location.global_relative_frame
        self.position = {"lat": location.lat, "lon": location.lon, "alt": location.alt}
        self.path.append(self.position)
        return self.position

    def move_to_position(self, target_position):
        """Moves the drone to the given target position."""
        waypoint = LocationGlobalRelative(target_position["lat"], target_position["lon"], target_position["alt"])
        self.vehicle.simple_goto(waypoint)

    def return_to_launch(self):
        print(f"Returning to Launch for {self.connection_string}...")
        self.vehicle.mode = VehicleMode("RTL")
        while self.vehicle.mode.name != "RTL":
            time.sleep(1)

    def disconnect(self):
        self.vehicle.close()
        print(f"Vehicle disconnected from {self.connection_string}.")
