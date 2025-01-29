import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
import threading

w = 0.5       # Inertia weight
c1 = 1.5      # Cognitive coefficient
c2 = 1.5      # Social coefficient


class Drone:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.vehicle = None

    def connect(self):
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
            self.vehicle.armed = True
            time.sleep(1)
        print(f"Vehicle armed ({self.connection_string}).")

        print(f"Taking off to {altitude} meters... ({self.connection_string})")
        self.vehicle.simple_takeoff(altitude)
        while True:
            # print(f" Altitude: {self.vehicle.location.global_relative_frame.alt:.2f} ({self.connection_string})")
            if self.vehicle.location.global_relative_frame.alt >= altitude * 0.95:
                print(f"Reached target altitude. ({self.connection_string})")
                break
            time.sleep(1)

    def get_current_location(self):
        location = self.vehicle.location.global_relative_frame
        print(f"Current Location for {self.connection_string} -> Latitude: {location.lat}, Longitude: {location.lon}, Altitude: {location.alt}")
        return location

    def go_south(self, distance, altitude):
        print(f"Moving {distance} meters south for {self.connection_string}...")
        current_location = self.vehicle.location.global_relative_frame
        target_lat = current_location.lat - (distance / 111320)  # Convert meters to degrees
        target_lon = current_location.lon
        target_location = LocationGlobalRelative(target_lat, target_lon, altitude)
        self.vehicle.simple_goto(target_location)

        while True:
            current_location = self.vehicle.location.global_relative_frame
            remaining_distance = self.get_distance_meters(current_location, target_location)
            # print(f" Distance to target: {remaining_distance:.2f} meters ({self.connection_string})")
            if remaining_distance < 2:
                print(f"Reached target location. ({self.connection_string})")
                break
            time.sleep(1)

    def perform_circles(self, center_lat, center_lon, radius, altitude, num_circles):
        print(f"Performing {num_circles} circle(s) around ({center_lat}, {center_lon}) at {altitude} meters altitude for {self.connection_string}.")
        for _ in range(num_circles):
            for angle in range(0, 360, 10):  # Adjust step for smoother or quicker circles
                angle_rad = math.radians(angle)
                target_lat = center_lat + (radius / 111320) * math.cos(angle_rad)
                target_lon = center_lon + (radius / (111320 * math.cos(math.radians(center_lat)))) * math.sin(angle_rad)
                waypoint = LocationGlobalRelative(target_lat, target_lon, altitude)
                self.vehicle.simple_goto(waypoint)
                while True:
                    current_location = self.vehicle.location.global_relative_frame
                    distance = self.get_distance_meters(current_location, waypoint)
                    # print(f" Distance to circle point: {distance:.2f} meters ({self.connection_string})")
                    if distance < 2:
                        break
                    time.sleep(0.5)
        print(f"Completed {num_circles} circle(s) for {self.connection_string}.")

    def return_to_launch(self):
        print(f"Returning to Launch for {self.connection_string}...")
        self.vehicle.mode = VehicleMode("RTL")
        while self.vehicle.mode.name != "RTL":
            # print(f" Waiting for RTL mode to activate... ({self.connection_string})")
            time.sleep(1)
        print(f"Vehicle is returning to launch. ({self.connection_string})")

    def get_distance_meters(self, location1, location2):
        """Calculate the ground distance in meters between two locations."""
        dlat = location2.lat - location1.lat
        dlon = location2.lon - location1.lon
        return math.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(location1.lat)))**2)

    def disconnect(self):
        self.vehicle.close()
        print(f"Vehicle disconnected from {self.connection_string}.")

    def execute_mission(self, takeoff_altitude, south_distance, circle_radius, num_circles):
        self.arm_and_takeoff(takeoff_altitude)
        self.get_current_location()
        self.go_south(south_distance, takeoff_altitude)
        current_location = self.vehicle.location.global_relative_frame
        self.perform_circles(current_location.lat, current_location.lon, circle_radius, takeoff_altitude, num_circles)
        self.return_to_launch()

def mission_for_drone(drone, takeoff_altitude, south_distance, circle_radius, num_circles):
    drone.execute_mission(takeoff_altitude, south_distance, circle_radius, num_circles)
    drone.disconnect()

def main():
    # Example connection strings for multiple drones (these would need to be actual addresses)
    connection_strings = [
        "tcp:127.0.0.1:5762",  # Drone 1
        "tcp:127.0.0.1:5773",  # Drone 2
        "tcp:127.0.0.1:5782"
        # Add more drones as needed
    ]

    # Mission parameters
    takeoff_altitude = 5  # meters
    south_distance = 30  # meters
    circle_radius = 10  # meters
    num_circles = 1

    # Create drone objects and start a thread for each drone
    drones = []
    threads = []
    for connection_string in connection_strings:
        drone = Drone(connection_string)
        drone.connect()
        drones.append(drone)
        thread = threading.Thread(target=mission_for_drone, args=(drone, takeoff_altitude, south_distance, circle_radius, num_circles))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All missions completed. Vehicles disconnected.")

if __name__ == "__main__":
    main()
