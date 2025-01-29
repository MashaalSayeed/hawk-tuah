import time
import requests
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative


API_URL = "http://localhost:5050"
running = True
commands = [
  {"action": "ARM", "arguments": {}},
  {"action": "TAKEOFF", "arguments": {"altitude": 10}},
  {"action": "WAYPOINT", "arguments": {"latitude": -35.363261, "longitude": 149.165230, "altitude": 10}},
  {"action": "RETURN", "arguments": {}},
  {"action": "LAND", "arguments": {}},
  {"action": "CHANGE_MODE", "arguments": {"mode": "LOITER"}},
  {"action": "VELOCITY", "arguments": {"forward": 5, "right": 0, "up": 0}}
]


def arm_vehicle(vehicle):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)
    print("Vehicle armed.")

def takeoff_vehicle(vehicle, altitude):
    print(f"Taking off to {altitude} meters...")
    vehicle.simple_takeoff(altitude)
    while True:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
            print("Reached target altitude.")
            break
        time.sleep(1)

def go_to_waypoint(vehicle, latitude, longitude, altitude):
    print(f"Flying to waypoint: Lat={latitude}, Lon={longitude}, Alt={altitude} meters")
    waypoint = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(waypoint)
    time.sleep(10)

def return_to_launch(vehicle):
    print("Returning to Launch...")
    vehicle.mode = VehicleMode("RTL")

def land_vehicle(vehicle):
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(" Waiting for disarm after landing...")
        time.sleep(1)
    print("Vehicle landed and disarmed.")

def change_mode(vehicle, mode):
    print(f"Changing mode to {mode}...")
    vehicle.mode = VehicleMode(mode)
    while vehicle.mode.name != mode:
        print(f" Waiting for mode change to {mode}...")
        time.sleep(1)
    print(f"Mode changed to {mode}.")

def change_velocity(vehicle, forward, right, up):
    print(f"Changing velocity: Forward={forward}, Right={right}, Up={up}")
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b110111111000, 0, 0, 0, forward, right, up, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def execute_command(vehicle, command):
    global running
    action = command.get("action")
    arguments = command.get("arguments", {})

    if action == "ARM":
        arm_vehicle(vehicle)
    elif action == "TAKEOFF":
        takeoff_vehicle(vehicle, **arguments)
    elif action == "WAYPOINT":
        go_to_waypoint(vehicle, **arguments)
    elif action == "RETURN":
        return_to_launch(vehicle)
    elif action == "LAND":
        land_vehicle(vehicle)
    elif action == "CHANGE_MODE":
        change_mode(vehicle, **arguments)
    elif action == "VELOCITY":
        change_velocity(vehicle, **arguments)
    elif action == "DISCONNECT":
        running = False
    else:
        print(f"Unknown action: {action}")


def post_telemetry(vehicle):
    payload = {"status": vehicle.mode.name}
    try:
        response = requests.post(f"{API_URL}/telemetry", json=payload)
        response.raise_for_status()
        print(f"Telemetry posted: {payload}")
    except Exception as e:
        print(f"Error posting telemetry: {e}")


def fetch_command():
    try:
        response = requests.get(f"{API_URL}/command/execute")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch commands: {e}")
        return None


def main():
    connection_string = "tcp:127.0.0.1:5762"

    print(f"Connecting to vehicle on {connection_string}...")
    vehicle = connect(connection_string, wait_ready=True)
    print("Connected to vehicle.")

    while running == True:
        # Step 1: Post telemetry status
        post_telemetry(vehicle)
        command = fetch_command()
        print(command)

        if command and command['command']:
            execute_command(vehicle, command['command'])

        print("Waiting for next fetch cycle...")
        time.sleep(1)

    vehicle.close()

if __name__ == "__main__":
    main()