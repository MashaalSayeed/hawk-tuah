from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Connect to the vehicle
print("Connecting...")
vehicle = connect('udp:127.0.0.1:14551', wait_ready=True)

# Get some basic info
print(f"GPS: {vehicle.gps_0}")
print(f"Battery: {vehicle.battery}")
print(f"Mode: {vehicle.mode.name}")

# Example: Arm and Takeoff
def arm_and_takeoff(target_altitude):
    print(f"Waiting for position estimate......")
    while not vehicle.gps_0.fix_type >= 3 or not vehicle.ekf_ok:
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f}")
        if alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

arm_and_takeoff(10)

# Land
print("Landing...")
vehicle.mode = VehicleMode("LAND")
time.sleep(5)

print("Vehicle Status:")
print(f"GPS: {vehicle.gps_0}")
print(f"Battery: {vehicle.battery}")
print(f"Mode: {vehicle.mode.name}")

# Close connection
vehicle.close()
