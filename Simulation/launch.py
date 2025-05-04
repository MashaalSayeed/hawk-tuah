import threading
import signal
import sys
import time
from dronekit_sitl import SITL

# Define the start positions
start_positions = [
    {"lat": 26.8616042, "lon": 75.8124822, "alt": 584, "heading": 0},  # Original position
    {"lat": 26.8617000, "lon": 75.8125000, "alt": 584, "heading": 0},  # Slightly north-east
    {"lat": 26.8615000, "lon": 75.8124600, "alt": 584, "heading": 0},  # Slightly south-west
]

# List to store SITL instances
sitls = []

# Function to launch a single SITL instance
def launch_sitl(index, position):
    sitl = SITL()
    sitl.download('copter', '3.3')

    sitl_args = [
        f"-I{index}",
        "--model", "quad",
        f"--home={position['lat']},{position['lon']},{position['alt']},{position['heading']}"
    ]
    
    sitl.launch(sitl_args, await_ready=True, restart=True)
    print(f"Drone {index} launched at: {sitl.connection_string()}")

    sitl.block_until_ready(verbose=True)
    sitls.append(sitl)

# Function to handle Ctrl+C and shut down all SITL instances
def shutdown_sitl(signal_received=None, frame=None):
    print("\nShutting down all SITL instances...")
    for sitl in sitls:
        sitl.stop()
    sys.exit(0)

# Register the shutdown function for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, shutdown_sitl)

# Create and start threads
threads = []
for i, pos in enumerate(start_positions):
    thread = threading.Thread(target=launch_sitl, args=(i, pos))
    thread.start()
    threads.append(thread)

    print(f"SITL {i} launched at {pos['lat']}, {pos['lon']}.")


# Wait for all threads to complete
for thread in threads:
    thread.join()
