# Simulation Setup

1. Start ArduPilot SITL with n drones
2. Use Mavproxy to forward mavlink connection to
   ```
   tcp://127.0.0.1:5760
   tcp://127.0.0.1:5770
   tcp://127.0.0.1:5780
   ...
   ```
3. Connect GCS to visualize drone movement (optional)
4. Use web interface to send fire location command to backend server
5. Backend service will start the mission using swarm algorithms
6. Swarm simulation will then initializ random fire locations around the map using a fire spread model
7. Drones connected to DroneKit will be controlled by swarm algorithms to suppress the fire locations


```bash
docker exec -it sitl ./Tools/autotest/sim_vehicle.py -v ArduCopter -f quad -l 26.861264,75.813278,0,0 -w --instance 0 --mavproxy-args="--out udp:host.docker.internal:14550 --out udp:host.docker.internal:14551 --state-basedir=/tmp/mavlink-sitl0"
```

HOME_LOCATION=26.861264,75.813278,0,0
TARGET_LOCATION=26.861406,75.812826,0,0