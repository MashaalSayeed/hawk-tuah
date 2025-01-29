# Hawk Tuah

This project demonstrates a **Particle Swarm Optimization (PSO)** algorithm applied to a swarm of drones using DroneKit. The drones optimize their path to a target position while avoiding obstacles, simulating a real-world drone swarm scenario. The system includes functionality to connect drones, control their movement, and optimize swarm performance using PSO.

## Requirements

```sh
pip install dronekit matplotlib pandas
```

## Setup

1. Install requirements
2. Connect to drones through SITL for simulation (using MissionPlanner)
3. Get connection strings for each drone
4. Start simulation

```sh
python swarm.py
```

## Visualization

After running the optimization, drone positions and fitness metrics in `swarm_log.csv`. To visualize this data run:

```sh
python visualize.py
```
