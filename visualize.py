import pandas as pd
import matplotlib.pyplot as plt

target_position = {"lat": -35.3622810, "lon": 149.1650623, "alt": 10}
def visualize_swarm_movement(log_file="swarm_log.csv"):
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each drone's path
    for drone_id in df["Drone ID"].unique():
        drone_data = df[df["Drone ID"] == drone_id]
        plt.plot(drone_data["Longitude"], drone_data["Latitude"], marker="o", linestyle="--", label=f"Drone {drone_id}")
    
    # Plot target position
    plt.scatter(target_position["lon"], target_position["lat"], color="red", marker="X", s=200, label="Target Position")
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Swarm Movement Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitness(log_file="swarm_log.csv"):
    df = pd.read_csv(log_file)
    
    # Get unique iterations and their global best fitness values
    iteration_fitness = df.groupby("Iteration")["Global Best Fitness"].min()
    
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_fitness.index, iteration_fitness.values, marker="o", linestyle="-", color="b")
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Fitness")
    plt.title("PSO Optimization - Global Best Fitness Over Iterations")
    plt.grid(True)
    plt.show()

# Call this function after running the swarm simulation
visualize_swarm_movement()
plot_fitness()
