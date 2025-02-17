import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
file_path = "aco_log.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Create a figure
plt.figure(figsize=(10, 6))

# Unique drones
drones = df["Drone ID"].unique()

# Plot drone paths with pheromone intensity
for drone_id in drones:
    drone_data = df[df["Drone ID"] == drone_id]
    
    plt.scatter(drone_data["Longitude"], drone_data["Latitude"], 
                c=drone_data["Pheromone"], cmap="viridis", 
                edgecolors='k', label=f"Drone {drone_id}", alpha=0.7)

    plt.plot(drone_data["Longitude"], drone_data["Latitude"], linestyle="--", alpha=0.5)

# Mark the fire hotspot
fire_location = (-35.3622191, 149.1650770)  # Adjust based on your actual fire position
plt.scatter(fire_location[1], fire_location[0], marker="*", color="red", s=200, label="Fire Hotspot")

# Add labels
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Drone Swarm ACO Path Visualization")

# Add a color bar for pheromone intensity
plt.colorbar(label="Pheromone Level")

# Show legend
plt.legend()

# Show plot
plt.show()
