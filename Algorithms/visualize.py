import csv
import matplotlib.pyplot as plt
import numpy as np

# Read results from CSV file
results = []
with open('results.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        results.append({
            "Name": row["Simulation"],
            "Completed": int(row["Completed"]),
            "Iterations": float(row["Iterations"]),
            "Leader Energy": float(row["Leader Energy"]),
            "Follower Energy": float(row["Follower Energy"]),
        })

# Extract data
names = [result["Name"] for result in results]
completed = [result["Completed"] for result in results]
iterations = [result["Iterations"] for result in results]
leader_energy = [result["Leader Energy"] for result in results]
follower_energy = [result["Follower Energy"] for result in results]

# Plot configurations
x = np.arange(len(names))  # X-axis positions
width = 0.4  # Width of bars

# Function to add values on top of bars
def add_values(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 1),  # Offset above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='black')

# Create 4 separate plots with larger figure size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of plots

# Apply a consistent style for all plots
# plt.style.use('seaborn-darkgrid')

# Plot 1: Completion Percentage
ax1 = axes[0][0]
bars1 = ax1.bar(x, completed, width, color='royalblue')
ax1.set_title("Completion Percentage (%)", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, fontsize=10)
ax1.set_ylabel("Completed (%)", fontsize=12)
add_values(bars1, ax1)

# Plot 2: Iterations Required
ax2 = axes[0][1]
bars2 = ax2.bar(x, iterations, width, color='mediumseagreen')
ax2.set_title("Avg Iterations Required", fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, fontsize=10)
ax2.set_ylabel("Iterations", fontsize=12)
add_values(bars2, ax2)

# Plot 3: Leader Energy Used
ax3 = axes[1][0]
bars3 = ax3.bar(x, leader_energy, width, color='indianred')
ax3.set_title("Avg Leader Energy Used (Joules)", fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(names, rotation=15, fontsize=10)
ax3.set_ylabel("Energy (J)", fontsize=12)
add_values(bars3, ax3)

# Plot 4: Follower Energy Used
ax4 = axes[1][1]
bars4 = ax4.bar(x, follower_energy, width, color='goldenrod')
ax4.set_title("Avg Follower Energy Used (Joules)", fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(names, rotation=15, fontsize=10)
ax4.set_ylabel("Energy (J)", fontsize=12)
add_values(bars4, ax4)

# Adjust layout for better spacing and add a main title
fig.suptitle("Comparison of Fire Detection Strategies", fontsize=16, fontweight='bold')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.show()
