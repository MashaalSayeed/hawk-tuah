# results = [
#     {
#         "Name": "Grid Sweep",
#         "Completed": 100,
#         "Iterations": 351.73,
#         "Leader Energy": 232.76299999999802,
#         "Follower Energy": 339.92350000000135,
#     },
#     {
#         "Name": "ACO",
#         "Completed": 97,
#         "Iterations": 302.07,
#         "Leader Energy": 182.68479999999917,
#         "Follower Energy": 292.0110000000011,
#     },
#     {
#         "Name": "Modified ACO",
#         "Completed": 100,
#         "Iterations": 277.81,
#         "Leader Energy": 165.87939999999833,
#         "Follower Energy": 267.91850000000005,
#     }
# ]

import csv
import matplotlib.pyplot as plt
import numpy as np


results = []
with open('results2.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        results.append({
            "Name": row["Simulation"],
            "Completed": int(row["Completed"]),
            "Iterations": float(row["Iterations"]),
            "Leader Energy": float(row["Leader Energy"]),
            "Follower Energy": float(row["Follower Energy"]),
        })

# Data
names = [result["Name"] for result in results]
completed = [result["Completed"] for result in results]
iterations = [result["Iterations"] for result in results]
leader_energy = [result["Leader Energy"] for result in results]
follower_energy = [result["Follower Energy"] for result in results]

# Bar width
width = 0.2  # Width of the bars
x = np.arange(len(names))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - 1.5 * width, completed, width, label="Completed (%)")
bars2 = ax.bar(x - 0.5 * width, iterations, width, label="Iterations")
bars3 = ax.bar(x + 0.5 * width, leader_energy, width, label="Leader Energy")
bars4 = ax.bar(x + 1.5 * width, follower_energy, width, label="Follower Energy")

# Function to add values on top of bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # Offset above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='black')

# Adding values
for bars in [bars1, bars2, bars3, bars4]:
    add_values(bars)

# Labels and title
ax.set_xlabel("Fire Detection Strategy")
ax.set_ylabel("Value")
ax.set_title("Comparison of Fire Detection Strategies for 100 Simulations")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

