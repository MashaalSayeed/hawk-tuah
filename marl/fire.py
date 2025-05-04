import numpy as np
import random

# Constants
BASE_SPREAD_RATE = 0.15
BASE_BURN_PERIOD = 0.2  # β in the paper (average burning period)
SUPPRESSION_EFFECT = 0.5  # ∆β in the paper (fire controlling effect)
WIND_STRENGTH = 0.1

# States
GREEN = 0  # Not fired yet
RED = 1    # Currently burning
GRAY = 2   # Already burnt

class FireGrid:
    def __init__(self, grid_size, fire_count=10, fire_range=35, fire_offset=(0, 0), 
                 spread_rate=BASE_SPREAD_RATE, burn_period=BASE_BURN_PERIOD):
        self.width, self.height = grid_size
        self.grid_size = grid_size

        self.starting_fire_count = fire_count
        self.fire_range = fire_range
        self.fire_offset = fire_offset

        # α in the paper - probability of fire transfer
        self.alpha = spread_rate
        # β in the paper - average burning period
        self.beta = burn_period
        
        # Additional properties for cell tracking
        self.fuel_remaining = np.ones(self.grid_size)  # Amount of fuel in each cell
        self.burning_time = np.zeros(self.grid_size)   # How long a cell has been burning
        
        self.reset()

    def init_fires(self):
        """Initialize fire ignition points"""
        for _ in range(self.starting_fire_count):
            x = min(max(self.fire_offset[0] + np.random.randint(-self.fire_range, self.fire_range), 0), self.width-1)
            y = min(max(self.fire_offset[1] + np.random.randint(-self.fire_range, self.fire_range), 0), self.height-1)
            
            if self.grid[x, y] == GREEN:
                self.grid[x, y] = RED
                # Randomize fuel for variety
                self.fuel_remaining[x, y] = np.random.uniform(0.7, 1.0)

        self.total_fire_count = np.sum(self.grid == RED)

    def spread_fire(self):
        """Model fire spread according to the Markov model"""
        # Randomly update wind direction occasionally
        if random.random() < 0.05:
            self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

        new_grid = self.grid.copy()
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Process RED cells (burning)
                if self.grid[i, j] == RED:
                    # Update burning time
                    self.burning_time[i, j] += 1
                    
                    # Reduce fuel
                    burn_rate = np.random.uniform(0.05, 0.15)
                    self.fuel_remaining[i, j] -= burn_rate
                    
                    # Cell burns out when fuel is depleted or based on beta probability
                    if self.fuel_remaining[i, j] <= 0 or random.random() < self.beta:
                        new_grid[i, j] = GRAY
                        self.burning_time[i, j] = 0
                    
                    # Try to spread fire to neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx, ny = i + dx, j + dy
                        
                        # Check if neighbor is in bounds and unburnt
                        if (0 <= nx < self.width and 0 <= ny < self.height and 
                            self.grid[nx, ny] == GREEN):
                            
                            # Calculate spread probability (alpha)
                            spread_chance = self.alpha * self.fuel_remaining[i, j]
                            
                            # Wind effect
                            if (dx, dy) == self.wind_direction:
                                spread_chance += WIND_STRENGTH
                                
                            # Random spread based on probability
                            if random.random() < spread_chance:
                                new_grid[nx, ny] = RED

        # Update grid state and statistics
        self.grid = new_grid
        self.total_fire_count = np.sum(self.grid == RED)

    def suppress_fire(self, x, y):
        """Apply fire suppression at position (x,y)"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
            
        # Apply suppression to target and neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[nx, ny] == RED:
                        # Increase burning rate (delta-beta effect) to suppress faster
                        suppression_chance = self.beta * (1 + SUPPRESSION_EFFECT)
                        if random.random() < suppression_chance:
                            self.grid[nx, ny] = GRAY
                            self.burning_time[nx, ny] = 0
                        else:
                            # Reduce fuel faster when suppression is applied
                            self.fuel_remaining[nx, ny] *= 0.5
    
    def reset(self):
        """Reset the grid to initial state"""
        self.grid = np.zeros(self.grid_size, dtype=int)  # All cells start as GREEN
        self.fuel_remaining = np.ones(self.grid_size)    # Full fuel
        self.burning_time = np.zeros(self.grid_size)     # No burning time
        self.wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.init_fires()
        
    def get_state_counts(self):
        """Return counts of cells in each state"""
        green_count = np.sum(self.grid == GREEN)
        red_count = np.sum(self.grid == RED)
        gray_count = np.sum(self.grid == GRAY)
        return {
            'green': green_count,
            'red': red_count,
            'gray': gray_count
        }