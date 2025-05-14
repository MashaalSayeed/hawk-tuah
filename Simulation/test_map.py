import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import contextily as ctx
from pyproj import Transformer

def plot_gps_location_with_radius(latitude, longitude, radius_meters, zoom=15, map_style='OpenStreetMap'):
    """
    Plot a map centered on a GPS location with a circle of specified radius in meters.
    
    Parameters:
    -----------
    latitude : float
        The latitude of the center point in decimal degrees
    longitude : float
        The longitude of the center point in decimal degrees
    radius_meters : float
        The radius of the circle in meters
    zoom : int, optional
        The zoom level for the map (higher is more detailed)
    map_style : str, optional
        The map tile style to use, default is 'OpenStreetMap'
        
    Returns:
    --------
    fig, ax : tuple
        The matplotlib figure and axis objects
    """
    # Convert GPS coordinates to Web Mercator projection (which is used by most web maps)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_center, y_center = transformer.transform(longitude, latitude)
    
    # Calculate the approximate size to display based on radius
    # A larger buffer ensures the circle is fully visible
    buffer_factor = 1.5
    display_size = radius_meters * buffer_factor
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Add the circle representing the radius
    circle = Circle((x_center, y_center), radius_meters, 
                   fill=False, edgecolor='red', linewidth=2, alpha=0.7)
    ax.add_patch(circle)
    
    # Set the extent of the axis to show the area of interest
    x_min = x_center - display_size
    x_max = x_center + display_size
    y_min = y_center - display_size
    y_max = y_center + display_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add the base map
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Error loading map tiles: {e}")
        print("If you don't have internet connection or contextily installed, you'll see a blank map.")
    
    # Mark the center point
    ax.plot(x_center, y_center, 'bo', markersize=10, label='Center Point')
    
    # Add a title with the coordinates and radius information
    ax.set_title(f"Map centered at {latitude}, {longitude} with {radius_meters}m radius")
    
    # Remove axis values as they're not meaningful for this visualization
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a legend
    ax.legend()
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Example coordinates (San Francisco, CA)
    latitude = 26.861406
    longitude =  75.812826
    radius_meters = 100  # 500 meters radius
    
    fig, ax = plot_gps_location_with_radius(latitude, longitude, radius_meters, zoom=1)
    plt.show()