import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import fiona
import matplotlib
import numpy as np
from pyproj import CRS, Transformer
import pandas as pd
import pyproj
import seaborn as sns
import matplotlib.ticker as ticker
from shapely.geometry import Point
import os
import logging
from shapely.geometry import Point, Polygon   

# # Create CRS objects
proj_wgs84 = pyproj.CRS('EPSG:4326')  # WGS84

# Function to get UTM zone dynamically
def get_utm_zone(longitude, latitude):
    zone_number = int((longitude + 180) // 6) + 1
    hemisphere = 'north' if latitude >= 0 else 'south'
    return zone_number, hemisphere

def create_utm_proj(zone_number, hemisphere):
    proj_string = f"+proj=utm +zone={zone_number} +{'north' if hemisphere == 'north' else 'south'} +datum=WGS84 +units=m +no_defs"
    return pyproj.CRS(proj_string)

# Function to transform UTM to WGS84
def transform_to_wgs84(utm_x, utm_y, utm_crs):
    #utm_crs = create_utm_proj(latitude, longitude)
    transformer = pyproj.Transformer.from_crs(utm_crs, proj_wgs84, always_xy=True)
    longitude, latitude = transformer.transform(utm_x, utm_y)
    return longitude, latitude

# Function to transform WGS84 to UTM
def transform_to_utm(longitude, latitude, utm_crs):
    #utm_crs = create_utm_proj(lon, lat)
    transformer = pyproj.Transformer.from_crs(proj_wgs84, utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(longitude, latitude)
    return utm_x, utm_y


def shp_to_grid(filename, gridsize):
    
    # Set SHAPE_RESTORE_SHX config option to YES
    fiona.drvsupport.supported_drivers['ESRI Shapefile'] = 'raw'
    with fiona.Env(SHAPE_RESTORE_SHX='YES'):
        gdf = gpd.read_file(filename)

    # Set the CRS to WGS84 if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)  # WGS84

    
    # Convert to a suitable projected CRS for accurate centroid calculation
    projected_gdf = gdf.to_crs(epsg=32633)  # Using UTM zone 33N (example)

    # Calculate the centroid in the projected CRS
    centroid_projected = projected_gdf.geometry.centroid.iloc[0]

    # Convert the centroid back to geographic CRS (WGS84) if needed
    centroid = gpd.GeoSeries([centroid_projected], crs=projected_gdf.crs).to_crs(epsg=4326).iloc[0]

    # Determine the UTM zone based on the centroid
    utm_zone, hemisphere = get_utm_zone(centroid.x, centroid.y)

    # Create UTM CRS
    utm_crs = create_utm_proj(utm_zone, hemisphere)
    gdf_utm = gdf.to_crs(utm_crs)

    # Get the boundary in UTM
    boundary_utm = gdf_utm.geometry.unary_union

    # Get boundary coordinates in UTM
    boundary_coords_utm = np.array(boundary_utm.exterior.coords)

    # Get the bounding box of the polygon in UTM
    minx, miny, maxx, maxy = boundary_utm.bounds

    # Define grid resolution in meters
    grid_size = gridsize  # Adjust as needed for desired grid size

    # Generate grid points in UTM
    x = np.arange(minx, maxx, grid_size)
    y = np.arange(miny, maxy, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    
    # Filter points inside the polygon
    inside_points = [Point(p).within(boundary_utm) for p in grid_points]
    grid_points = grid_points[inside_points]

    # Optionally, convert grid points back to WGS84
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    grid_points_wgs84 = np.array([transformer.transform(p[0], p[1]) for p in grid_points])
    
    
    # Plotting the grid and polygon
    plt.figure(figsize=(10, 10))
    plt.plot(boundary_coords_utm[:, 0], boundary_coords_utm[:, 1], 'r-', linewidth=2, label='Boundary')
    plt.scatter(grid_points[:, 0], grid_points[:, 1], s=1, c='blue', label='Grid Points')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Grid within Boundary (UTM)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/nicolaiaustad/Desktop/CropCounter2/XXX_last_grid_plot_utm.png')

    values_gps = np.zeros(len(grid_points_wgs84))
    #values_gps = np.full(len(grid_points), np.nan)
    df_gps = pd.DataFrame(grid_points_wgs84, columns=['x', 'y'])
    df_gps['values'] = values_gps
    df_gps["measured"] = np.zeros(len(grid_points_wgs84), dtype=bool)
    
    values_utm = np.zeros(len(grid_points))   #Change back to zeros
    #values_utm = np.full(len(grid_points), np.nan) #Initiliaze with nan in the newest update to make heatmap gray
    df_utm = pd.DataFrame(grid_points, columns=['x', 'y'])
    df_utm['values'] = values_utm
    df_utm["measured"] = np.zeros(len(grid_points), dtype=bool)
    
    return grid_points, grid_points_wgs84, df_utm, df_gps, boundary_coords_utm 


def find_grid_cell(longitude, latitude, grid_size, df):                  
    # Get the minimum values of x and y from the DataFrame
    min_x = df['x'].min()
    min_y = df['y'].min()
    
    # Adjust the coordinates based on the grid starting point
    adjusted_longitude = longitude - min_x
    adjusted_latitude = latitude - min_y
    
    # Round to the nearest grid point
    cell_x = np.floor(adjusted_longitude / grid_size) * grid_size + min_x
    cell_y = np.floor(adjusted_latitude / grid_size) * grid_size + min_y

    # Return the corresponding row in the DataFrame
    row = df[(df['x'] == cell_x) & (df['y'] == cell_y)]
    if not row.empty:
        return row.index[0]
    return None



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import generic_filter
from scipy.spatial import distance_matrix

def save_heatmap_to_shapefile(df_data, output_path, crs):
    geometry = [Point(xy) for xy in zip(df_data['x'], df_data['y'])]
    gdf = gpd.GeoDataFrame(df_data, crs=crs, geometry=geometry)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf_wgs84.to_file(output_path, driver='ESRI Shapefile')
    #print(f"Shapefile saved to {output_path}")
    logging.info(f"Shapefile saved to {output_path}")



import numpy as np
from scipy.ndimage import gaussian_filter

def custom_interpolation(pivot_table, sigma=1, min_value=0, max_value=500):
    grid_z = pivot_table.values
    
    # Step 1: Set up a custom function for interpolation
    def mean_if_measured(values):
        center_value = values[4]  # Center value in the 3x3 grid
        
        # If the center pixel is non-zero, keep it as is
        if center_value != 0:
            return center_value
        
        # Get direct neighbors (up, down, left, right)
        neighbors = values[[1, 7, 3, 5]]  # Get values from the positions [up, down, left, right]
        
        # Consider only non-zero neighbors
        non_zero_neighbors = neighbors[neighbors != 0]
        
        # If there are non-zero neighbors, return their mean
        if len(non_zero_neighbors) > 0:
            return np.mean(non_zero_neighbors)
        
        # If no non-zero neighbors, return 0
        return 0

    # Step 2: Apply the custom filter
    interpolated_grid = generic_filter(grid_z, mean_if_measured, size=3, mode='constant', cval=0)

    # Step 3: Clip to avoid negative values and apply smoothing
    interpolated_grid = np.clip(interpolated_grid, min_value, max_value)
    smoothed_grid = gaussian_filter(interpolated_grid, sigma=sigma)

    return smoothed_grid





def make_heatmap_and_save(df_data, grid_size, heatmap_output_path, shapefile_output_path, crs):
    # Create pivot table for 'values'
    pivot_table_values = df_data.pivot(index='y', columns='x', values='values')

    # Perform interpolation and smoothing
    smoothed_grid = custom_interpolation(pivot_table_values)

    # Extract the grid points for saving
    unique_x = pivot_table_values.columns.values
    unique_y = pivot_table_values.index.values
    
    df_data_interpolated = []
    for i in range(len(unique_y)):
        for j in range(len(unique_x)):
            if smoothed_grid[i, j] != 0:
                df_data_interpolated.append({
                    'x': unique_x[j],
                    'y': unique_y[i],
                    'values': smoothed_grid[i, j]
                })
    df_data_interpolated = pd.DataFrame(df_data_interpolated)
    
    # Save heatmap
    plt.figure(figsize=(12, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    # cmap.set_bad(color='gray')
    
    sns.heatmap(smoothed_grid, cmap=cmap, annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False)
    plt.gca().invert_yaxis()
    plt.title('Heatmap of Values')
    plt.xlabel('UTM X Coordinate')
    plt.ylabel('UTM Y Coordinate')
    plt.savefig(heatmap_output_path) #Think this saves to usb stick in combination with load settings
    
    save_directory = "/home/dataplicity/remote/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    # Ensure the heatmap_output_path does not start with /tmp/
    heatmap_output_path = heatmap_output_path.lstrip("/tmp/")
    plt.savefig("/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/"+heatmap_output_path)
     
    plt.savefig(f"{save_directory}"+heatmap_output_path)
    plt.close()
    logging.info(f"Heatmap saved to {heatmap_output_path}")

    # Save to shapefile
    save_heatmap_to_shapefile(df_data_interpolated, shapefile_output_path, crs)
    
 


def inverse_distance_weighting(x, y, values, xi, yi, power=2):
    # Break into smaller chunks if needed
    chunk_size = 10000  # Adjust this based on your memory
    interpolated_values = np.zeros(xi.shape[0])
    for start in range(0, xi.shape[0], chunk_size):
        end = min(start + chunk_size, xi.shape[0])
        dist = distance_matrix(np.c_[x, y], np.c_[xi[start:end], yi[start:end]])
        
        # Replace zero distances with a small value to avoid division by zero
        dist[dist == 0] = 1e-10
        
        weights = 1 / np.power(dist, power)
        weights /= weights.sum(axis=0)
        
        interpolated_values[start:end] = np.dot(weights.T, values)
    
    return interpolated_values
    

def generate_idw_heatmap(df_data, bound, grid_size, heatmap_output_path, shapefile_output_path, crs, sigma=1, power=2):

    x = df_data['x'].values
    y = df_data['y'].values
    values = df_data['values'].values
   
    # Create a grid of points for the heatmap
    xii = np.arange(x.min(), x.max(), 1)   #Adjust spacing to more than 1 m if necessary
    yii = np.arange(y.min(), y.max(), 1)
    xi, yi =np.meshgrid(xii, yii)
   
    # Flatten the grid to pass it to the IDW function
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    
    # Apply IDW interpolation with chunking
    interpolated_values = inverse_distance_weighting(x, y, values, xi_flat, yi_flat, power=power)
    
    # Create a Polygon from boundary_coords_utm
    boundary_polygon = Polygon(bound)
    
    # Reshape the interpolated values back to the grid shape
    grid_z = interpolated_values.reshape(xi.shape)
    
   
    grid_z = gaussian_filter(grid_z, sigma=sigma)
   
    mask = np.array([boundary_polygon.contains(Point(px, py)) for px, py in zip(xi_flat, yi_flat)])
    mask = mask.reshape(xi.shape)

    # Apply the mask to set values outside the boundary to NaN
    grid_z[~mask] = np.nan
    
    extent = [xii.min(), xii.max(), yii.min(), yii.max()]
    # Create and save the heatmap
    plt.figure(figsize=(12, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    # Use imshow to display the heatmap with the correct extent
    plt.imshow(grid_z, cmap=cmap, origin='lower', extent=extent, aspect='auto')
    
    plt.colorbar(label='IDW Interpolated Values')
    plt.title('IDW Heatmap of Values')
    plt.xlabel('UTM X Coordinate')
    plt.ylabel('UTM Y Coordinate')
    plt.tight_layout()
    

    # Save the heatmap
    plt.savefig(heatmap_output_path)
    save_directory = "/home/dataplicity/remote/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Ensure the heatmap_output_path does not start with /tmp/
    heatmap_output_path = heatmap_output_path.lstrip("/tmp/")
    plt.savefig("/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/"+heatmap_output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Heatmap saved to {heatmap_output_path}")
    
    # Save the interpolated data to a shapefile (optional)
    save_heatmap_to_shapefile(pd.DataFrame({'x': xi_flat, 'y': yi_flat, 'values': interpolated_values}), shapefile_output_path, crs)