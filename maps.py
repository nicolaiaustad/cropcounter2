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
    plt.savefig('/home/nicolaiaustad/Desktop/CropCounter/XXX_last_grid_plot_utm.png')

    #values_gps = np.zeros(len(grid_points_wgs84))
    values_gps = np.full(len(grid_points), np.nan)
    df_gps = pd.DataFrame(grid_points_wgs84, columns=['x', 'y'])
    df_gps['values'] = values_gps
    df_gps["measured"] = np.zeros(len(grid_points_wgs84), dtype=bool)
    
    #values_utm = np.zeros(len(grid_points))
    values_utm = np.full(len(grid_points), np.nan) #Initiliaze with nan in the newest update to make heatmap gray
    df_utm = pd.DataFrame(grid_points, columns=['x', 'y'])
    df_utm['values'] = values_utm
    df_utm["measured"] = np.zeros(len(grid_points), dtype=bool)
    
    return grid_points, grid_points_wgs84, df_utm, df_gps


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

# def save_heatmap_to_shapefile(df_data, grid_size, output_path, crs):
#     geometry = [Point(xy) for xy in zip(df_data['x'], df_data['y'])]
#     gdf = gpd.GeoDataFrame(df_data, crs=crs, geometry=geometry)
#     gdf_wgs84 = gdf.to_crs(epsg=4326)
#     gdf_wgs84.to_file(output_path, driver='ESRI Shapefile')
#     print(f"Shapefile saved to {output_path}")

# def make_heatmap_and_save(df_data, grid_size, heatmap_output_path, shapefile_output_path, crs):
#     pivot_table = df_data.pivot(index='y', columns='x', values='values')
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(pivot_table, cmap='RdYlGn', annot=False, fmt="f", cbar=True)
#     xticks = np.arange(df_data['x'].min(), df_data['x'].max() + grid_size, grid_size)
#     yticks = np.arange(df_data['y'].min(), df_data['y'].max() + grid_size, grid_size)
#     plt.gca().set_xticks(np.arange(len(xticks)))
#     plt.gca().set_yticks(np.arange(len(yticks)))
#     plt.gca().set_xticklabels([f'{int(x)}' for x in xticks], rotation=45, ha='right')
#     plt.gca().set_yticklabels([f'{int(y)}' for y in yticks])
#     plt.gca().invert_yaxis()
#     plt.title('Heatmap of Values')
#     plt.xlabel('UTM X Coordinate')
#     plt.ylabel('UTM Y Coordinate')
#     plt.savefig(heatmap_output_path)
#     plt.savefig("/home/nicolaiaustad/Desktop/heatmap16jul.png")
#     plt.close()
#     print(f"Heatmap saved to {heatmap_output_path}")
#     save_heatmap_to_shapefile(df_data, grid_size, shapefile_output_path, crs)

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import geopandas as gpd
# from shapely.geometry import Point
# from scipy.interpolate import griddata
# from scipy.ndimage import gaussian_filter

# def save_heatmap_to_shapefile(df_data, grid_size, output_path, crs):
#     geometry = [Point(xy) for xy in zip(df_data['x'], df_data['y'])]
#     gdf = gpd.GeoDataFrame(df_data, crs=crs, geometry=geometry)
#     gdf_wgs84 = gdf.to_crs(epsg=4326)
#     gdf_wgs84.to_file(output_path, driver='ESRI Shapefile')
#     print(f"Shapefile saved to {output_path}")

# def interpolate_and_smooth(df_data, grid_size):
#     # Extract x, y, and values from the dataframe
#     x = df_data['x'].values
#     y = df_data['y'].values[::-1]
#     values = df_data['values'].values
    
#     # Create grid
#     grid_x, grid_y = np.mgrid[x.min():x.max():grid_size*1j, y.min():y.max():grid_size*1j]
    
#     # Interpolate missing values
#     grid_z = griddata((x, y), values, (grid_x, grid_y), method='cubic')
    
#     # Handle outliers by replacing them with the median or a capped value
#     z_median = np.nanmedian(grid_z)
#     z_std = np.nanstd(grid_z)
#     outlier_threshold = 3 * z_std
#     grid_z = np.where(np.abs(grid_z - z_median) > outlier_threshold, z_median, grid_z)
    
#     # Apply Gaussian filter for smoothing
#     grid_z = gaussian_filter(grid_z, sigma=1)
    
#     return grid_x, grid_y, grid_z

# def make_heatmap_and_save(df_data, grid_size, heatmap_output_path, shapefile_output_path, crs):
#     # Interpolate and smooth the data
#     grid_x, grid_y, grid_z = interpolate_and_smooth(df_data, grid_size)
    
#     # Prepare data for saving to shapefile
#     df_data_interpolated = []
#     for i in range(grid_x.shape[0]):
#         for j in range(grid_x.shape[1]):
#             df_data_interpolated.append({
#                 'x': grid_x[i, j],
#                 'y': grid_y[i, j],
#                 'values': grid_z[i, j]
#             })
#     df_data_interpolated = pd.DataFrame(df_data_interpolated)
    
#     # Create and save the heatmap
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(grid_z.T, cmap='RdYlGn', annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False)
#     plt.title('Heatmap of Values')
#     plt.xlabel('UTM X Coordinate')
#     plt.ylabel('UTM Y Coordinate')
#     plt.savefig(heatmap_output_path)
#     plt.savefig("/home/nicolaiaustad/Desktop/heatmap16jul.png")
#     plt.close()
#     print(f"Heatmap saved to {heatmap_output_path}")
    
#     # Save to shapefile
#     save_heatmap_to_shapefile(df_data_interpolated, grid_size, shapefile_output_path, crs)

# # Example usage:
# # df_data should be a DataFrame with columns: 'x', 'y', 'values'
# # make_heatmap_and_save(df_data, grid_size=0.1, heatmap_output_path='heatmap.png', shapefile_output_path='heatmap.shp', crs='epsg:32633')



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import generic_filter

def save_heatmap_to_shapefile(df_data, output_path, crs):
    geometry = [Point(xy) for xy in zip(df_data['x'], df_data['y'])]
    gdf = gpd.GeoDataFrame(df_data, crs=crs, geometry=geometry)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf_wgs84.to_file(output_path, driver='ESRI Shapefile')
    #print(f"Shapefile saved to {output_path}")
    logging.info(f"Shapefile saved to {output_path}")

def interpolate_and_smooth(pivot_table, method='cubic', sigma=1, min_value=0, max_value=500):
    # Interpolate missing values in the pivot table
    grid_z = pivot_table.values
    mask = np.isnan(grid_z)
    
    # Use the chosen method for interpolation
    grid_z = np.where(mask, np.nanmedian(grid_z), grid_z)
    
    # Handle outliers by clipping values to the desired range
    grid_z = np.clip(grid_z, min_value, max_value)
    
    # Apply Gaussian filter for smoothing
    #grid_z = gaussian_filter(grid_z, sigma=sigma)  #Removed gaussian blurring
    
    # Restore the mask to keep original boundaries
    grid_z[mask] = np.nan
    
    
    return grid_z

# def enclosed_nan_interpolation(pivot_table):
#     grid_z = pivot_table.values
#     mask = np.isnan(grid_z)
    
#     def nanmean_enclosed(values):
#         center_value = values[len(values) // 2]
        
#         # If the center value is not NaN, return it as is
#         if not np.isnan(center_value):
#             return center_value
        
#         # Check if all surrounding values are non-NaN
#         surrounding_values = np.array(values)
#         surrounding_values = np.delete(surrounding_values, len(values) // 2)  # Remove center value
#         if np.all(~np.isnan(surrounding_values)):
#             return np.nanmean(surrounding_values)
#         else:
#             return np.nan  # Keep as NaN if not fully enclosed

#     # Apply the nanmean_enclosed function using a 3x3 kernel
#     interpolated_grid = generic_filter(grid_z, nanmean_enclosed, size=3, mode='constant', cval=np.nan)
    
#     return interpolated_grid

# # Create a test DataFrame with X, Y, and Z columns
# test_df = pd.DataFrame({
#     'X': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
#     'Y': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#     'Z': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 15, np.nan, 12.0, np.nan]
# })

# # Pivot the DataFrame to create a 2D grid
# pivot_table = test_df.pivot(index='Y', columns='X', values='Z')

# # Apply the enclosed_nan_interpolation function
# interpolated_result = enclosed_nan_interpolation(pivot_table)
# interpolated_df = pd.DataFrame(interpolated_result, columns=pivot_table.columns, index=pivot_table.index)

# # Print the original and interpolated DataFrames
# print("Original DataFrame:")
# print(pivot_table)

# print("\nInterpolated DataFrame:")
# print(interpolated_df)

def enclosed_nan_interpolation(pivot_table):
    grid_z = pivot_table.values
    
    def nanmean_enclosed(values):
        center_value = values[4]  # The center value in a 3x3 grid (index 4 in a flattened array)
        
        # If the center value is not NaN, return it as is
        if not np.isnan(center_value):
            return center_value
        
        # Check if the direct neighbors (up, down, left, right) are non-NaN
        up = values[1]
        down = values[7]
        left = values[3]
        right = values[5]
        
        # If all direct neighbors are non-NaN, return their mean
        if not np.isnan(up) and not np.isnan(down) and not np.isnan(left) and not np.isnan(right):
            return np.nanmean([up, down, left, right])
        else:
            return np.nan  # Keep as NaN if not fully enclosed by these neighbors

    # Apply the nanmean_enclosed function using a 3x3 kernel
    interpolated_grid = generic_filter(grid_z, nanmean_enclosed, size=3, mode='constant', cval=np.nan)
    
    return interpolated_grid

# # Create a test DataFrame with X, Y, and Z columns
# test_df = pd.DataFrame({
#     'X': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
#     'Y': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#     'Z': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, np.nan, np.nan, 12.0, 15]
# })

# # Pivot the DataFrame to create a 2D grid
# pivot_table = test_df.pivot(index='Y', columns='X', values='Z')

# # Apply the enclosed_nan_interpolation function
# interpolated_result = enclosed_nan_interpolation(pivot_table)
# interpolated_df = pd.DataFrame(interpolated_result, columns=pivot_table.columns, index=pivot_table.index)




def make_heatmap_and_save(df_data, grid_size, heatmap_output_path, shapefile_output_path, crs):
    # Create pivot table
    pivot_table = df_data.pivot(index='y', columns='x', values='values')
    
    # Interpolate and smooth the data
    #grid_z = interpolate_and_smooth(pivot_table)
    grid_z = enclosed_nan_interpolation(pivot_table)  #Updated interpolation to my new function. Unsure if it works, could mess up heatmap generation
    
    # Extract the grid points
    unique_x = pivot_table.columns.values
    unique_y = pivot_table.index.values
    
    # Prepare data for saving to shapefile
    df_data_interpolated = []
    for i in range(len(unique_x)):
        for j in range(len(unique_y)):
            if not np.isnan(grid_z[j, i]):
                df_data_interpolated.append({
                    'x': unique_x[i],
                    'y': unique_y[j],
                    'values': grid_z[j, i]  # Note the order: grid_z is transposed
                })
    df_data_interpolated = pd.DataFrame(df_data_interpolated)
    
    save_directory = "/home/dataplicity/remote/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    # Create and save the heatmap
    plt.figure(figsize=(12, 10))
    
    # Use a colormap with white for NaN values
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color='gray')  # Set color for NaNs to grey. This is an update. Hope it works
    
    sns.heatmap(grid_z, cmap=cmap, annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False, mask=np.isnan(grid_z))
    plt.gca().invert_yaxis()
    plt.title('Heatmap of Values')
    plt.xlabel('UTM X Coordinate')
    plt.ylabel('UTM Y Coordinate')
    plt.savefig(heatmap_output_path) #Save to usb stick
    # Ensure the heatmap_output_path does not start with /tmp/
    heatmap_output_path = heatmap_output_path.lstrip("/tmp/")
    plt.savefig("/home/nicolaiaustad/Desktop/CropCounter/generated_heatmaps/"+heatmap_output_path)
     
    plt.savefig(f"{save_directory}"+heatmap_output_path)
    plt.close()
    logging.info(f"Heatmap saved to {heatmap_output_path}")
    #print(f"Heatmap saved to {heatmap_output_path}")

    # Save to shapefile
    save_heatmap_to_shapefile(df_data_interpolated, shapefile_output_path, crs)

# Example usage:
# df_data should be a DataFrame with columns: 'x', 'y', 'values'
# make_heatmap_and_save(df_data, grid_size=0.1, heatmap_output_path='heatmap.png', shapefile_output_path='heatmap.shp', crs='epsg:32633')
