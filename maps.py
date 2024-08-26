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
    print("gdf_utm before generating utm")
    print(gdf_utm)

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
    
    print("Grid_points from shape file before filtering")
    print(grid_points.shape)
    # Filter points inside the polygon
    inside_points = [Point(p).within(boundary_utm) for p in grid_points]
    grid_points = grid_points[inside_points]

    print("Grid_points from shape file in the ebginneing")
    print(grid_points.shape)
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
    plt.savefig('/home/nicolaiaustad/Desktop/CropCounter2/XXX_last_grid_plot_utm222.png')

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
    print("DFUTM-head;  ")
    print(df_utm.head())
    print("DF_UTM before return")
    print(df_utm.shape)
    
    return grid_points, grid_points_wgs84, df_utm, df_gps, inside_points, boundary_coords_utm 


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
    



from shapely.geometry import Point, Polygon    
from shapely.ops import unary_union
import alphashape

# def inverse_distance_weighting(x, y, values, xi, yi, power=2):
#     # Break into smaller chunks if needed
#     chunk_size = 10000  # Adjust this based on your memory
#     interpolated_values = np.zeros(xi.shape[0])

#     for start in range(0, xi.shape[0], chunk_size):
#         end = min(start + chunk_size, xi.shape[0])
#         dist = distance_matrix(np.c_[x, y], np.c_[xi[start:end], yi[start:end]])
        
#         # Replace zero distances with a small value to avoid division by zero
#         dist[dist == 0] = 1e-10
        
#         weights = 1 / np.power(dist, power)
#         weights /= weights.sum(axis=0)
        
#         interpolated_values[start:end] = np.dot(weights.T, values)
    
#     return interpolated_values

# Step 3: Perform IDW interpolation
# def inverse_distance_weighting(x, y, values, xi, yi, power=2, chunk_size=5000):
#     interpolated_values = np.zeros(xi.shape[0])
    
#     print("Length of xi flat=interpolat values is: "+str(len(interpolated_values)))
#     for start in range(0, xi.shape[0], chunk_size):
    
#         end = min(start + chunk_size, xi.shape[0])
        
        
#         # Extract the current chunk of grid points
#         xi_chunk = xi[start:end]
#         yi_chunk = yi[start:end]
        
#         # Compute the distance matrix between known points and current chunk of grid points
#         dist = distance_matrix(np.c_[x, y], np.c_[xi_chunk, yi_chunk])
        
#         # Replace zero distances with a small value to avoid division by zero
#         dist[dist == 0] = 1e-10
        
#         # Compute the weights based on distance
#         weights = 1 / np.power(dist, power)
#         weights /= weights.sum(axis=0)
        
#         # Compute the weighted sum to get the interpolated values
#         interpolated_values[start:end] = np.dot(weights.T, values)
    
#     return interpolated_values


def inverse_distance_weighting(x, y, values, xi, yi, power=2):
    # Break into smaller chunks if needed
    chunk_size = 10000  # Adjust this based on your memory
    interpolated_values = np.zeros(xi.shape[0])
    print("interpolated_values")
    print(interpolated_values.shape)
    for start in range(0, xi.shape[0], chunk_size):
        end = min(start + chunk_size, xi.shape[0])
        dist = distance_matrix(np.c_[x, y], np.c_[xi[start:end], yi[start:end]])
        
        # Replace zero distances with a small value to avoid division by zero
        dist[dist == 0] = 1e-10
        
        weights = 1 / np.power(dist, power)
        weights /= weights.sum(axis=0)
        
        interpolated_values[start:end] = np.dot(weights.T, values)
    
    return interpolated_values


from shapely.geometry import Point, Polygon, MultiPolygon
# def generate_idw_heatmap_pivot(df_data, filename, grid_size, heatmap_output_path, shapefile_output_path, crs, sigma=1, power=2, chunk_size=5000):
#     # # Set SHAPE_RESTORE_SHX config option to YES
#     # fiona.drvsupport.supported_drivers['ESRI Shapefile'] = 'raw'
#     # with fiona.Env(SHAPE_RESTORE_SHX='YES'):
#     #     gdf = gpd.read_file(filename)

#     # # Set the CRS to WGS84 if not already set
#     # if gdf.crs is None:
#     #     gdf.set_crs(epsg=4326, inplace=True)  # WGS84

    
#     # # Convert to a suitable projected CRS for accurate centroid calculation
#     # projected_gdf = gdf.to_crs(epsg=32633)  # Using UTM zone 33N (example)

#     # # Calculate the centroid in the projected CRS
#     # centroid_projected = projected_gdf.geometry.centroid.iloc[0]

#     # # Convert the centroid back to geographic CRS (WGS84) if needed
#     # centroid = gpd.GeoSeries([centroid_projected], crs=projected_gdf.crs).to_crs(epsg=4326).iloc[0]

#     # # Determine the UTM zone based on the centroid
#     # utm_zone, hemisphere = get_utm_zone(centroid.x, centroid.y)

#     # # Create UTM CRS
#     # utm_crs = create_utm_proj(utm_zone, hemisphere)
#     # gdf_utm = gdf.to_crs(utm_crs)
   

#     # # Get the boundary in UTM
#     # boundary_utm = gdf_utm.geometry.unary_union

#     # # Get boundary coordinates in UTM
#     # boundary_coords_utm = np.array(boundary_utm.exterior.coords)

#     # # Get the bounding box of the polygon in UTM
#     # minx, miny, maxx, maxy = boundary_utm.bounds

#     # # Define grid resolution in meters
#     # grid_size = grid_size  # Adjust as needed for desired grid size

#     # # Generate grid points in UTM
#     # x = np.arange(minx, maxx, grid_size)
#     # y = np.arange(miny, maxy, grid_size)
#     # xx, yy = np.meshgrid(x, y)
#     # grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # # Filter points inside the polygon
#     # inside_points = [Point(p).within(boundary_utm) for p in grid_points]
#     # grid_points = grid_points[inside_points]

    
#     # # Plotting the grid and polygon
#     # plt.figure(figsize=(10, 10))
#     # plt.plot(boundary_coords_utm[:, 0], boundary_coords_utm[:, 1], 'r-', linewidth=2, label='Boundary')
#     # plt.scatter(grid_points[:, 0], grid_points[:, 1], s=1, c='blue', label='Grid Points')
#     # plt.xlabel('Easting (m)')
#     # plt.ylabel('Northing (m)')
#     # plt.title('Grid within Boundary (UTM)')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.savefig('/home/nicolaiaustad/Desktop/CropCounter2/Funker.png')

    
#     # # Extract x, y, and values
#     # x_temp = df_data['x'].values
#     # y_temp = df_data['y'].values
#     # values = df_data['values'].values
    
#     # # # Create a grid of points for the heatmap
#     # # xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 50),
#     # #                      np.linspace(y.min(), y.max(), 50))

#     # # Flatten the grid to pass it to the IDW function
#     # xi_flat = xx.ravel()
#     # yi_flat = yy.ravel()

#     # # Apply IDW interpolation
#     # interpolated_values = inverse_distance_weighting(x, y, values, xi_flat, yi_flat, chunk_size=chunk_size)

#     # # Reshape the interpolated values back to the grid shape
#     # grid_z = interpolated_values.reshape(xx.shape)

#     # # Apply Gaussian smoothing
#     # grid_z = gaussian_filter(grid_z, sigma=sigma)

#     # # Step 4: Mask out points outside the concave hull
#     # #mask = np.array([not alpha_shape.contains(Point(xi_point, yi_point)) for xi_point, yi_point in zip(xi_flat, yi_flat)])
#     # #mask = mask.reshape(xi.shape)
#     # #grid_z[mask] = np.nan  # Mask out points outside the boundary

#     # # Step 5: Visualize the concave hull and IDW heatmap
#     # fig, ax = plt.subplots(1, 2, figsize=(16, 8))



#     # # Plot the heatmap on the second subplot
#     # sns.heatmap(grid_z, cmap=sns.color_palette("RdYlGn", as_cmap=True), annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False, ax=ax[1])
#     # ax[1].invert_yaxis()
#     # ax[1].set_title('IDW Heatmap of Values')
#     # ax[1].set_xlabel('X Coordinate')
#     # ax[1].set_ylabel('Y Coordinate')

#     # # Save the plot
#     # plt.savefig(f"/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/{heatmap_output_path.lstrip('/tmp/')}")
    
#     # # Save the interpolated data to a shapefile (optional)
#     # save_heatmap_to_shapefile(grid_z, shapefile_output_path, crs)
    
#        # Read and process the shapefile
#     # Read and process the shapefile
#     # Set SHAPE_RESTORE_SHX config option to YES
#     fiona.drvsupport.supported_drivers['ESRI Shapefile'] = 'raw'
#     with fiona.Env(SHAPE_RESTORE_SHX='YES'):
#         gdf = gpd.read_file(filename)

#     if gdf.crs is None:
#         gdf.set_crs(epsg=4326, inplace=True)

#     projected_gdf = gdf.to_crs(epsg=32633)  # Assuming UTM zone 33N

#     # Get the boundary in UTM
#     boundary_utm = projected_gdf.geometry.unary_union

#     # Get boundary coordinates in UTM
#     if isinstance(boundary_utm, (Polygon, MultiPolygon)):
#         boundary_coords_utm = np.array(boundary_utm.exterior.coords)
#     else:
#         raise ValueError("Boundary is not a Polygon or MultiPolygon")

#     # # Get the bounding box of the polygon in UTM
#     # minx, miny, maxx, maxy = boundary_utm.bounds

#     # # Generate grid points in UTM
#     # x = np.arange(minx, maxx, grid_size)
#     # y = np.arange(miny, maxy, grid_size)
#     # xx, yy = np.meshgrid(x, y)
#     # grid_points = np.c_[xx.ravel(), yy.ravel()]
   
    
#     # xi_flat = xx.ravel()
#     # yi_flat = yy.ravel()
#     # # Filter points inside the polygon
#     # inside_points = [Point(p).within(boundary_utm) for p in grid_points]
#     # grid_points = grid_points[inside_points]

#     # print("Gridpoints form when we created a new gdf")
#     # print(grid_points)
#     # print(grid_points.shape)
    
#     # if len(grid_points) == 0:
#     #     raise ValueError("No grid points fall inside the polygon boundary.")

#     # #xxx = grid_points[:, 0]
#     # #yyy = grid_points[:, 1]

#     # Extract x, y, and values from df_data
#     x_temp = df_data['x'].values
#     y_temp = df_data['y'].values
#     values = df_data['values'].values
    
#     minx, maxx = x_temp.min(), x_temp.max()
#     miny, maxy = y_temp.min(), y_temp.max()
#     x = np.arange(minx, maxx, grid_size)
#     y = np.arange(miny, maxy, grid_size)
#     xx, yy = np.meshgrid(x, y)
#     grid_points= np.c_[xx.ravel(), yy.ravel()]
#     # Assign filtered grid points to xxx and yyy
#     xxx = grid_points[:, 0]
#     yyy = grid_points[:, 1]
#     print("Gridpoints old")
#     print(grid_points)
#     print(grid_points.shape)
#     print("xxx")
#     print(xxx)
#     print(xxx.shape)
#     print("length of dt_data X and Y is : "+str(len(x_temp))+" "+str(len(y_temp)))
#     # Apply IDW interpolation to the grid points
#     interpolated_values = inverse_distance_weighting(x_temp, y_temp, values, xxx, yyy, power=power)

#     # Apply Gaussian smoothing
#     smoothed_values = gaussian_filter(interpolated_values, sigma=sigma)

#     # # Reshape smoothed values to grid shape for heatmap
#     # grid_shape = (len(y), len(x))  # Shape should match the meshgrid shape
#     # smoothed_grid = np.full(grid_shape, np.nan)
#     # smoothed_grid[inside_points] = smoothed_values
    
#     # Create a GeoDataFrame for the interpolated values
#     interpolated_gdf = gpd.GeoDataFrame({
#         'geometry': [Point(x, y) for x, y in zip(xx, yy)],
#         'value': smoothed_values
#     }, crs=projected_gdf.crs)

#     print("About smoothed values: ")
#     print(smoothed_values.shape)
#     print(smoothed_values)
 
#     # Plot the heatmap based on the interpolated values
#     plt.figure(figsize=(10, 10))
#     plt.tricontourf(xx2, yy2, smoothed_values, levels=14, cmap=sns.color_palette("RdYlGn", as_cmap=True))
#     plt.plot(boundary_coords_utm[:, 0], boundary_coords_utm[:, 1], 'r-', linewidth=2, label='Boundary')
#     #sns.heatmap(smoothed_values, cmap=cmap, annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False)
#     plt.gca().invert_yaxis()
#     plt.title('Heatmap of Values')
#     plt.xlabel('UTM X Coordinate')
#     plt.ylabel('UTM Y Coordinate')
#     plt.savefig(f"/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/{heatmap_output_path.lstrip('/tmp/')}")
#     plt.close()

#     cmap = sns.color_palette("RdYlGn", as_cmap=True)
#     plt.figure(figsize=(10, 10))
    
#     sns.heatmap(smoothed_values, cmap=cmap, annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False)
#     plt.gca().invert_yaxis()
#     plt.title('Heatmap of Values')
#     plt.xlabel('UTM X Coordinate')
#     plt.ylabel('UTM Y Coordinate')
#     plt.savefig(f"/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/WOOOW.png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def generate_idw_heatmap(df_data, grid_size, heatmap_output_path, shapefile_output_path, crs, sigma=1, power=2):
#     # Extract the 'x', 'y', and 'values' directly from the DataFrame
#     x = df_data['x'].values
#     y = df_data['y'].values
#     values = df_data['values'].values
   
    
#     # Ensure that there are enough points to interpolate
#     if len(x) == 0 or len(y) == 0 or len(values) == 0:
#         logging.warning("No valid data points for interpolation. Skipping IDW interpolation.")
#         return
    
#       # Print out the dimensions and number of points
#     print(f"Number of x points: {len(np.unique(x))}")
#     print(f"Number of y points: {len(np.unique(y))}")
#     print(f"Total points in df_data: {len(x)}")
    
#     # Apply IDW interpolation directly on the data points
#     interpolated_values = inverse_distance_weighting(x, y, values, x, y, power=power)
    
#     # Check if the number of interpolated values matches the expected size
#     expected_size = len(np.unique(x)) * len(np.unique(y))
#     print(f"Expected size for reshaping: {expected_size}")
#     print(f"Actual size of interpolated_values: {interpolated_values.size}")
    
#     # Reshape the interpolated values to match the original data structure
#     try:
#         grid_z = interpolated_values.reshape(len(np.unique(y)), len(np.unique(x)))
#     except ValueError as e:
#         logging.error(f"Reshape error: {e}")
#         return
    
    
#     # Apply Gaussian smoothing
#     grid_z = gaussian_filter(grid_z, sigma=sigma)
    
#     # Create and save the heatmap
#     plt.figure(figsize=(12, 10))
    
#     cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
#     sns.heatmap(grid_z, cmap=cmap, annot=False, fmt="f", cbar=True, 
#                 xticklabels=False, yticklabels=False)
#     plt.gca().invert_yaxis()
#     plt.title('IDW Heatmap of Values')
#     plt.xlabel('UTM X Coordinate')
#     plt.ylabel('UTM Y Coordinate')
    
#     # Save the heatmap
#     plt.savefig(heatmap_output_path)
    
#     save_directory = "/home/dataplicity/remote/"
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
#     # Ensure the heatmap_output_path does not start with /tmp/
#     heatmap_output_path = heatmap_output_path.lstrip("/tmp/")
#     plt.savefig("/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/"+heatmap_output_path)
     
#     plt.savefig(f"{save_directory}"+heatmap_output_path)
#     plt.close()
#     logging.info(f"Heatmap saved to {heatmap_output_path}")
    
#     # Save the interpolated data to a shapefile (optional)
#     save_heatmap_to_shapefile(grid_z, shapefile_output_path, crs)


def generate_idw_heatmap(df_data, inside_points, bound, grid_size, heatmap_output_path, shapefile_output_path, crs, sigma=1, power=2):
    print("Length of x values in df_data")
    print(len(df_data['x'].values))
    print("Length of y values in df_data")
    print(len(df_data['y'].values))
    
    # fiona.drvsupport.supported_drivers['ESRI Shapefile'] = 'raw'
    # with fiona.Env(SHAPE_RESTORE_SHX='YES'):
    #     gdf = gpd.read_file(filename)

    # if gdf.crs is None:
    #     gdf.set_crs(epsg=4326, inplace=True)

    # projected_gdf = gdf.to_crs(epsg=32633)  # Assuming UTM zone 33N
    # centroid_projected = projected_gdf.geometry.centroid.iloc[0]
    # centroid = gpd.GeoSeries([centroid_projected], crs=projected_gdf.crs).to_crs(epsg=4326).iloc[0]

    # # Determine the UTM zone based on the centroid
    # utm_zone, hemisphere = get_utm_zone(centroid.x, centroid.y)

    # # Create UTM CRS
    # utm_crs = create_utm_proj(utm_zone, hemisphere)
    # gdf_utm = gdf.to_crs(utm_crs)
    # # Get the boundary in UTM
    # boundary_utm = gdf_utm.geometry.unary_union

    # # Get boundary coordinates in UTM
    # if isinstance(boundary_utm, (Polygon, MultiPolygon)):
    #     boundary_coords_utm = np.array(boundary_utm.exterior.coords)
    # else:
    #     raise ValueError("Boundary is not a Polygon or MultiPolygon")

    # # Get the bounding box of the polygon in UTM
    # minx, miny, maxx, maxy = boundary_utm.bounds

    # # Generate grid points in UTM
    # x1 = np.arange(minx, maxx, grid_size)
    # y1 = np.arange(miny, maxy, grid_size)
    # xx, yy = np.meshgrid(x1, y1)
    # print("xx length")
    # print(xx.shape)
    # grid_points = np.c_[xx.ravel(), yy.ravel()]
   
    # print("Gridpoints from new gdf to form boundary")
    #print(grid_points.shape)

    # xi_flat = xx.ravel()
    # yi_flat = yy.ravel()
    # Filter points inside the polygon
    # outside_points = [Point(p).within(boundary_utm) for p in grid_points]
    #grid_points = grid_points[inside_points]
    # Filter points outside the polygon
    
    # inside_points = np.array([Point(p).within(boundary_utm) for p in grid_points])
    # print("Inside points")
    # print(inside_points.shape)
    # # Reshape the mask to match the original grid shape
    # mask = inside_points.reshape(xx.shape)
    # print("Mask after reshape: ")
    # print(mask.shape)
    
    
    # Extract the 'x', 'y', and 'values' directly from the DataFrame
    x = df_data['x'].values
    y = df_data['y'].values
    values = df_data['values'].values
    print("df_Data head;  ")
    print(df_data.head())
    # # Ensure that there are enough points to interpolate
    # if len(x) == 0 or len(y) == 0 or len(values) == 0:
    #     logging.warning("No valid data points for interpolation. Skipping IDW interpolation.")
    #     return
    
    # Create a grid of points for the heatmap
    xii = np.arange(x.min(), x.max(), grid_size)
    yii = np.arange(y.min(), y.max(), grid_size)
    xi, yi =np.meshgrid(xii, yii)
    
    print("x from df_data and xii")
    print(len(x))
    print(len(xii))
    
    # if xi.size == 0 or yi.size == 0:
    #     logging.warning("Grid size is too small. Skipping IDW interpolation.")
    #     return
    
    # Flatten the grid to pass it to the IDW function
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    
    # Apply IDW interpolation with chunking
    interpolated_values = inverse_distance_weighting(x, y, values, xi_flat, yi_flat, power=power)
    
    
    # Create a Polygon from boundary_coords_utm
    boundary_polygon = Polygon(bound)
    
    # Reshape the interpolated values back to the grid shape
    grid_z = interpolated_values.reshape(xi.shape)
    
    
    print("GridZ before masking")
    print(grid_z.shape)
    # Apply Gaussian smoothing
    grid_z = gaussian_filter(grid_z, sigma=sigma)
    # Create the mask to set values outside the boundary to NaN
    # Create the mask to set values outside the grid_points to NaN
    # Create the mask by checking if each point is inside the boundary polygon
    mask = np.array([boundary_polygon.contains(Point(px, py)) for px, py in zip(xi_flat, yi_flat)])
    mask = mask.reshape(xi.shape)

    # Apply the mask to set values outside the boundary to NaN
    grid_z[~mask] = np.nan
    
    # mask = np.ones_like(grid_z, dtype=bool)  # Start with all True (masked out)
    
    # # Find the index of grid_points within the xi and yi grids
    # for p in grid_points:
    #     # Find the closest index in xi and yi grids
    #     idx_x = np.argmin(np.abs(xii - p[0]))
    #     idx_y = np.argmin(np.abs(yii - p[1]))
    #     mask[idx_y, idx_x] = False  # Unmask the valid points inside the boundary

    # grid_z[mask] = np.nan  # Apply the mask to the interpolated values
    
    print("xi range:", xii.min(), xii.max())
    print("yi range:", yii.min(), yii.max())
    print("grid_z shape:", grid_z.shape)
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
    # Plot the boundary on top of the heatmap
    #plt.plot(bound[:, 0], bound[:, 1], 'g-', linewidth=0.5, label='Boundary')

   
    
    # sns.heatmap(grid_z, cmap=cmap, annot=False, fmt="f", cbar=True, 
    #             xticklabels=False, yticklabels=False, mask=np.isnan(grid_z), extent=extent)
    # plt.gca().invert_yaxis()
    # # Setting plot limits to match the data extent
    
    # plt.title('IDW Heatmap of Values')
    # plt.xlabel('UTM X Coordinate')
    # plt.ylabel('UTM Y Coordinate')
    # Setting plot limits to match the data extent
    

    # Save the heatmap
    plt.savefig(heatmap_output_path)
    save_directory = "/home/dataplicity/remote/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Ensure the heatmap_output_path does not start with /tmp/
    heatmap_output_path = heatmap_output_path.lstrip("/tmp/")
    plt.savefig("/home/nicolaiaustad/Desktop/CropCounter2/generated_heatmaps/25"+heatmap_output_path, bbox_inches='tight')
     
#     plt.savefig(f"{save_directory}"+heatmap_output_path)
    plt.close()
    logging.info(f"Heatmap saved to {heatmap_output_path}")
    
    # Save the interpolated data to a shapefile (optional)
    save_heatmap_to_shapefile(pd.DataFrame({'x': xi_flat, 'y': yi_flat, 'values': interpolated_values}), shapefile_output_path, crs)