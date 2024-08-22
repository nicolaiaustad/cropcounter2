#This script can be used to open newly generated shapefiles and read their data

import geopandas as gpd
import matplotlib.pyplot as plt

# Path to your shapefile
shapefile_path = "/home/nicolaiaustad/Desktop/generated_shape_files/SHAPE.shp"

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Print the first few rows of the GeoDataFrame
print("First few rows of the shapefile:")
print(gdf.head())

# Print the column names (attribute fields)
print("\nColumn names (attribute fields):")
print(gdf.columns)

# Print the geometry type
print("\nGeometry type:")
print(gdf.geom_type)

# Print the CRS (Coordinate Reference System)
print("\nCRS (Coordinate Reference System):")
print(gdf.crs)

# Print the total number of features
print("\nTotal number of features:")
print(len(gdf))

# Print summary statistics for the numerical columns
print("\nSummary statistics for numerical columns:")
print(gdf.describe())

# Print the bounds of the geometries
print("\nBounds of the geometries:")
print(gdf.total_bounds)

# Check for null values in the GeoDataFrame
print("\nNull values in each column:")
print(gdf.isnull().sum())

# If you want to visualize the geometries
gdf.plot()

plt.title("Shapefile Geometries")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig('/home/nicolaiaustad/Desktop/newly_generated_shapefile.png')
