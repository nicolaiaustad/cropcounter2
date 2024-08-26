# import io
# from picamera2 import Picamera2
# from PIL import Image
# from ultralytics import YOLO
# import time
# from multiprocessing import Pool
from ultralytics import YOLO

# model = YOLO("best.pt")
# model.export(format="onnx", dynamic=True, int8=True)
# onnx_model = YOLO("best.onnx")
# results = onnx_model.predict("logged_images/havre_image_20240814_135240_216_lat63.844473_lon11.383946.jpeg", imgsz=2816)
# results[0].save("output_with_boxes.jpg")

# model = YOLO('best.pt')



# model.export(format="ncnn", half=True, imgsz=2464)
# new_ncnn_model = YOLO("best_ncnn_model", task="detect")
# results = new_ncnn_model.predict("logged_images/01Move_G3_E500_S4_hvete_20240805_174334.png", imgsz=2464, conf=0.5)
# results[0].save("output_with_boxes_ncnn.jpg")

# new_ncnn_model = YOLO("best.pt", task="detect")
# results = new_ncnn_model.predict("logged_images/01Move_G3_E500_S4_hvete_20240805_174334.png", imgsz=2464, conf=0.5)
# results[0].save("output_with_boxes_best.jpg")




# # Load the YOLO model
# model = YOLO('/home/nicolaiaustad/Desktop/CropCounter2/model/best.pt')

# def capture_image_in_memory(picam2):
#     """Capture an image from the camera and store it in memory."""
#     image_stream = io.BytesIO()
#     picam2.capture_file(image_stream, format='png')
#     image_stream.seek(0)  # Rewind the stream to the beginning
#     image = Image.open(image_stream).convert("RGB")
#     return image

# def split_image(image, num_patches):
#     """Split the image into a specified number of patches."""
#     width, height = image.size
#     patch_width = width // 2
#     patch_height = height // 2
    
#     patches = []
#     for i in range(2):
#         for j in range(2):
#             box = (i * patch_width, j * patch_height, (i + 1) * patch_width, (j + 1) * patch_height)
#             patches.append(image.crop(box))
    
#     return patches

# def run_inference_on_patch(patch):
#     """Run inference on a single patch."""
#     results = model.predict(patch, imgsz= 1232)
#     return results

# def process_image(image):
#     """Split an image into patches and run inference on each patch using multiprocessing."""
#     patches = split_image(image, num_patches=4)

#     with Pool(processes=4) as pool:
#         results = pool.map(run_inference_on_patch, patches)

#     return results

# if __name__ == "__main__":
#     # Initialize and start the camera
#     picam2 = Picamera2()
#     camera_config = picam2.create_still_configuration(
#         main={
#             "size": (2464, 2464),  # Adjust size if needed
#             "format": "RGB888"
#         }
#     )
#     picam2.configure(camera_config)
#     picam2.start()

#     # Allow camera to warm up
#     time.sleep(2)

#     try:
#         while True:
#             # Capture image in memory
#             image = capture_image_in_memory(picam2)

#             # Process the image by splitting it into patches and running inference
#             results = process_image(image)

#             # Print the results for each patch
#             for i, result in enumerate(results):
#                 print(f"Patch {i + 1}: {len(result[0].boxes)} boxes detected")

#             # Wait before capturing the next image
#             time.sleep(2)  # Adjust timing as needed
#     finally:
#         picam2.stop()  # Stop the camera when done

# import os
# import time
# from PIL import Image
# from ultralytics import YOLO
# from multiprocessing import Pool, cpu_count
# import logging

# ncnn_model = YOLO("best_ncnn_model")

# # Directory containing images
# image_dir = '/home/nicolaiaustad/Desktop/CropCounter2/running_images'

# # Filter and get all image files in the directory
# image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# def run_single_inference(ncnn_model, image_files):
#     """Run inference on images one by one and measure time."""
#     start_time = time.time()
    
#     for image_file in image_files:
#         image = Image.open(image_file).convert("RGB")
#         results = ncnn_model.predict(image, imgsz=2816)
#         print(f"Processed {image_file}: {len(results[0].boxes)} boxes detected")
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     return total_time

# def process_image_batch(ncnn_model, batch_files):
#     """Process a batch of images and return the results."""
#     results = []
#     for image_file in batch_files:
#         image = Image.open(image_file).convert("RGB")
#         result = ncnn_model.predict(image, imgsz=2816)
#         results.append((image_file, result))
#     return results

# def run_batch_inference(ncnn_model, image_files, batch_size=6):
#     """Run inference on images in smaller batches and measure time."""
#     start_time = time.time()
#     pool = Pool(processes=cpu_count())
#     results = []
    
#     for i in range(0, len(image_files), batch_size):
#         batch_files = image_files[i:i + batch_size]
#         result = pool.apply_async(process_image_batch, (ncnn_model, batch_files))
#         results.append(result)
    
#     pool.close()
#     pool.join()

#     for result in results:
#         batch_results = result.get()
#         for image_file, res in batch_results:
#             print(f"Processed {image_file}: {len(res[0].boxes)} boxes detected")
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     return total_time

# if __name__ == "__main__":
#     # Ensure there are images to process
#     if not image_files:
#         print("No images found in the directory. Please add images to the folder and try again.")
#     else:
#         # Run batch inference timing
#         batch_inference_time = run_batch_inference(ncnn_model, image_files)
#         print(f"Total time for batch inference: {batch_inference_time:.2f} seconds")
        
#         # Run single inference timing
#         single_inference_time = run_single_inference(ncnn_model, image_files)
#         print(f"Total time for single image inference: {single_inference_time:.2f} seconds")
        
#         # Compare the results
#         print("\nComparison:")
#         print(f"Single image inference took {single_inference_time:.2f} seconds")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
import alphashape
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter

# # Step 1: Create a test DataFrame with points forming an irregular shape and add some "values"
# df = pd.DataFrame({
#     'x': [0, 2, 4, 1, 5, 3, 6, 3.5, 2, 5],
#     'y': [0, 1, 0, 3, 3, 6, 5, 2.5, 4, 4],
#     'values': [10, 20, 15, 25, 5, 30, 35, 40, 45, 50],
#     'measured': [True, True, True, True, True, True, True, True, True, True],  # All points are measured
#     'grp_name':['set1','set1','set1','set1','set1','set1','set1','set1','set1','set1']})



# # df = pd.DataFrame({'name':['a1','a2','a3','a4','a5','a6'],
# #                    'loc_x':[0,1,2,3,4,5],
# #                    'loc_y':[1,2,3,4,5,6],
# #                    'grp_name':['set1','set1','set1','set2','set2','set2']})

# df['points'] = gpd.points_from_xy(df.x, df.y)

# df = df.groupby('grp_name').agg(
#      name     = pd.NamedAgg(column='name',   aggfunc = lambda x: '|'.join(x)),
#      geometry = pd.NamedAgg(column='points', aggfunc = lambda x: Polygon(x.values))
#     ).reset_index()

# geodf = gpd.GeoDataFrame(df, geometry='geometry')

# # Step 2: Create a Concave Hull (Alpha Shape) from the points
# points = gpd.GeoSeries([Point(x, y) for x, y in zip(df['x'], df['y'])])
# alpha_shape = alphashape.alphashape(points, alpha=0.5)  # Adjust alpha as needed
# alpha_shape = alpha_shape.simplify(0.1, preserve_topology=True)
# # Step 3: Get the boundary of the alpha shape
# boundary = gpd.GeoSeries([alpha_shape]).boundary
# # Plot the boundary
# fig, ax = plt.subplots()
# boundary.plot(ax=ax, color='blue', linewidth=2)
# ax.set_title('Outer Boundary of Polygons')
# plt.savefig("UOter bound1.jpeg")

# Generate 1000 unique x and y values
np.random.seed(42)
x_unique = np.sort(np.random.choice(range(0, 2000), 1000, replace=False))
y_unique = np.sort(np.random.choice(range(0, 2000), 1000, replace=False))



# # Create an irregular shape by picking random points to duplicate
# num_duplicates = 500
# x_duplicates = np.random.choice(x_unique, num_duplicates, replace=True)
# y_duplicates = np.random.choice(y_unique, num_duplicates, replace=True)

# # Concatenate the unique and duplicate points
# x_all = np.concatenate([x_unique, x_duplicates])
# y_all = np.concatenate([y_unique, y_duplicates])



# # Assign random values to these points
# values = np.random.randint(1, 101, size=len(x_unique))

# # Assign measured status (for simplicity, all are measured)
# measured = np.array([True] * len(x_unique))

# # Create the DataFrame
# df_test = pd.DataFrame({
#     'x': x_unique,
#     'y': y_unique,
#     'values': values,
#     'measured': measured
# })


# df_test = pd.DataFrame({
#     'x': [0, 2, 4, 1, 5, 3, 6, 3.5, 2, 5],
#     'y': [0, 1, 0, 3, 3, 6, 5, 2.5, 4, 4],
#     'values': [10, 20, 15, 25, 5, 30, 35, 40, 45, 50],
#     'measured': [True, True, True, True, True, True, True, True, True, True] # All points are measured
#     })


# Set the parameters for the grid
x_min, x_max = 0, 100
y_min, y_max = 0, 100
spacing = 1  # Equal spacing between points

# Generate the grid points
x_values = np.arange(x_min, x_max + spacing, spacing)
y_values = np.arange(y_min, y_max + spacing, spacing)
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Flatten the grid to create a list of x, y pairs
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Generate random 'values' and set all 'measured' to True
values = np.ones(len(x_flat))  # Random values between 5 and 50
measured = np.full(x_flat.shape, True)  # All points are measured

# Create the expanded dataframe
df_test = pd.DataFrame({
    'x': x_flat,
    'y': y_flat,
    'values': values,
    'measured': measured
})

# Convert the grid into a DataFrame for easier manipulation
df_grid = df_test.pivot(index='y', columns='x', values='values')

# Shift some entire rows
np.random.seed(0)  # For reproducibility
shifted_rows = np.random.choice(df_grid.index, size=50, replace=False)  # Randomly pick 3 rows to shift
for row in shifted_rows:
    shift_value = np.random.uniform(-50, 50)
    df_test.loc[df_test['y'] == row, 'x'] += shift_value

# Shift some entire columns
shifted_columns = np.random.choice(df_grid.columns, size=50, replace=False)  # Randomly pick 3 columns to shift
for col in shifted_columns:
    shift_value = np.random.uniform(-50, 50)
    df_test.loc[df_test['x'] == col, 'y'] += shift_value

print(df_test['x'].unique)
print(df_test.shape)

#df_test = pd.DataFrame(data)

#print(df_test.shape)
# Step 2: Create a Concave Hull (Alpha Shape) from the points
points = gpd.GeoSeries([Point(x, y) for x, y in zip(df_test['x'], df_test['y'])])
alpha_shape = alphashape.alphashape(points, alpha=0.5)  # Adjust alpha as needed
alpha_shape = alpha_shape.simplify(0.1, preserve_topology=True)
# Step 3: Get the boundary of the alpha shape
boundary = gpd.GeoSeries([alpha_shape]).boundary

# Step 3: Perform IDW interpolation
def inverse_distance_weighting(x, y, values, xi, yi, power=2, chunk_size=10000):
    interpolated_values = np.zeros(xi.shape[0])
    
    for start in range(0, xi.shape[0], chunk_size):
        end = min(start + chunk_size, xi.shape[0])
        
        # Extract the current chunk of grid points
        xi_chunk = xi[start:end]
        yi_chunk = yi[start:end]
        
        # Compute the distance matrix between known points and current chunk of grid points
        dist = distance_matrix(np.c_[x, y], np.c_[xi_chunk, yi_chunk])
        
        # Replace zero distances with a small value to avoid division by zero
        dist[dist == 0] = 1e-10
        
        # Compute the weights based on distance
        weights = 1 / np.power(dist, power)
        weights /= weights.sum(axis=0)
        
        # Compute the weighted sum to get the interpolated values
        interpolated_values[start:end] = np.dot(weights.T, values)
    
    return interpolated_values

# Extract the 'x', 'y', and 'values' directly from the DataFrame
x = df_test['x'].values
y = df_test['y'].values
values = df_test['values'].values

# Create a grid of points for the heatmap
xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                     np.linspace(y.min(), y.max(), 100))

# Flatten the grid to pass it to the IDW function
xi_flat = xi.ravel()
yi_flat = yi.ravel()

# Apply IDW interpolation
interpolated_values = inverse_distance_weighting(x, y, values, xi_flat, yi_flat)

# Reshape the interpolated values back to the grid shape
grid_z = interpolated_values.reshape(xi.shape)

# Apply Gaussian smoothing
grid_z = gaussian_filter(grid_z, sigma=1)

# Step 4: Mask out points outside the concave hull
mask = np.array([not alpha_shape.contains(Point(xi_point, yi_point)) for xi_point, yi_point in zip(xi_flat, yi_flat)])
mask = mask.reshape(xi.shape)
grid_z[mask] = np.nan  # Mask out points outside the boundary

# Step 5: Visualize the concave hull and IDW heatmap
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Plot the concave hull polygon on the first subplot
gdf = gpd.GeoDataFrame(geometry=[alpha_shape])
gdf.plot(ax=ax[0], edgecolor='blue', facecolor='none', linewidth=2, label='Concave Hull (Alpha Shape)')
ax[0].scatter(df_test['x'], df_test['y'], color='red', label='Points')
ax[0].set_title('Concave Hull (Alpha Shape) of Points')
ax[0].set_xlabel('X Coordinate')
ax[0].set_ylabel('Y Coordinate')
ax[0].set_aspect('equal')
ax[0].legend()

# Plot the heatmap on the second subplot
sns.heatmap(grid_z, cmap=sns.color_palette("RdYlGn", as_cmap=True), annot=False, fmt="f", cbar=True, xticklabels=False, yticklabels=False, ax=ax[1])
ax[1].invert_yaxis()
ax[1].set_title('IDW Heatmap of Values')
ax[1].set_xlabel('X Coordinate')
ax[1].set_ylabel('Y Coordinate')

# Save the plot
plt.savefig("concave30.jpeg")





