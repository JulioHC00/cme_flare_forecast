import os
import zarr
import numpy as np
from tqdm import tqdm

def safe_add(a, b, tol=1e-6):
    try:
        sum_ab = np.add(a, b, dtype=np.longdouble)
    except FloatingPointError:
        print("Warning: Numerical limit reached. Value set to maximum representable number.")
        return np.finfo(np.longdouble).max
    if np.any(np.abs(sum_ab - a - b) > tol):
        print("Warning: Addition might be imprecise.")
        print(sum_ab)
    return sum_ab

def calculate_statistics(root_folder):
    sum_values = np.zeros(3, dtype=np.longdouble)
    sum_squared_diff = np.zeros(3, dtype=np.longdouble)
    count_values = np.zeros(1, dtype=np.ulonglong)

    # Calculate mean in the first pass
    folders = os.listdir(root_folder)
    for folder in tqdm(folders, desc='Calculating mean'):
        folder_path = os.path.join(root_folder, folder)
        try:
            zarr_file = zarr.open(folder_path, mode="r").astype(np.longdouble)
        except zarr.errors.PathNotFoundError:
            print(f"Folder {folder} not found")
            continue
        sum_values = safe_add(sum_values, np.sum(zarr_file, axis=(0, 2, 3)))
        count_values = safe_add(count_values, np.prod(zarr_file.shape) // zarr_file.shape[1])

    mean_values = sum_values / count_values

    # Calculate variance in the second pass
    for folder in tqdm(folders, desc='Calculating variance'):
        folder_path = os.path.join(root_folder, folder)
        try:
            zarr_file = zarr.open(folder_path, mode="r").astype(np.longdouble)
        except zarr.errors.PathNotFoundError:
            print(f"Folder {folder} not found")
            continue
        diff_squared = (zarr_file - mean_values.reshape((1, 3, 1, 1))) ** 2
        sum_squared_diff = safe_add(sum_squared_diff, np.sum(diff_squared, axis=(0, 2, 3)))

    variance_values = sum_squared_diff / count_values

    print("Double pass results:")
    print(f"Mean: {mean_values}")
    print(f"Variance: {variance_values}")

    return mean_values, variance_values

def normalize_and_calculate_stats(ROOT, global_mean, global_var):
    new_sum_values = np.zeros(3, dtype=np.longdouble)
    new_sum_squared_diff = np.zeros(3, dtype=np.longdouble)
    new_count_values = np.zeros(1, dtype=np.ulonglong)
    
    global_std = np.sqrt(global_var)
    
    for folder in tqdm(os.listdir(ROOT)):
        try:
            zarr_file = zarr.open(os.path.join(ROOT, folder), mode="r")
        except zarr.errors.PathNotFoundError:
            print(f"Folder {folder} not found")
            continue

        zarr_file = zarr_file.astype(np.longdouble)

        # Reshape global mean and std to match the shape of the data

        global_mean_reshaped = global_mean.reshape((1, 3, 1, 1))
        global_std_reshaped = global_std.reshape((1, 3, 1, 1))

        global_mean_extended = np.repeat(global_mean_reshaped, zarr_file.shape[0], axis=0)
        global_std_extended = np.repeat(global_std_reshaped, zarr_file.shape[0], axis=0)
        global_mean_extended = np.repeat(global_mean_extended, zarr_file.shape[2], axis=2)
        global_std_extended = np.repeat(global_std_extended, zarr_file.shape[2], axis=2)
        global_mean_extended = np.repeat(global_mean_extended, zarr_file.shape[3], axis=3)
        global_std_extended = np.repeat(global_std_extended, zarr_file.shape[3], axis=3)

        # Normalize the data using the provided global mean and variance
        normalized_data = (zarr_file - global_mean_extended) / global_std_extended

        new_sum_values += np.sum(normalized_data, axis=(0, 2, 3), dtype=np.longdouble)
        
        batch_size = zarr_file.shape[0]
        height = zarr_file.shape[2]
        width = zarr_file.shape[3]
        
        new_count_values += batch_size * height * width
    
    new_mean_values = new_sum_values / new_count_values

    # Second pass for new variance
    for folder in tqdm(os.listdir(ROOT)):
        try:
            zarr_file = zarr.open(os.path.join(ROOT, folder), mode="r")
        except zarr.errors.PathNotFoundError:
            print(f"Folder {folder} not found")
            continue

        global_mean_extended = np.repeat(global_mean_reshaped, zarr_file.shape[0], axis=0)
        global_std_extended = np.repeat(global_std_reshaped, zarr_file.shape[0], axis=0)
        global_mean_extended = np.repeat(global_mean_extended, zarr_file.shape[2], axis=2)
        global_std_extended = np.repeat(global_std_extended, zarr_file.shape[2], axis=2)
        global_mean_extended = np.repeat(global_mean_extended, zarr_file.shape[3], axis=3)
        global_std_extended = np.repeat(global_std_extended, zarr_file.shape[3], axis=3)

        zarr_file = zarr_file.astype(np.longdouble)
        normalized_data = (zarr_file - global_mean_extended) / global_std_extended

        # Also extend the new mean

        new_mean_extended = np.repeat(new_mean_values.reshape((1, 3, 1, 1)), zarr_file.shape[0], axis=0)
        new_mean_extended = np.repeat(new_mean_extended, zarr_file.shape[2], axis=2)
        new_mean_extended = np.repeat(new_mean_extended, zarr_file.shape[3], axis=3)

        new_sum_squared_diff += np.sum((normalized_data - new_mean_extended) ** 2, axis=(0, 2, 3), dtype=np.longdouble)
    
    new_variance_values = new_sum_squared_diff / new_count_values
    
    print("Normalization results:")
    print(f"Mean: {new_mean_values}")
    print(f"Variance: {new_variance_values}")
    print(f"Standard deviation: {np.sqrt(np.abs(new_variance_values))}") 

    return new_mean_values, new_variance_values

ROOT = "/home/julio/cmeml/data/cutouts/"
global_mean, global_var = calculate_statistics(ROOT)
final_mean, final_var = normalize_and_calculate_stats(ROOT, global_mean, global_var)