import os
import h5py
import numpy as np
import cv2
from config import get_config

# This script supports REGC, DISK, and HDF files.
# It extracts the China-region image from FY4A AGRI data.

config = get_config("config.txt")

# Selected channels and their clipping ranges
channels = [12]
max_values = [320]
min_values = [180]

# Global caching
cache_dict = {}
cache_list = []

# China region offset (fixed for FY4A AGRI China area)
china_start_col = 750
china_start_row = 190

# Local region defined in config
region_start_col = config.start_col
region_start_row = config.start_row
nb_cols = config.end_col - config.start_col
nb_rows = config.end_row - config.start_row

# Absolute target region inside full FY4A disk image
absolute_start_col = region_start_col + china_start_col
absolute_start_row = region_start_row + china_start_row


# -----------------------------------------------------------
# Main function: Convert FY4A HDF to PNG
# -----------------------------------------------------------
def generate_images(image_path, store_path):
    """
    Load an HDF file into cache and export the China-region image as PNG.
    """

    # Ensure output directory exists
    parent_dir = os.path.dirname(store_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    files_to_write = update_cache(image_path)

    global cache_dict
    if files_to_write:
        for file in files_to_write:
            data2png(cache_dict[file].copy(), store_path)
            cache_dict.pop(file, None)


# -----------------------------------------------------------
# Cache Management
# -----------------------------------------------------------
def update_cache(image_path):
    """
    If the file is not in cache, load and decode it.

    Returns:
        list: files that need to be processed/written.
    """

    global cache_dict

    # Only one file per call
    files_to_read = [image_path]

    for file in files_to_read:
        try:
            nom_imgs = read_hdf(file, region_start_row, region_start_col, nb_cols, nb_rows)
            cache_dict[file] = nom_imgs
            return files_to_read

        except Exception as e:
            print(f"[Error] Failed to load HDF file {file}: {e}")
            return []

    return []


# -----------------------------------------------------------
# Convert cached array to PNG
# -----------------------------------------------------------
def data2png(imgs, output_path):
    """
    Convert calibrated FY4A image data to brightness PNG.
    """

    bright = 255 * (max_values[0] - imgs[0]) / (max_values[0] - min_values[0])
    bright = np.clip(bright, 0, 255).astype("uint8")

    # Replace extension with .png safely
    output_png = os.path.splitext(output_path)[0] + ".png"

    cv2.imwrite(output_png, bright)


# -----------------------------------------------------------
# Read FY4A AGRI HDF
# -----------------------------------------------------------
def read_hdf(file_path, region_start_row=176, region_start_col=504, nb_cols=2747, nb_rows=2747):
    """
    Read FY4A HDF file and extract the requested China subregion.

    Args:
        file_path (str): HDF file path.
        region_start_row, region_start_col: Target region (absolute).
        nb_cols, nb_rows: Output image size.

    Returns:
        numpy array of shape [C, nb_rows, nb_cols]
    """

    print(f"Reading file: {file_path}")

    fdi_file = h5py.File(file_path)

    start_col = fdi_file.attrs["Begin Pixel Number"][0]
    start_row = fdi_file.attrs["Begin Line Number"][0]
    print(f"File start_col/start_row: {start_col}, {start_row}")

    # Determine crop indices inside the file
    row_start, row_end, col_start, col_end = get_region_from_all(
        start_row, start_col,
        absolute_start_row, absolute_start_col,
        nb_cols, nb_rows
    )
    print("Crop region:", row_start, row_end, col_start, col_end)

    # Container for multi-channel output
    nom_all = np.ones((len(channels), nb_rows, nb_cols)) * 65535

    for i, c in enumerate(channels):

        # Read only the required crop region
        nom_c = fdi_file[f"NOMChannel{str(c).zfill(2)}"][row_start:row_end, col_start:col_end]

        # Convert index → calibrated value
        cal_c = fdi_file[f"CALChannel{str(c).zfill(2)}"][:]
        idx_err = nom_c >= len(cal_c)

        nom_c[idx_err] = 0
        nom_c = cal_c[nom_c]
        nom_c[idx_err] = 65535

        # Place into correct output position
        if row_start == 0:
            # Handle upper-edge incomplete coverage
            offset = start_row - absolute_start_row
            nom_all[i, offset:offset + nom_c.shape[0], :] = nom_c
        else:
            nom_all[i, :, :] = nom_c

        # Clip to predefined range
        nom_all[i] = np.clip(nom_all[i], min_values[i], max_values[i])

    fdi_file.close()
    return nom_all


# -----------------------------------------------------------
# Convert global absolute region → file-local pixel coordinates
# -----------------------------------------------------------
def get_region_from_all(start_row, start_col, region_start_row, region_start_col, nb_cols, nb_rows):
    """
    Convert absolute region coordinates to relative coordinates inside file.
    """

    if region_start_row >= start_row:
        row_start = region_start_row - start_row
        row_end = row_start + nb_rows
        col_start = region_start_col - start_col
        col_end = col_start + nb_cols
    else:
        row_start = 0
        row_end = region_start_row + nb_rows - start_row
        col_start = region_start_col - start_col
        col_end = col_start + nb_cols

    return row_start, row_end, col_start, col_end


# -----------------------------------------------------------
# Batch Processing
# -----------------------------------------------------------
paths = ["./HDFfile_examples"]
store_paths = ["./imgs_china"]

for input_dir, output_dir in zip(paths, store_paths):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        try:
            image_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            generate_images(image_path, output_path)

        except Exception as e:
            print(f"[Error] Failed processing {file}: {e}")
