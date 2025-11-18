import os
import h5py
import numpy as np
import cv2

from config import get_config

config = get_config('config.txt')

# Selected channel(s)
channels = [12]
max_values = [320]
min_values = [180]

# Global variables
global files_list, pair_dict, cache_dict, cache_list
cache_dict = {}
cache_list = []

# Region selection (relative -> absolute)
region_start_col = config.start_col
region_start_row = config.start_row

absolute_start_col = region_start_col
absolute_start_row = region_start_row


# -----------------------------------------------------------
# Main Function: Convert one HDF file to PNG image
# -----------------------------------------------------------
def generate_images(image_path, store_path):
    """
    Read, cache, and convert satellite HDF data to PNG images.
    """

    # Ensure output directory exists
    parent_dir = os.path.dirname(store_path)
    if parent_dir != "" and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    file_need_write = update_cache(image_path)

    global cache_dict
    if file_need_write:
        for file in file_need_write:
            data2png(cache_dict[file].copy(), store_path)
            cache_dict.pop(file)


# -----------------------------------------------------------
# Cache Management
# -----------------------------------------------------------
def update_cache(image_path):
    """
    Update cache dictionary with the data from the input HDF file.

    Return:
        list of files that need to be written (if newly loaded)
    """

    global cache_dict

    # We only load the requested file
    files_need_read = [image_path]

    for file in files_need_read:
        try:
            nom_imgs = read_hdf(file, region_start_row, region_start_col)
            cache_dict[file] = nom_imgs
            return files_need_read

        except Exception as e:
            print("Error in update_cache:", e)
            return []

    return []


# -----------------------------------------------------------
# Convert HDF cached data to PNG
# -----------------------------------------------------------
def data2png(imgs, output_path):
    """
    Convert the loaded satellite data to a PNG image.
    """

    # Scaling to brightness (same logic as original code)
    bright = 255 * (max_values[0] - imgs[0]) / (max_values[0] - min_values[0])
    bright = bright.astype('uint8')

    # Replace file extension with .png safely
    output_path_png = os.path.splitext(output_path)[0] + ".png"

    cv2.imwrite(output_path_png, bright)


# -----------------------------------------------------------
# Read a FY4A AGRI HDF file
# -----------------------------------------------------------
def read_hdf(file_path, region_start_row=176, region_start_col=504, nb_cols=2747, nb_rows=2747):
    """
    Read FY4A AGRI HDF file and extract the target region.

    Args:
        file_path: Path to the HDF file.
        region_start_row, region_start_col: Absolute region coordinates.
        nb_cols, nb_rows: Output crop size.

    Returns:
        Array [C, nb_rows, nb_cols] of selected channel data.
    """

    print("Reading:", file_path)
    fdi_file = h5py.File(file_path)

    start_col = fdi_file.attrs['Begin Pixel Number'][0]
    start_row = fdi_file.attrs['Begin Line Number'][0]
    print("File start_row/start_col:", start_row, start_col)

    row_start, row_end, col_start, col_end = get_region_from_all(
        start_row, start_col, absolute_start_row, absolute_start_col, nb_cols, nb_rows
    )

    print("Crop region:", row_start, row_end, col_start, col_end)

    nom_all = np.ones((len(channels), nb_rows, nb_cols)) * 65535

    for i, c in enumerate(channels):
        nom_c = fdi_file[f'NOMChannel{str(c).zfill(2)}'][row_start:row_end, col_start:col_end]

        # Calibration table
        cal_c = fdi_file[f'CALChannel{str(c).zfill(2)}'][:]
        idx_err = nom_c >= len(cal_c)

        nom_c[idx_err] = 0
        nom_c = cal_c[nom_c]
        nom_c[idx_err] = 65535

        # Handle vertical alignment
        if row_start == 0:
            nom_all[i, start_row - absolute_start_row:start_row - absolute_start_row + nom_c.shape[0], :] = nom_c
        else:
            nom_all[i, :, :] = nom_c

        # Clipping
        nom_all[i, :, :] = np.clip(nom_all[i, :, :], min_values[i], max_values[i])

    fdi_file.close()
    return nom_all


# -----------------------------------------------------------
# Compute region indices relative to the HDF file
# -----------------------------------------------------------
def get_region_from_all(start_row, start_col, region_start_row, region_start_col, nb_cols, nb_rows):
    """
    Convert absolute region coordinates into local coordinates of the file.
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
# Main Execution
# -----------------------------------------------------------
if __name__ == '__main__':

    paths = ["./HDFfile_examples"]     # Input HDF directory
    store_paths = ["./imgs_full"]      # Output directory for PNG images

    for dir, store_path in zip(paths, store_paths):

        # Ensure output directory exists
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)

        for file in os.listdir(dir):
            try:
                image_path = os.path.join(dir, file)
                output_path = os.path.join(store_path, file)

                generate_images(image_path, output_path)

            except Exception as e:
                print("Error processing file:", file, e)
