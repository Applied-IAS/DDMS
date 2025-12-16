import PIL.Image as Image
import os
import glob
from functools import cmp_to_key

# ---------------------------------------------------------
# 1. Core Fusion Function 
# (Kept mostly unchanged, tweaked save path logic)
# ---------------------------------------------------------
def fusion(path1, path2, output_path):
    """
    Fuse satellite image (path1) and mask image (path2), and save to output_path
    """
    try:
        layer1 = Image.open(path1).convert('RGBA')   # Base image background
        # Cropping operation: retained based on your original code
        layer1 = layer1.crop((0, 19, 1280, 19+730))
        
        layer2 = Image.open(path2).convert('RGBA')    # mask

        # Ensure sizes match to prevent errors
        if layer1.size != layer2.size:
            layer2 = layer2.resize(layer1.size, Image.BILINEAR)

        final = Image.new("RGBA", layer1.size)             # Synthesized image
        final = Image.alpha_composite(final, layer1)
        final = Image.alpha_composite(final, layer2)

        final = final.convert('RGB')
        final.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error processing {path1}: {e}")
        return None

# ---------------------------------------------------------
# 2. New: GIF Generation Function
# ---------------------------------------------------------
def create_gif(image_paths, save_name, duration=200):
    """
    Synthesize a list of images into a GIF
    :param image_paths: List of image paths (sorted)
    :param save_name: Filename for the saved GIF
    :param duration: Duration of each frame (in milliseconds)
    """
    if not image_paths:
        return

    frames = []
    for path in image_paths:
        try:
            frames.append(Image.open(path))
        except IOError:
            continue
    
    if frames:
        # save_all=True saves all frames, loop=0 means infinite loop
        frames[0].save(
            save_name, 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True, 
            duration=duration, 
            loop=0
        )
        print(f"GIF saved: {save_name}")

# ---------------------------------------------------------
# 3. Core Processing Logic (Integrates Fusion and GIF Creation)
# ---------------------------------------------------------
def process_sequence(gt_satellite, gt_convection):
    """
    Process a single sequence: fuse images -> generate GIF
    """
    fusion_dir = os.path.join(gt_satellite, 'fusion')
    if not os.path.exists(fusion_dir):
        os.makedirs(fusion_dir)
    
    processed_images = []
    
    # Assume each sequence has 16 frames (based on your code range(16))
    for i in range(16):
        path1 = os.path.join(gt_satellite, f'{i}.png')
        path2 = os.path.join(gt_convection, f'{i}.png')
        
        # Check if files exist to prevent crashing
        if os.path.exists(path1) and os.path.exists(path2):
            save_path = os.path.join(fusion_dir, f'{i}.png')
            result = fusion(path1, path2, save_path)
            if result:
                processed_images.append(result)
        else:
            # print(f"Warning: Frame {i} missing in {gt_satellite}")
            pass

    # --- Create GIF ---
    # Ensure images are sorted in order 0, 1, 2...
    # processed_images is already added in loop order, so usually sorted,
    # but to be safe, sort again numerically based on filename.
    processed_images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    if processed_images:
        gif_name = os.path.join(gt_satellite, 'animation.gif') # GIF saved in the sequence directory
        create_gif(processed_images, gif_name, duration=200) # 200ms = 5fps

# ---------------------------------------------------------
# 4. Main Loop Entry
# ---------------------------------------------------------
def write_gifs(root_path):
    # Get all numeric folders under root (0, 1, 2...)
    # os.listdir might be unordered or contain non-directory files
    subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    # Sort folders numerically, e.g., '0', '1', '10' -> 0, 1, 10
    try:
        subdirs.sort(key=lambda x: int(x))
    except ValueError:
        subdirs.sort() # If folder names are not pure numbers, fallback to string sort

    print(f"Found {len(subdirs)} sequences to process.")

    for subdir in subdirs:
        current_seq_path = os.path.join(root_path, subdir)
        
        convection_dir = os.path.join(current_seq_path, 'seg_new_bar')
        # satellite_dir is the current sequence directory itself, 
        # as per your code structure where satellite images are directly inside the numeric folder
        satellite_dir = current_seq_path 
        
        print(f"Processing sequence: {subdir}")
        process_sequence(satellite_dir, convection_dir)

if __name__ == '__main__':
    # Replace with your actual path
    target_path = '../results/evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'
    write_gifs(target_path)