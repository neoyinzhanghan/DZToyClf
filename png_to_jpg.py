import os
from PIL import Image
from tqdm import tqdm

png_dir = "/media/hdd3/neo/not_enough_focus_regions_topviews/BMA_grid_rep"
save_dir = "/media/hdd3/neo/not_enough_focus_regions_topviews_jpg/BMA_grid_rep"

os.makedirs(save_dir, exist_ok=True)
# get the paths to all the png files in the directory
png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]


for png_file in tqdm(png_files, desc="Converting PNG to JPG"):
    # Open the PNG file
    img = Image.open(png_file)
    
    # Save the image as a JPG file
    jpg_path = os.path.join(save_dir, os.path.basename(png_file).replace(".png", ".jpg"))