import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
from scipy import signal
import sys
import warnings
import os
import math
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.processing.semicontrolled_data_cleaning as scd_cleaning  # noqa: E402



def tile_png_images(input_folder):
    """
    Selects a folder, finds all PNG images, tiles them into a grid,
    and saves the result to a user-specified location.
    """
    # --- Setup Tkinter (needed for dialogs) ---
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    input_path = Path(input_folder)
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a valid directory. Exiting.")
        return

    print(f"Selected folder: {input_folder}")

    # --- 2. Find PNG Images ---
    png_files = sorted(list(input_path.glob('*.png'))) # Use glob for pattern matching, sort for consistent order

    if not png_files:
        print(f"No PNG files found in '{input_folder}'. Exiting.")
        return

    print(f"Found {len(png_files)} PNG files:")
    # for f in png_files:
    #     print(f" - {f.name}") # Optionally print file names

    # --- 3. Determine Grid Size ---
    num_images = len(png_files)
    # Calculate grid dimensions (preferring closer to square)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    print(f"Creating a {cols}x{rows} grid.")

    # --- 4. Load Images and Get Tile Size ---
    images = []
    tile_width = 0
    tile_height = 0
    first_image = True

    print("Loading images...")
    try:
        for i, file_path in enumerate(png_files):
            img = Image.open(file_path)
            # Use the dimensions of the first image as the standard tile size
            if first_image:
                tile_width, tile_height = img.size
                print(f"Using tile size from first image ({file_path.name}): {tile_width}x{tile_height}")
                first_image = False
            
            # Optional: Resize images if they are not the standard size
            # Uncomment the following lines if you want all tiles to be forced to the first image's size
            # if img.size != (tile_width, tile_height):
            #     print(f" - Resizing {file_path.name} from {img.size} to ({tile_width}, {tile_height})")
            #     img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)

            images.append(img)
            
    except Exception as e:
        print(f"\nError loading image '{file_path.name}': {e}")
        print("Please ensure all PNG files are valid image files. Exiting.")
        # Clean up already opened images
        for img in images:
            img.close()
        return
        
    if tile_width == 0 or tile_height == 0:
        print("Error: Could not determine tile dimensions (maybe the first image was invalid?). Exiting.")
        # Clean up images
        for img in images:
            img.close()
        return

    # --- 5. Create Output Canvas ---
    canvas_width = cols * tile_width
    canvas_height = rows * tile_height
    # Use 'RGBA' to handle potential transparency in PNGs
    output_canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0)) # Transparent background
    print(f"Created output canvas: {canvas_width}x{canvas_height}")

    # --- 6. Tile Images onto Canvas ---
    print("Tiling images...")
    current_x = 0
    current_y = 0
    for i, img in enumerate(images):
        col_index = i % cols
        row_index = i // cols

        paste_x = col_index * tile_width
        paste_y = row_index * tile_height

        # Make sure image mode is compatible for pasting (e.g., convert P mode)
        if img.mode == 'P': # Palette mode often needs conversion
             img = img.convert('RGBA')
        elif img.mode != 'RGBA' and img.mode != 'RGB': # Convert other common modes if needed
             img = img.convert('RGBA')

        # Paste the image. If the image has transparency, it will be used.
        try:
            # Ensure pasting considers alpha channel if present
            if img.mode == 'RGBA':
                 output_canvas.paste(img, (paste_x, paste_y), img) 
            else: # Paste without mask if no alpha
                 output_canvas.paste(img, (paste_x, paste_y))
                 
        except ValueError as ve:
             print(f"\nWarning: Could not paste image {i+1} ({png_files[i].name}) correctly.")
             print(f"   Error: {ve}")
             print(f"   Image mode: {img.mode}, Canvas mode: {output_canvas.mode}")
             print("   Skipping this image. The resulting grid might have gaps.")
             # Optionally, fill the space with a color or leave it transparent
             # from PIL import ImageDraw
             # draw = ImageDraw.Draw(output_canvas)
             # draw.rectangle([paste_x, paste_y, paste_x + tile_width, paste_y + tile_height], fill=(200, 200, 200, 255)) # Grey placeholder
             # del draw
        
        # Close the source image after pasting to free memory
        img.close()

    if output_canvas is None:
        a=1

    return output_canvas



# --- Run the function ---
if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    generate_report = True
    
    filename_output_end = "_TTLs_alignement_stacked.png"

    # result saving parameters
    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "3_merged", "sorted_by_block")
    # get output base directory
    db_path_output = os.path.join(db_path, "4_images", "TTLs_alignement")
    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    # Session names
    sessions_ST13 = ['2022-06-14_ST13-01',
                     '2022-06-14_ST13-02',
                     '2022-06-14_ST13-03']

    sessions_ST14 = ['2022-06-15_ST14-01',
                     '2022-06-15_ST14-02',
                     '2022-06-15_ST14-03',
                     '2022-06-15_ST14-04']

    sessions_ST15 = ['2022-06-16_ST15-01',
                     '2022-06-16_ST15-02']

    sessions_ST16 = ['2022-06-17_ST16-02',
                     '2022-06-17_ST16-03',
                     '2022-06-17_ST16-04',
                     '2022-06-17_ST16-05']

    sessions_ST18 = ['2022-06-22_ST18-01',
                     '2022-06-22_ST18-02',
                     '2022-06-22_ST18-04']
    sessions = []
    sessions = sessions + sessions_ST13
    sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    sessions = sessions + sessions_ST16
    sessions = sessions + sessions_ST18
    print(sessions)

    lag_list = []
    ratio_list = []
    file_list = []
    comment_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        
        output_canvas = tile_png_images(curr_dir)
        
        if output_canvas is None:
            continue
        
        # --- 8. Save the Output Image ---
        if save_results:
            try:
                output_file = os.path.join(db_path_output, f"{session}_{filename_output_end}")
                print(f"Saving tiled image to: {output_file}")
                # Ensure saving preserves transparency if saving as PNG
                output_canvas.save(output_file)
                print("Image saved successfully!")
            except Exception as e:
                print(f"Error saving image: {e}")
            finally:
                output_canvas.close() # Ensure canvas is closed

            

    print("done.")

