import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import re
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys
import warnings
import time


# homemade libraries
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.processing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
from libraries.processing.semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


def plot_heatmap(x, y, z, x_label='', y_label='', z_label='Neuron IFF Mean', fig_id=-1):
    """
    Plots a heatmap of interpolated data using x, y, and z values.
    
    Parameters:
        x (array-like): The x-values for the plot.
        y (array-like): The y-values for the plot.
        z (array-like): The z-values for the plot, representing the heatmap color.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        z_label (str): Label for the color bar.
    """
    # Create a grid for interpolation
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate the data
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Plot the heatmap
    if fig_id:
        plt.figure(fig_id, figsize=(8, 6))
    else:
        plt.figure(figsize=(8, 6))

    heatmap = plt.contourf(xi, yi, zi, 100, cmap='viridis')
    plt.colorbar(heatmap, label=z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{z_label}')
    # plt.show()


if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    # ----------------------
    save_results = False
    # ----------------------
    # ----------------------
    # ----------------------

    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "3_global_result")
    # get output filenames
    filename_input = "semicontrolled_single-touch_summary-statistics.csv"
    filename_input_abs = os.path.join(db_path_input, filename_input)

    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "3_global_result")
    if not os.path.exists(db_path_output) and save_results:
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    data = pd.read_csv(filename_input_abs)

    # Iterate over unique values and create chunks
    for neuron_id in data["neuron_ID"].unique():
        # Filter the DataFrame for the specified neuron
        data_neuron = data[data["neuron_ID"] == neuron_id]
        chunk = data_neuron[["stimulus_type", "contact_vel", "contact_area", "contact_depth", "neuron_IFF_mean"]]
        chunk = chunk.dropna()

        # Extract the columns
        stype = "tap"
        # Filter the DataFrame for the specified stimulus type
        filtered_chunk = chunk[chunk["stimulus_type"] == stype]
        title_base = f'{neuron_id}_{data_neuron.neuron_type.iloc[0]}'

        # Extract values for x, y, and z from the filtered DataFrame
        x = filtered_chunk['contact_vel'].values
        y = filtered_chunk['contact_depth'].values
        z = filtered_chunk['neuron_IFF_mean'].values
        filtered_chunk = filtered_chunk.sort_values(by='contact_vel')
        plot_heatmap(x, y, z, 'velocity (cm/sec)', 'area (cm^2 (?))', f'{title_base}: {stype}, IFF Mean', 1)
        plt.figure(2, figsize=(8, 6))
        plt.plot(x, z)
        plt.title(f'{title_base}: {stype}, IFF_mean(Velocity)')
        
        # Extract the columns
        stype = "stroke"
        # Filter the DataFrame for the specified stimulus type
        filtered_chunk = chunk[chunk["stimulus_type"] == stype]
        filtered_chunk = filtered_chunk.sort_values(by='contact_vel')

        # Extract values for x, y, and z from the filtered DataFrame
        x = filtered_chunk['contact_vel'].values
        y = filtered_chunk['contact_depth'].values
        z = filtered_chunk['neuron_IFF_mean'].values
        plot_heatmap(x, y, z, 'velocity (cm/sec)', 'area (cm^2 (?))', f'{title_base}: {stype}, IFF Mean', 3)
        
        plt.figure(4, figsize=(8, 6))
        plt.plot(x, z)
        plt.title(f'{title_base}: {stype}, IFF_mean(Velocity)')
        plt.show()
        
        pass 
        if 0:
            # One-way ANOVA for each factor ?
            for factor in ['stimulus_type', 'contact_vel', 'contact_area', 'contact_depth']:
                model = ols(f'neuron_IFF_mean ~ C({factor})', data=chunk).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print(f"ANOVA for factor {factor}")
                print(anova_table)

            unique_counts = chunk.groupby(factor)['neuron_IFF_mean'].nunique()
            if any(unique_counts == 1):
                print(f"Warning: Factor '{factor}' has groups with no variance.")

            model = ols(f'neuron_IFF_mean ~ C({factor})', data=chunk).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f"\nANOVA for factor {factor}")
            print(anova_table)
            

    # 5. save endpoints results
    if save_results:
        if not os.path.exists(db_path_output):
            os.makedirs(db_path_output)
        

    print("done.")







