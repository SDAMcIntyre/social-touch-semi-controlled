import os
# os.environ["MPLBACKEND"] = "Agg"  # Use a non-interactive backend

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.signal import correlate2d
import shutil
import sys
import warnings

# homemade libraries
# current_dir = Path(__file__).resolve()
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402
import numpy as np



if __name__ == "__main__":
    # parameters visualisation
    verbose = True
    show = False  # If user wants to monitor what's happening
    show_steps = False

    print("Step 0: Extract the data embedded in the selected sessions.")
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "1_kinect_and_nerve_shandata")
    db_path_input = os.path.join(db_path, "0_1_by-units_correct-contact")

    traceability_file_path = os.path.join(db_path_input, 'traceability.csv')
    df = pd.read_csv(traceability_file_path)
    # Create a new column for the x-axis labels by combining Session and Block ID
    df['Session_Block'] = df['Session'].astype(str) + ' Block-' + df['Block ID'].astype(str)
    # Keep only the part after 'ST' (including 'ST') in the Session_Block column
    df['Session_Block'] = df['Session_Block'].apply(lambda x: x.split('ST', 1)[-1] if 'ST' in x else x)


    # Plot Max Correlation with xlabels as session+block
    plt.figure(figsize=(10, 6))
    plt.plot(df['Session_Block'], df['Max Correlation'], marker='o')
    plt.xlabel('Session_Block')
    plt.ylabel('Max Correlation')
    plt.title('Max Correlation by Session and Block')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show(block=True)

    # Plot areaRaw, depthRaw, velAbsRaw, velLatRaw, velLongRaw, velVertRaw with legends
    plt.figure(figsize=(10, 6))
    plt.plot(df['Session_Block'], df['areaRaw'], marker='o', label='areaRaw')
    plt.plot(df['Session_Block'], df['depthRaw'], marker='o', label='depthRaw')
    plt.plot(df['Session_Block'], df['velAbsRaw'], marker='o', label='velAbsRaw')
    plt.plot(df['Session_Block'], df['velLatRaw'], marker='o', label='velLatRaw')
    plt.plot(df['Session_Block'], df['velLongRaw'], marker='o', label='velLongRaw')
    plt.plot(df['Session_Block'], df['velVertRaw'], marker='o', label='velVertRaw')
    plt.xlabel('Session_Block')
    plt.ylabel('Values')
    plt.title('Various Metrics by Session and Block')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    print("done.")








