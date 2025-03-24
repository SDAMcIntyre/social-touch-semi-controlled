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
from scipy.signal import correlate2d, resample
from sklearn.preprocessing import MinMaxScaler
import shutil
import sys
import warnings

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# homemade libraries
# current_dir = Path(__file__).resolve()
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402
import numpy as np



def extract_session(s):
    # Define the pattern to match the required parts of the string
    pattern = r"(\d{4}-\d{2}-\d{2})_ST(\d{2})-(\d{2})\\\\block-order-(\d{2})\\\\"
    match = re.match(pattern, s)
    if match:
        date = match.group(1)
        st_value = match.group(2)
        after_st = match.group(3)
        block_order = match.group(4)
        return date, st_value, after_st, block_order
    else:
        return None
    


def prepare_plot_dataframes(dataframes, datacol, timecol='time', timeabsolute=True, title=None, subtitles=None):
    if not timeabsolute:
        for df in dataframes:
            df.loc[:, timecol] = df[timecol] - df[timecol].min()

    num_plots = len(datacol)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 7))

    for idx, col in enumerate(datacol):
        for i, df in enumerate(dataframes):
            subtitle = subtitles[i] if subtitles and i < len(subtitles) else f'DataFrame {i+1}'
            df.plot(x=timecol, y=col, ax=axes[idx], label=subtitle)
        axes[idx].legend()
        axes[idx].set_title(col)

    # Set the main title if provided
    if title:
        fig.suptitle(title)

    plt.tight_layout()



def lag_finder(y1, y2, show=False):
    n = len(y1)
    delay_arr = np.linspace(-0.5*n/2, 0.5*n/2, n).astype(int)
    # Perform cross-correlation between y2 and y1
    cross_corr = signal.correlate(y2, y1, mode='same')

    # Perform auto-correlation for y1 and y2
    auto_corr_y1 = signal.correlate(y1, y1, mode='same')
    auto_corr_y2 = signal.correlate(y2, y2, mode='same')

    # Calculate normalization factor using the middle value of auto-correlations
    normalization_factor = np.sqrt(auto_corr_y1[int(n/2)] * auto_corr_y2[int(n/2)])

    # Normalize the cross-correlation result
    corr = cross_corr / normalization_factor
    max_corr = corr[np.argmax(corr)]
    delay = delay_arr[np.argmax(corr)]
    print(f'y2 is {delay} behind y1 (corr={max_corr})')

    if show:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(delay, 3)) + ' samples')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show(block=True)
    return max_corr, delay



if __name__ == "__main__":
    # parameters saving and processes
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    save_figures = True
    # parameters visualisation
    verbose = True
    show = False  # If user wants to monitor what's happening

    # parameters behavior
    xcorr_with_shift = True

    print("Step 0: Extract the data embedded in the selected sessions.")
    # get database directory and input base directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    db_path_input = os.path.join(db_path, "2_processed", "kinect")
    db_path_september_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "0_3_by-units_jan-and-sept_standard-cols_block-order")

    # get output base directory
    db_path_output = os.path.join(db_path, "99_misc", "kinect", "compare_shan-september_basil-rerun-2025-03")
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
    
    # Initialize an empty list to store results
    results = []

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        file = [f.name for f in Path(db_path_september_input).iterdir() if session in f.name]
        if len(file) != 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        file = file[0]
        file_abs = os.path.join(db_path_september_input, file)
        data_sept = pd.read_csv(file_abs)
        data_sept.rename(columns={'t': 'time'}, inplace=True)
        data_sept.rename(columns={'areaRaw': 'contact_area'}, inplace=True)
        data_sept.rename(columns={'depthRaw': 'contact_depth'}, inplace=True)
        print(f'{session}: data has {len(data_sept)} samples.')

        curr_data = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_data, ending='somatosensory_data.csv')
        for file_abs, file in zip(files_abs, files):
            block_2025 = pd.read_csv(file_abs)
            
            match = re.search(r'block-order-(\d+)', file_abs)
            if match:
                block_id = int(match.group(1))
                data_sept['block_order'] == block_id
            block_sept = data_sept[data_sept['block_order'] == block_id]

            if block_sept.empty:
                continue

            # Normalize the specified columns
            scaler = MinMaxScaler()
            block_sept.loc[:, ['contact_area', 'contact_depth']] = scaler.fit_transform(block_sept[['contact_area', 'contact_depth']])
            block_2025.loc[:, ['contact_area', 'contact_depth']] = scaler.fit_transform(block_2025[['contact_area', 'contact_depth']])
        
            depth_march = block_2025['contact_depth'].values

            # downsample shan's data (1kHz up sampling)
            depth_sept_1kHz = block_sept['contact_depth'].values
            depth_sept_1kHz[np.isnan(depth_sept_1kHz)] = 0
            Fs_sept = 1 / np.mean(np.diff(block_sept['time']))
            Fs_march = 1 / np.mean(np.diff(block_2025['time']))
            # Ensure the signal length is a multiple of the downsampling factor
            factor = int(Fs_sept // Fs_march)  # Downsampling factor
            trimmed_signal = depth_sept_1kHz[:len(depth_sept_1kHz) - (len(depth_sept_1kHz) % factor)]
            depth_sept = np.mean(trimmed_signal.reshape(-1, factor), axis=1)
        
            # Append zeros to the shorter vector
            nsample_march = len(depth_march)
            nsample_sept = len(depth_sept)
            if nsample_march < nsample_sept:
                depth_march = np.append(depth_march, [0] * (nsample_sept - nsample_march))
            else:
                depth_sept = np.append(depth_sept, [0] * (nsample_march - nsample_sept))
            # remove any nan values
            depth_march[np.isnan(depth_march)] = 0

            if xcorr_with_shift:
                # Calculate the highest cross-correlation and the best shift
                corr_score, delay = lag_finder(depth_sept, depth_march, show=False)
                shift = -delay
            else:
                corr_score = np.corrcoef(depth_sept, depth_march)[0, 1]
                shift = 0
                
            if save_results:
                # Extract info of the loaded data
                session_str = file_abs.replace(db_path_input+"\\", "")
                session_folder, block_folder, _ = session_str.split("\\")
                date = re.search(r'\d{4}-\d{2}-\d{2}', session_folder).group()
                integers = re.findall(r'\d+', session_folder.split('_')[1])
                part_id = int(integers[0])
                neuron_id = int(integers[1])
                block_order_val = int(re.findall(r'\d+', block_folder)[0])
                
                # Store the result with metadata
                results.append({
                    'date': date,
                    'participant': part_id,
                    'neuron id': neuron_id,
                    'block order': block_order_val,
                    'correlation score': corr_score,
                    'correlation shift (sample)': shift
                })
            
            if save_figures or show:
                title_base = "_".join(file_abs.replace(db_path_input+"\\", "").split("\\")[:2])
                title = title_base + f"\ncorrelation score = {corr_score:.2f}"
                title += f"\nsample shift applied on march data = {shift}"
                
                shifted_block_2025 = block_2025.shift(periods=shift)
                prepare_plot_dataframes([block_sept, shifted_block_2025], 
                                        datacol=['contact_area', 'contact_depth'], timecol='time', 
                                        timeabsolute=False, title=title, subtitles=['september 2024', 'march 2025 shifted'])
                if save_figures:
                    fig_filename = title_base + '_shift-impact.png'
                    fig_filename_abs = os.path.join(db_path_output, fig_filename)
                    if force_processing or not os.path.exists(fig_filename_abs):
                        # Get the current figure, save and close
                        fig = plt.gcf()
                        fig.savefig(fig_filename_abs)
                if show:
                    plt.show(block=True)
                else:
                    plt.close(fig)

    output_filename_abs = os.path.join(db_path_output, 'sept-rerun2025_comparison-depth_correlation-results.csv')
    if xcorr_with_shift:
        output_filename_abs.replace(".csv", "_with-shifting.csv")
    
    if save_results:
        if force_processing or not os.path.exists(output_filename_abs):
            # Create a DataFrame from the results
            df = pd.DataFrame(results)
            # Save the DataFrame as a CSV file
            df.to_csv(output_filename_abs, index=False)

    
                








