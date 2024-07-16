import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True
    generate_report = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = path_tools.get_database_path()

    # get input base directory
    db_path_input = os.path.join(db_path, "semi-controlled", "2_processed", "nerve", "2_block-order")

    # get metadata file
    md_neuron_filename_abs = os.path.join(db_path, "semi-controlled", "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    md_neuron_df = pd.read_csv(md_neuron_filename_abs)

    # get output base directory
    db_path_output = os.path.join(db_path, "semi-controlled", "2_processed", "nerve", "3_cond-velocity-adj")
    if not os.path.exists(db_path_output):
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

    filenames = []
    calculated_lag_sec = []
    calculated_lag_sample = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_path_dir = os.path.join(db_path_input, session)

        files_nerve_abs, files_nerve = path_tools.find_files_in_directory(curr_path_dir, ending='_nerve.csv')

        for file_nerve_abs, file_nerve in zip(files_nerve_abs, files_nerve):
            print(f"current file: {file_nerve}")
            output_dir_abs = os.path.join(db_path_output, session)
            if not os.path.exists(output_dir_abs):
                os.makedirs(output_dir_abs)
            output_filename_abs = os.path.join(output_dir_abs, file_nerve)
            if not force_processing:
                try:
                    with open(output_filename_abs, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass

            nerve = pd.read_csv(file_nerve_abs)

            # get the neuron code from the file name
            neuron_id = file_nerve.split("_")[1]
            # extract the row corresponding to the neuron code / unit name
            curr_neuron_row = md_neuron_df[md_neuron_df["Unit_name"] == neuron_id]
            # calculate the expected lag between the mechanoreceptor and the electrode recording
            cond_vel_m_s = curr_neuron_row["conduction_velocity (m/s)"].values[0]
            distance_cm = curr_neuron_row["electrode_endorgan_distance (cm)"].values[0]
            distance_m = distance_cm * .01
            lag_sec = distance_m / cond_vel_m_s

            # calculate the number of samples that needs to be shifted
            t = nerve.Sec_FromStart.values
            data_Fs = 1 / np.mean(np.diff(t))
            lag_nsample = int(data_Fs * lag_sec)

            print(f"estimated lag: {lag_sec} seconds ({lag_nsample} samples).")
            print(f"---")

            # shift the neuron sample, so it matches better the reality, hence the video recordings
            df_output = nerve.copy()
            df_output['Nervespike1'] = nerve['Nervespike1'].shift(-lag_nsample, fill_value=0)
            df_output['Freq'] = nerve['Freq'].shift(-lag_nsample, fill_value=0)

            # keep variables for the report
            if generate_report:
                filenames.append(file_nerve)
                calculated_lag_sec.append(lag_sec)
                calculated_lag_sample.append(lag_nsample)

            if show:
                # Create a figure with two subplots
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

                x = nerve.Sec_FromStart
                sig1 = nerve["Freq"].values
                sig2 = df_output["Freq"].values

                # Plot the first signal on the first subplot
                ax1.plot(x, sig1, label='Original Nerve signal')
                ax1.set_title('Original Nerve signal')
                ax1.set_ylabel('Amplitude')
                ax1.legend()

                # Plot the second signal on the second subplot
                ax2.plot(x, sig2, label='Shifted nerve signal', color='orange')
                ax2.set_title(f"Shifted nerve signal by {lag_nsample} samples.")
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Amplitude')
                ax2.legend()

                # Plot the second signal on the second subplot
                ax3.plot(x, sig1 - sig2, label='sig1 - sig2', color='red')
                ax3.set_title("Difference")
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Amplitude')
                ax3.legend()

                # Adjust the layout
                plt.tight_layout()
                # Show the plot and wait until the window is closed
                plt.show()

            # save data on the hard drive ?
            if save_results:
                df_output.to_csv(output_filename_abs, index=False)

    if generate_report:
        report_filename = os.path.join(db_path_output, "conduction_velocity_lag_report.csv")
        report_data = []
        for filename, lag_sec, lag_sample in zip(filenames, calculated_lag_sec, calculated_lag_sample):
            report_data.append({"filename": filename, "lag_sec": lag_sec, "lag_sample": lag_sample})

        with open(report_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "lag_sec", "lag_sample"])
            writer.writeheader()
            for row in report_data:
                writer.writerow(row)

    print("done.")

























