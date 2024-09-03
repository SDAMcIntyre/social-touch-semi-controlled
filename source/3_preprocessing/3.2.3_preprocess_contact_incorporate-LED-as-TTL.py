import csv
import os
import pandas as pd
import re
import sys
import warnings
import matplotlib.pyplot as plt

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
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "2_processed", "kinect")

    # get input base directory
    db_path_input_contact = os.path.join(db_path, "contact", "1_block-order")
    db_path_input_led = os.path.join(db_path, "led", "0_block-order")

    # get output base directory
    db_path_output = os.path.join(db_path, "contact_and_led", "0_block-order")
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
    sessions_ST16 = ['2022-06-17_ST16-02']

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

    diff_ms_all = []
    names_contact = []
    names_led = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_contact_path_dir = os.path.join(db_path_input_contact, session)
        curr_contact_led_dir = os.path.join(db_path_input_led, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_contact_path_dir, ending='_contact.csv')

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            print(f"current file: {file_contact}")
            # check if led and nerve files exist for this contact file
            file_led = file_contact.replace("contact.csv", "LED.csv")
            file_led_abs = os.path.join(curr_contact_led_dir, file_led)
            try:
                with open(file_led_abs, 'r'):
                    pass
            except FileNotFoundError:
                print("Matching LED file does not exist.")
                continue

            output_filename = file_contact.replace("_contact.csv", "_kinect.csv")
            output_dir_abs = os.path.join(db_path_output, session)
            if not os.path.exists(output_dir_abs):
                os.makedirs(output_dir_abs)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing:
                try:
                    with open(output_filename_abs, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass

            contact = pd.read_csv(file_contact_abs)
            led = pd.read_csv(file_led_abs)

            # check if the two datasets are somewhat similar
            nframe_contact = len(contact)
            nframe_led = len(led)
            ratio = abs(nframe_contact-nframe_led)/nframe_contact
            # if the ratio of frames is too large, ignore the files
            if ratio > .05:
                warnings.warn(f"number of elements are too different between contact and led csv files: {ratio*100}%")
                contact_tmax = contact['t'].iloc[-1]
                led_tmax = led['time (second)'].iloc[-1]
                diff_ms = (contact_tmax-led_tmax) * 1000
                print(f"current filename = {file_contact.replace('_contact.csv', '')}:")
                print(f"contact = {contact_tmax:.3f} seconds.")
                print(f"led = {led_tmax:.3f} seconds.")
                print(f"diff = {diff_ms:.1f} milliseconds.")
                print(f"diff = {abs(nframe_contact-nframe_led)} frames.")
                print(f"------------------\n")

            # prepare dataframes for merging
            led = led.rename(columns={"time (second)": "t"}).round(3)
            contact["t"] = contact["t"].round(3)

            # generate output dataframe
            df_output = contact
            result = pd.merge(led, contact, on='t', how='outer')
            df_output = pd.merge(df_output, led,  on='t', how='outer')

            # keep variables for the report
            if generate_report:
                diff_ms_all.append(len(contact)-len(led))
                names_contact.append(file_contact)
                names_led.append(file_led)

            if show:
                print(df_output)

            # save data on the hard drive ?
            if save_results:
                df_output.to_csv(output_filename_abs, index=False)

    if generate_report:
        report_filename = os.path.join(db_path_output, "frame_differences_report.csv")
        report_data = []
        for name_contact, name_led, diff_ms in zip(names_contact, names_led, diff_ms_all):
            report_data.append({"filename_contact": name_contact, "filename_led": name_led, "frame_difference": diff_ms})
        with open(report_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["filename_contact", "filename_led", "frame_difference"])
            writer.writeheader()
            for row in report_data:
                writer.writerow(row)

    print("done.")

























