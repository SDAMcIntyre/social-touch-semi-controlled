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



def create_output_filename(contact_file, led_file):
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    input_string = contact_file

    # Extract date
    date_match = re.search(date_pattern, input_string)
    date = date_match.group() if date_match else None
    if date is None:
        input_string = led_file
        date_match = re.search(date_pattern, input_string)
        date = date_match.group() if date_match else None

    # Extract integers from the input string without date
    integer_pattern = r'\d+'
    integers = re.findall(integer_pattern, input_string.replace(date, ""))

    # Construct new string with the extracted date and integers
    output_string = f"{date}_ST{integers[0]}-{integers[1]}_semicontrolled_block-order{integers[2]}_kinect.csv"

    return output_string



if __name__ == "__main__":   
    if os.path.sep == '/':  # Running on a Unix-like system
        OS_linux = True
    elif os.path.sep == '\\':  # Running on Windows
        OS_linux = False

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    generate_report = True

    input_contact_filename_ending = 'somatosensory_data_no-outlier.csv'
    input_led_filename_ending = '_LED.csv'
    show_warning = False  # If user wants to monitor what's happening
    show = False  # If user wants to monitor what's happening

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path_linux = os.path.expanduser('~/Documents/datasets/semi-controlled')
    if OS_linux:
        db_path = db_path_linux
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input_contact = os.path.join(db_path, "2_processed", "kinect")
    db_path_input_led = os.path.join(db_path, "2_processed", "kinect")
    # set output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect")
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
                     '2022-06-15_ST14-04'
                     ]

    sessions_ST15 = ['2022-06-16_ST15-01',
                     '2022-06-16_ST15-02']

    sessions_ST16 = ['2022-06-17_ST16-02',
                     '2022-06-17_ST16-03',
                     '2022-06-17_ST16-04',
                     '2022-06-17_ST16-05']

    sessions_ST18 = ['2022-06-22_ST18-01',
                     '2022-06-22_ST18-02',
                     '2022-06-22_ST18-04'
                     ]
    
    use_specific_sessions = True
    if not use_specific_sessions:
        sessions = []
        sessions = sessions + sessions_ST13
        sessions = sessions + sessions_ST14
        sessions = sessions + sessions_ST15
        sessions = sessions + sessions_ST16
        sessions = sessions + sessions_ST18
    else:
        sessions = ['2022-06-17_ST16-02']
    
    use_specific_blocks = True
    specific_blocks = ['block-order-16']

    print(sessions)

    diff_ms_all = []
    names_contact = []
    names_led = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_contact_path_dir = os.path.join(db_path_input_contact, session)
        curr_contact_led_dir = os.path.join(db_path_input_led, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_contact_path_dir, ending=input_contact_filename_ending)

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            if use_specific_blocks:
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file_contact_abs:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue
            print(f"current file: {file_contact_abs}")
            # if not ('block-order-08' in file_contact_abs): continue

            # check if led and nerve files exist for this contact file
            input_dirname = os.path.dirname(file_contact_abs)
            files_led_abs, files_led = path_tools.find_files_in_directory(input_dirname, ending=input_led_filename_ending)
            if not (isinstance(files_led_abs, list) and 1 == len(files_led_abs)):
                print(f"Not exactly one {input_led_filename_ending} file detected.")
                print(f"skip this file...")
                continue
            file_led_abs = files_led_abs[0]
            file_led = files_led[0]

            output_dir_abs = input_dirname.replace(db_path_input_contact, db_path_output)
            output_filename = create_output_filename(file_contact, file_led)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing and os.path.exists(output_filename_abs):
                print(f"File '{file_contact}' already exists and the processing is not forced. Move to next file...")
                continue

            contact = pd.read_csv(file_contact_abs)
            led = pd.read_csv(file_led_abs)

            # check if the two datasets are somewhat similar
            nframe_contact = len(contact)
            nframe_led = len(led)
            ratio = abs(nframe_contact-nframe_led)/nframe_contact
            # if the ratio of frames is too large, ignore the files
            if ratio > .05:
                warnings.warn(f"number of elements are too different between contact and led csv files: {ratio*100}%")
                contact_tmax = contact['time'].iloc[-1]
                led_tmax = led['time (second)'].iloc[-1]
                diff_ms = (contact_tmax-led_tmax) * 1000
                print(f"current filename = {file_contact.replace('_contact.csv', '')}:")
                print(f"contact = {contact_tmax:.3f} seconds.")
                print(f"led = {led_tmax:.3f} seconds.")
                print(f"diff = {diff_ms:.1f} milliseconds.")
                print(f"diff = {abs(nframe_contact-nframe_led)} frames.")
                print(f"------------------\n")

            # prepare dataframes for merging
            led = led.drop('green level', axis=1)
            led = led.rename(columns={"time (second)": "time"}).round(3)
            contact["time"] = contact["time"].round(3)

            # generate output dataframe
            df_output = contact
            result = pd.merge(led, contact, on='time', how='outer')
            df_output = pd.merge(df_output, led,  on='time', how='outer')
            
            # keep variables for the report
            if generate_report:
                diff_ms_all.append(len(contact)-len(led))
                dname = os.path.join(os.path.dirname(file_contact_abs).split(os.path.sep)[-2], os.path.dirname(file_contact_abs).split(os.path.sep)[-1])
                names_contact.append(os.path.join(dname, file_contact))
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

























