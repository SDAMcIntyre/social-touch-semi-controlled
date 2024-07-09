import os
# homemade libraries
import path_tools as path_tools  # noqa: E402


def rename_files_in_folder(root_folder, old_word, new_word):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if old_word in filename:
                new_filename = filename.replace(old_word, new_word)
                old_filepath = os.path.join(dirpath, filename)
                new_filepath = os.path.join(dirpath, new_filename)
                os.rename(old_filepath, new_filepath)
                print(f'Renamed: {old_filepath} -> {new_filepath}')


# get database directory
database_path = path_tools.get_database_path()
database_path = os.path.join(database_path, "semi-controlled")
# Set the root folder path
#root_folder = os.path.join(database_path, "processed", "kinect", "contact", "1_block-order")
#root_folder = os.path.join(database_path, "processed", "kinect", "led")
#root_folder = os.path.join(database_path, "primary", "kinect", "1_block-order")
root_folder = os.path.join(database_path, "primary", "kinect", "2_roi_led")
old_word = 'block'
new_word = 'block-order'

# Modify the files
rename_files_in_folder(root_folder, old_word, new_word)
