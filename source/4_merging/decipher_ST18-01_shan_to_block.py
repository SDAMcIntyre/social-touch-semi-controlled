import os
import pandas as pd

dir_path = r"E:\OneDrive - Linköpings universitet\_Teams\touch comm MNG Kinect\basil_tmp\data\semi-controlled\merged\archive\contact_and_neural\new_axes_3Dposition"
filename = "2022-06-22-ST18-unit1-semicontrol.csv"
filename_abs = os.path.join(dir_path, filename)

output_filename_abs = r"E:\OneDrive - Linköpings universitet\_Teams\touch comm MNG Kinect\basil_tmp\data\semi-controlled\ST18_shan_block_link.csv"

df = pd.read_csv(filename_abs)
df_stimuli = df[["block_id", "stimulus", "vel", "finger", "force"]]

unique_rows = []
prev_row = None  # Initialize prev_row to None

for idx, row in df_stimuli.iterrows():
    if prev_row is None or not row.equals(prev_row):
        print(f"({idx}/{len(df_stimuli)}")
        current_row = {
            'block_id': row["block_id"],
            'type': row["stimulus"],
            'velocity': row["vel"],
            'size': row["finger"],
            'force': row["force"],
        }
        unique_rows.append(current_row)
        prev_row = row

final_df = pd.DataFrame(unique_rows)
final_df.to_csv(output_filename_abs, index=False)
