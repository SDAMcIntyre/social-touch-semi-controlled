import pandas as pd
import numpy as np

def align_dataframes(
    centers_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    analysis_frame_col: str = 'frame_index',
    analysis_rgb_col: str = 'has_rgb',
    centers_frame_col: str = 'frame'
) -> pd.DataFrame:
    """
    Aligns a "centers" DataFrame with a master "analysis" DataFrame.

    This is a pure function that performs data transformation without any
    file I/O. It assumes centers_df contains data only for valid frames
    and aligns it to the complete set of frames in analysis_df.

    Args:
        centers_df (pd.DataFrame): DataFrame with sticker coordinate data.
        analysis_df (pd.DataFrame): Master DataFrame with analysis for every
                                    frame, including a frame index and a
                                    boolean column indicating data presence.
        analysis_frame_col (str): The name of the frame index column in analysis_df.
        analysis_rgb_col (str): The name of the boolean column in analysis_df
                                indicating if RGB data is present.
        centers_frame_col (str): The name of the original, implicit frame
                                   column in centers_df, which will be dropped.

    Returns:
        pd.DataFrame: A new DataFrame containing the aligned data, including
                      only frames present in centers_df.
        
    Raises:
        ValueError: If the number of rows in centers_df does not match the
                    number of frames marked as True in the analysis_rgb_col.
    """
    # 1. Identify the true frame indices from the analysis file where RGB data exists.
    rgb_frame_indices = analysis_df.loc[analysis_df[analysis_rgb_col] == True, analysis_frame_col]

    # 2. Validate that the counts match.
    if len(centers_df) != len(rgb_frame_indices):
        raise ValueError(
            f"Data mismatch: The number of rows in centers_df ({len(centers_df)}) "
            f"does not match the number of frames with '{analysis_rgb_col}=True' in "
            f"analysis_df ({len(rgb_frame_indices)})."
        )
        
    # 3. Create the mapping by adding the true frame index to the centers_df.
    centers_df_mapped = centers_df.copy()
    centers_df_mapped[analysis_frame_col] = rgb_frame_indices.values

    # 4. Perform an INNER merge to align the data.
    # This ensures only rows with matching frame indices in both dataframes are kept.
    aligned_df = pd.merge(analysis_df, centers_df_mapped, on=analysis_frame_col, how='inner')

    # 5. Clean up the merged DataFrame by dropping the redundant original frame column.
    if centers_frame_col in aligned_df.columns:
        aligned_df = aligned_df.drop(columns=[centers_frame_col])
        
    return aligned_df

def ensure_alignement_stickers_and_depth_data(
    center_csv_path: str,
    mkv_analysis_path: str,
    center_aligned_csv_path: str
) -> str:
    """
    Wrapper function to read data from CSVs, align them, and save the result.

    This function handles the file I/O and uses the align_dataframes
    function to perform the core logic.
    """
    try:
        # Load data from files
        centers_df = pd.read_csv(center_csv_path)
        analysis_df = pd.read_csv(mkv_analysis_path)

        # Call the core logic function
        aligned_data = align_dataframes(centers_df=centers_df, analysis_df=analysis_df)

        # Save the result to a new file
        aligned_data.to_csv(center_aligned_csv_path, index=False)
        
        print(f"Successfully created aligned data at: {center_aligned_csv_path}")
        return center_aligned_csv_path

    except (FileNotFoundError, ValueError) as e:
        print(f"An error occurred during the alignment process: {e}")
        raise
