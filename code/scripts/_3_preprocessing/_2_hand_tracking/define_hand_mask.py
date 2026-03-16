import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import QApplication

# Local Imports
# Preserving user's import structure
from preprocessing.motion_analysis import HandMaskSelectorGUI, HandMetadataFileHandler
from utils.should_process_task import should_process_task

# Configure Logging
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
# Default list of excluded vertex IDs to use if the output metadata file does not exist.
# Modify this list as required by specific architectural constraints.
DEFAULT_EXCLUSIONS: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 
    130, 131, 132, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 157, 158, 159, 
    160, 161, 162, 163, 178, 179, 180, 181, 182, 183, 184, 188, 190, 191, 192, 193, 196, 197, 198, 199, 
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 217, 218, 219, 220, 227, 228, 229, 
    230, 231, 232, 233, 234, 235, 236, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 
    250, 251, 252, 253, 254, 255, 256, 257, 259, 262, 264, 265, 266, 267, 268, 269, 
    270, 271, 275, 276, 277, 278, 279, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 356, 357, 358, 359, 
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 
    380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 
    440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 
    460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 
    500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 
    520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 
    560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 
    580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 
    620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 
    640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 
    660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 
    680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 
    700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 
    720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 
    740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 
    760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777
    ] 

def define_hand_mask(
    input_metadata_path: Path,
    hand_models_dir: Path,
    output_metadata_path: Path,
    *,
    force_processing: bool = False
) -> Optional[Path]:
    """
    Launches an interactive GUI to define a mask (vertices to remove) for the hand mesh.
    
    Initialization Logic:
    1. If output_metadata_path exists: Loads existing exclusions from that file.
    2. If not: Loads exclusions from the global DEFAULT_EXCLUSIONS constant.
    3. Input metadata is always read to establish model properties (name, chirality).

    Args:
        input_metadata_path (Path): Path to the baseline metadata JSON file.
        output_metadata_path (Path): Path where the updated metadata will be saved.
        hand_models_dir (Path): Directory containing the hand model files.
        force_processing (bool): If True, ignores timestamp checks.
    
    Returns:
        Path: The path to the output metadata file, or None if failed/cancelled.
    """
    
    # 1. Check task necessity
    # Note: If output exists and is up-to-date, this returns early.
    # To edit an existing mask, force_processing=True must be passed.
    if not should_process_task(
        input_paths=[input_metadata_path],
        output_paths=[output_metadata_path],
        force=force_processing
    ):
        logger.info(f"Skipping: {output_metadata_path} is up to date.")
        return output_metadata_path

    # 2. Load basic info needed for visualization via the FileHandler and Manager from INPUT
    manager = HandMetadataFileHandler.load(input_metadata_path)
    
    if manager is None:
        logger.error(f"Failed to load valid metadata from {input_metadata_path}")
        return None

    # Extract required structural fields from input
    model_name = manager.selected_hand_model_path.name
    is_left = manager.is_left_hand

    if not model_name:
        logger.error("Error: Metadata object is missing required model info.")
        return None

    # 3. Determine Initial Exclusions (The requested modification)
    existing_exclusions: List[int] = []

    if output_metadata_path.exists():
        logger.info(f"Output file detected at {output_metadata_path}. Attempting to load existing exclusions.")
        try:
            # Attempt to load the output file to retrieve the previously saved mask
            output_manager = HandMetadataFileHandler.load(output_metadata_path)
            if output_manager and output_manager.excluded_vertex_ids is not None:
                existing_exclusions = output_manager.excluded_vertex_ids
                logger.info(f"Loaded {len(existing_exclusions)} existing exclusions from output file.")
            else:
                logger.warning("Output file existed but contained no valid manager/exclusions. Reverting to default.")
                existing_exclusions = DEFAULT_EXCLUSIONS
        except Exception as e:
            logger.error(f"Failed to read existing output file: {e}. Reverting to default exclusions.")
            existing_exclusions = DEFAULT_EXCLUSIONS
    else:
        logger.info("No existing output file found. Using default hardcoded exclusions.")
        existing_exclusions = DEFAULT_EXCLUSIONS

    # 4. Setup Application
    # Check if a QApplication instance already exists (common in interactive environments)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 5. Launch GUI
    orientation_str = "left" if is_left else "right"
    logger.info(f"Opening Mask Selector for model: {model_name} ({orientation_str})")
    
    window = HandMaskSelectorGUI(
        hand_models_dir=hand_models_dir,
        model_name=model_name,
        is_left_hand=is_left,
        existing_excluded_indices=existing_exclusions
    )

    # --- Signal Handling Setup ---
    # We use a mutable container to capture the signal data from the inner scope.
    # Initialized to None to distinguish between "No Selection Made/Cancelled" (None) 
    # and "Empty Selection" ([]).
    selection_state: Dict[str, Optional[List[int]]] = {"indices": None}

    def on_validation(indices: List[int]):
        """Slot to capture data when user clicks Validate."""
        selection_state["indices"] = indices
        logger.info(f"Signal received: Validated {len(indices)} vertices for exclusion.")

    # Connect the signal from the GUI to our local slot
    window.selection_validated.connect(on_validation)
    
    window.show()
    
    # Block until the window is closed
    app.exec_()

    # 6. Process Result
    final_indices = selection_state["indices"]

    if final_indices is not None:
        try:
            # Update the manager object state with the new exclusions
            manager.excluded_vertex_ids = final_indices
            
            # Generate the dictionary output using the manager's logic
            output_data = manager.generate_output()
            
            # Ensure the directory for the output file exists
            output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Save using the FileHandler's static method
            HandMetadataFileHandler.save_json(output_data, output_metadata_path)
            
            logger.info(f"Hand mask definition completed successfully. Saved to: {output_metadata_path}")
            return output_metadata_path
            
        except Exception as e:
            logger.error(f"Failed to process and save mask data: {e}", exc_info=True)
            return None
    else:
        logger.warning("Mask selection cancelled or closed without validation.")
        return None

if __name__ == "__main__":
    # Example Usage Configuration
    input_meta = Path(r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/2_processed/kinect/2022-06-17_ST16-05/block-order-01/kinematics_analysis/2022-06-17_ST16-05_semicontrolled_block-order01_kinect_handmodel_stickers_location.json")
    models_dir = Path(r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/handmesh_models")
    output_meta = Path(r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/2_processed/kinect/2022-06-17_ST16-05/block-order-01/kinematics_analysis/2022-06-17_ST16-05_semicontrolled_block-order01_kinect_handmodel_metadata.json")
    
    # Note: To test the 'existing output' logic, force_processing must be True if the file already exists
    # otherwise should_process_task will skip execution.
    define_hand_mask(
        input_metadata_path=input_meta,
        hand_models_dir=models_dir,
        output_metadata_path=output_meta,
        force_processing=True 
    )