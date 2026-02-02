import json
import shutil
import inspect
from pathlib import Path
from typing import Union, Optional

from ..models.color_space_manager import ColorSpaceManager

class ColorSpaceFileHandler:
    """
    A handler class for reading from and writing to colorspace metadata files.
    """
    @staticmethod
    def load(filepath: Union[str, Path], create_from_template_if_missing: bool = False) -> ColorSpaceManager:
        """
        Reads a JSON file and returns a ColorSpaceManager object.

        Args:
            filepath (Union[str, Path]): Path to the metadata file.
            create_from_template_if_missing (bool): If True and filepath does not exist,
                attempts to copy 'colorspace_metadata_template.json' (located 
                alongside ColorSpaceManager) to filepath before loading.

        Returns:
            ColorSpaceManager: The loaded manager object.
        """
        path_obj = Path(filepath)

        if not path_obj.exists() and create_from_template_if_missing:
            # Locate the template alongside color_space_manager.py
            try:
                manager_source_path = Path(inspect.getfile(ColorSpaceManager))
                template_path = manager_source_path.parent / "colorspace_metadata_template.json"

                if template_path.exists():
                    print(f"⚙️  File '{path_obj.name}' missing. Initializing from template...")
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(template_path, path_obj)
                    print(f"✅ Template successfully copied to: {path_obj}")
                else:
                    print(f"⚠️  Template file not found at {template_path}. Cannot create default file.")
            except Exception as e:
                print(f"❌ Error during template initialization: {e}")
                # We do not raise here, we let the subsequent open() fail normally 
                # if the file was not created, to maintain standard error flow.

        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ColorSpaceManager(data)
        except FileNotFoundError:
            print(f"Error: The file at {filepath} was not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {filepath}. Details: {e}")
            raise
    
    @staticmethod
    def write(filepath: Union[str, Path], metadata: ColorSpaceManager):
        """
        Writes a ColorSpace object to a JSON file with pretty-printing.

        Args:
            filepath (str): The path to the file where data will be saved.
            metadata (ColorSpace): The metadata object to serialize.

        Raises:
            IOError: If an error occurs during file writing.
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Use the to_dict() method to get the serializable data
                json.dump(metadata.to_dict(), f, indent=4)
        except IOError as e:
            print(f"Error: Could not write to file at {filepath}. Details: {e}")
            raise