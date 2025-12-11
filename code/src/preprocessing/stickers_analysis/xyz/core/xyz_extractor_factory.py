from enum import Enum, auto
from typing import Type, Union

from .xyz_extractor_interface import XYZExtractorInterface

from .xyz_extractor_centroid import CentroidPointCloudExtractor
from .xyz_extractor_roi_centroid import ROICentroidPointCloudExtractor


class ExtractorChoice(Enum):
    """Enumeration of available XYZ extractor types."""

    CENTROID_3D = "centroid"
    ROI_CENTROID_3D = "roi_centroid"
    # To add a new extractor, first add its key here.
    # e.g., SKELETAL_POSE = "skeletal_pose"

    def __str__(self):
        return self.value


class XYZExtractorFactory:
    """Finds and instantiates a specific XYZ extractor based on a choice."""

    # A registry mapping the enum choice to the corresponding extractor class.
    # This makes lookup direct and efficient.
    _EXTRACTOR_REGISTRY: dict[ExtractorChoice, Type[XYZExtractorInterface]] = {
        ExtractorChoice.CENTROID_3D: CentroidPointCloudExtractor,
        ExtractorChoice.ROI_CENTROID_3D: ROICentroidPointCloudExtractor,
        # Add mappings for new extractors here.
    }

    @staticmethod
    def get_extractor(
        choice: Union[ExtractorChoice, str],
        debug: bool = False
    ) -> XYZExtractorInterface:
        """
        Finds and returns an instance of the specified extractor.

        This method acts as a factory, providing the correct extractor object
        based on the provided enum member or its string value.

        Args:
            choice: The desired extractor, specified either as an
                    ExtractorChoice enum member or a matching string
                    (e.g., "centroid").
            debug:  Boolean flag enabling debug mode in the created extractor.

        Returns:
            An instance of the requested XYZ extractor.

        Raises:
            ValueError: If the provided choice is not a valid or registered
                        extractor type.
        """
        try:
            # Normalize the input to an ExtractorChoice enum member.
            # This allows the method to flexibly accept either the enum itself
            # or its string representation.
            if isinstance(choice, str):
                extractor_enum = ExtractorChoice(choice)
            else:
                extractor_enum = choice

            # Look up the corresponding class in the registry.
            extractor_class = XYZExtractorFactory._EXTRACTOR_REGISTRY.get(extractor_enum)

            if extractor_class:
                # Instantiate with the debug flag.
                return extractor_class(debug=debug)
            else:
                raise ValueError(f"Extractor '{choice}' is recognized but not registered with a class.")

        except ValueError:
            # This catches errors from `ExtractorChoice(choice)` for invalid strings.
            valid_choices = [e.value for e in ExtractorChoice]
            raise ValueError(
                f"Invalid extractor choice: '{choice}'. Please use one of {valid_choices}."
            ) from None