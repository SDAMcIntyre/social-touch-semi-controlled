# file: motion_filter_factory.py
from enum import Enum
from typing import Any, Dict, List, Union

from .motion_filter_interface import MotionFilterInterface
from .butterworth_filter import ButterworthFilter
from .one_euro_filter import OneEuroFilter
from .savgol_filter import SavgolFilter


class FilterChoice(Enum):
    """Enumeration of available motion filter implementations."""

    BUTTERWORTH = "butterworth"
    SAVGOL = "savgol"
    ONE_EURO = "one_euro"

    def __str__(self) -> str:
        return self.value


class MotionFilterFactory:
    """Instantiates motion filters by name, following the XYZExtractorFactory pattern."""

    @staticmethod
    def get_filter(
        choice: Union[FilterChoice, str],
        filter_params: Dict[str, Any] | None = None,
    ) -> MotionFilterInterface:
        """Return an instance of the requested filter.

        Args:
            choice: A :class:`FilterChoice` enum member or its string value
                    (``"butterworth"``, ``"savgol"``, or ``"one_euro"``).
            filter_params: Mapping with keys ``"butterworth"``, ``"savgol"``,
                           and/or ``"one_euro"``, each containing keyword
                           arguments forwarded to the corresponding constructor.
                           Missing keys default to each filter's built-in
                           defaults.

        Raises:
            ValueError: If *choice* is not a recognised filter name.
        """
        if filter_params is None:
            filter_params = {}

        if isinstance(choice, str):
            try:
                choice = FilterChoice(choice)
            except ValueError:
                valid = [e.value for e in FilterChoice]
                raise ValueError(
                    f"Invalid filter choice: '{choice}'. Use one of {valid}."
                ) from None

        if choice is FilterChoice.BUTTERWORTH:
            kwargs = filter_params.get("butterworth", {})
            return ButterworthFilter(**kwargs)

        if choice is FilterChoice.SAVGOL:
            kwargs = filter_params.get("savgol", {})
            return SavgolFilter(**kwargs)

        if choice is FilterChoice.ONE_EURO:
            kwargs = filter_params.get("one_euro", {})
            return OneEuroFilter(**kwargs)

        raise ValueError(f"Unregistered filter: {choice}")  # unreachable safety net

    @staticmethod
    def get_all_filters(
        filter_params: Dict[str, Any] | None = None,
    ) -> List[MotionFilterInterface]:
        """Return one instance of every registered filter (used in compare mode)."""
        return [
            MotionFilterFactory.get_filter(choice, filter_params)
            for choice in FilterChoice
        ]
