# reporting.py
import logging
import webbrowser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Protocol, Union

class TableRenderer(Protocol):
    """
    Protocol defining the contract for table rendering strategies.
    """
    def render(self, df: pd.DataFrame, title: str = "Data Table") -> None:
        ...

class GreatTablesStrategy:
    """
    Renders tables to static HTML using 'great_tables' (GT).
    Optimized for publication-quality reporting.
    """
    def __init__(self, output_file: Union[str, Path]):
        self.output_file = Path(output_file)
        try:
            from great_tables import GT, style, loc
            self.GT = GT
            self.style = style
            self.loc = loc
            self.available = True
        except ImportError:
            logging.warning("Library 'great_tables' not found. HTML report generation disabled.")
            self.available = False

    def render(self, df: pd.DataFrame, title: str = "Data Table") -> None:
        if not self.available:
            return

        # GT requires a clean DataFrame. 
        try:
            gt_tbl = (
                self.GT(df)
                .tab_header(
                    title=title,
                    subtitle=f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            )
            
            # Apply heatmapping to numeric columns if present
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                gt_tbl = gt_tbl.data_color(
                    columns=numeric_cols,
                    palette=["#ffffff", "#e6f2ff", "#004d99"], # White to Blue
                )

            # Explicit I/O to bypass potential extension registry errors
            html_content = gt_tbl.as_raw_html()
            
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            abs_path = self.output_file.resolve()
            logging.info(f"Report generated at: {abs_path}")
            webbrowser.open(f"file://{abs_path}")
            
        except Exception as e:
            logging.error(f"Failed to render HTML report: {e}")

class TableContext:
    """
    Context manager for executing the selected rendering strategy.
    """
    def __init__(self, strategy: TableRenderer) -> None:
        self._strategy = strategy

    def execute_render(self, df: pd.DataFrame, title: str) -> None:
        self._strategy.render(df, title)