# src/data_manager.py
# src/data_manager.py

import pandas as pd
from pathlib import Path
from filelock import FileLock
from typing import List, Optional

# Imports for Excel formatting
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles.differential import DifferentialStyle


class DataManager:
    """
    Handles concurrency-safe reading and writing of the pipeline status to a
    formatted XLSX file. It applies conditional coloring only if a cell's text
    starts with a specific status keyword ('SUCCESS', 'FAILURE', etc.).
    """
    def __init__(self, report_file_path: Path):
        self.report_file_xlsx = report_file_path.with_suffix('.xlsx')
        self.report_file_csv = report_file_path.with_suffix('.csv') # For legacy migration
        self.lock_file = self.report_file_xlsx.with_suffix('.xlsx.lock')
        self.lock = FileLock(self.lock_file)
        self.report_file_xlsx.parent.mkdir(parents=True, exist_ok=True)

        self.STATUS_COLORS = {
            'SUCCESS': 'C6EFCE',  # Light Green
            'FAILURE': 'FFC7CE',  # Light Red
            'RUNNING': 'FFEB9C',  # Light Yellow
            'PENDING': 'D0D0D0',  # Gray
        }

    def _write_formatted_excel(self, df: pd.DataFrame):
        """
        Writes the DataFrame to an XLSX file and applies conditional formatting
        based on the starting text of the cells using the modern FormulaRule API.
        """
        with pd.ExcelWriter(self.report_file_xlsx, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Pipeline Status')
            worksheet = writer.sheets['Pipeline Status']

            if df.empty:
                return

            # Auto-adjust column widths
            for idx, col in enumerate(df.columns, 1):
                series = df[col]
                # Ensure series is not empty and handle NaN values
                if not series.empty:
                    max_len = max(
                        series.astype(str).map(len).max(),
                        len(str(series.name))
                    ) + 2
                    worksheet.column_dimensions[get_column_letter(idx)].width = max_len

            # Define the cell range (all columns except 'dataset')
            num_rows = len(df) + 1
            num_cols = len(df.columns)
            format_range = f"B2:{get_column_letter(num_cols)}{num_rows}"

            # Sort statuses to ensure longer matches are checked first (e.g., 'FAILURE' before 'FAIL')
            sorted_statuses = sorted(self.STATUS_COLORS.keys(), key=len, reverse=True)
            
            # *** MODIFIED SECTION: Apply conditional formatting using FormulaRule ***
            for status in sorted_statuses:
                color_hex = self.STATUS_COLORS[status]
                
                # 1. Define the cell fill and the differential style
                fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type='solid')
                dxf = DifferentialStyle(fill=fill)
                
                # 2. Define the formula. SEARCH is case-insensitive.
                # It checks if the status string is found at the beginning of the cell content.
                formula = [f'SEARCH("{status}", B2)=1']
                
                # 3. Create the rule object, assign the formatting, and add it
                rule = FormulaRule(formula=formula, stopIfTrue=True)
                rule.dxf = dxf
                worksheet.conditional_formatting.add(format_range, rule)

    def _read_report_unsafe(self) -> pd.DataFrame:
        """
        Reads the report file without a lock, prioritizing .xlsx and falling back to .csv.
        """
        if self.report_file_xlsx.exists():
            try:
                return pd.read_excel(self.report_file_xlsx, engine='openpyxl')
            except Exception:
                pass # Fallback to CSV if XLSX is corrupt or invalid

        if self.report_file_csv.exists():
            print(f"Legacy CSV report found. Migrating to '{self.report_file_xlsx}'.")
            return pd.read_csv(self.report_file_csv)
            
        return pd.DataFrame()

    def initialize(self, columns: List[str]):
        """Initializes an empty report file."""
        header = ['dataset'] + columns
        with self.lock:
            if not self.report_file_xlsx.exists() and not self.report_file_csv.exists():
                df = pd.DataFrame(columns=header)
                self._write_formatted_excel(df)
                print(f"📊 Report initialized at: {self.report_file_xlsx}")

    def update(self, dataset: str, event: str, status: str, message: str = "") -> Optional[pd.DataFrame]:
        """Updates the status and rewrites the formatted XLSX file."""
        value = f"{status}: {message}" if message else status
        
        try:
            with self.lock:
                df = self._read_report_unsafe()
                
                if df.empty and 'dataset' not in df.columns:
                        print(f"🚨 CRITICAL: Report not initialized. Call 'initialize' first.")
                        return None

                if event not in df.columns:
                    print(f"⚠️ Warning: Event '{event}' not found in columns. Ignoring.")
                    return df

                if dataset in df['dataset'].values:
                    df.loc[df['dataset'] == dataset, event] = value
                else:
                    new_row = {'dataset': dataset, event: value}
                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)

                self._write_formatted_excel(df)
                
                # If a legacy CSV exists, remove it after successful XLSX write
                if self.report_file_csv.exists():
                    self.report_file_csv.unlink()
                    
                return df
        except Exception as e:
            print(f"🚨 CRITICAL: Failed to update report file {self.report_file_xlsx}. Error: {e}")
            return None
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Safely reads the report file and returns it as a DataFrame."""
        with self.lock:
            return self._read_report_unsafe()


if __name__ == "__main__":
    import openpyxl
    from openpyxl.styles import PatternFill
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles.differential import DifferentialStyle

    # 1. Prepare data and create a new workbook
    data = [
        "SUCCESS: System boot complete.",
        "FAIL: Network connection timed out.",
        "FAIL: Database integrity check failed.",
        "SUCCESS: All services are running.",
        "SUCCESS: Data processed without errors.",
    ]
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Status Report"

    ws.append(["Log Messages"])
    for item in data:
        ws.append([item])

    # 2. Define the cell fills and the differential styles
    green_dxf = DifferentialStyle(fill=PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"))
    red_dxf = DifferentialStyle(fill=PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"))

    # 3. Define the formula-based rules
    data_range = f"A2:A{len(data)+1}"
    success_formula = ['SEARCH("SUCCESS", A2)=1']
    fail_formula = ['SEARCH("FAIL", A2)=1']

    # --- KEY CHANGE IS HERE ---
    # First, create the rule with its formula
    success_rule = FormulaRule(formula=success_formula, stopIfTrue=True)
    # Then, assign the formatting (dxf) to the rule object
    success_rule.dxf = green_dxf

    # Do the same for the "FAIL" rule
    fail_rule = FormulaRule(formula=fail_formula, stopIfTrue=True)
    fail_rule.dxf = red_dxf

    # Now, add the fully constructed rule objects to the worksheet
    ws.conditional_formatting.add(data_range, success_rule)
    ws.conditional_formatting.add(data_range, fail_rule)

    # 4. Save the file
    output_filename = "status_report_final.xlsx"
    wb.save(output_filename)

    print(f"✅ Successfully created '{output_filename}'. This should resolve the TypeError.")