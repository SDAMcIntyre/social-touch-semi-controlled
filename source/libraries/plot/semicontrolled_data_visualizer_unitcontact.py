import tkinter as tk
from collections import defaultdict
import numpy as np


class SemiControlledData_VisualizerNeuralContact:
    def __init__(self, root):
        self.root = root
        self.unit_type = ""
        self.contact_attr_names = ""
        self.categorized_contact_attr = ""
        self.categorized_contact_ranges = ""

        root.title("Unit Type and Singular Touches")
        root.geometry("500x400")

        self.unit_type_label = tk.Label(root, text="")
        self.unit_type_label.pack(pady=5)

        self.num_elements_label = tk.Label(root, text="")
        self.num_elements_label.pack(pady=5)

        self.occurrences_text = tk.Text(root, height=20, width=60)
        self.occurrences_text.pack(pady=5)

    def __del__(self):
        self.close_window()

    def set_vars(self,  unit_type, nunit, contact_attr_names, categorized_contact_attr, categorized_contact_ranges):
        self.unit_type = unit_type
        self.nunit = nunit
        self.contact_attr_names = contact_attr_names
        self.categorized_contact_attr = categorized_contact_attr
        self.categorized_contact_ranges = categorized_contact_ranges

    def update_label(self):
        unit_type_str = "Unit Type: " + self.unit_type
        nunit_str = "Number of Unit: " + self.nunit
        num_elements = "Number of Singular Touches: " + str(len(self.categorized_contact_attr))
        self.unit_type_label.config(text="{}\n{}".format(unit_type_str, nunit_str))
        self.num_elements_label.config(text=num_elements)

        # Calculate the occurrences
        self.occurrences_text.delete("1.0", tk.END)
        occurrences = defaultdict(int)
        for attrs in self.categorized_contact_attr:
            occurrences[attrs] += 1
        # Create a list to hold the lines of text to be displayed
        lines = []
        variable_names = ["velocity", "depth", "area"]
        # Add the bin ranges to the lines list
        for i, ranges in enumerate(self.categorized_contact_ranges):
            lines.append(f"Variable {i + 1} Bins (" + variable_names[i] + "):")
            for j, (min_val, max_val) in enumerate(ranges):
                lines.append(f"  Bin {j + 1}: {min_val:.2f} - {max_val:.2f}")
            lines.append("")  # Add a blank line for separation
        # Add the occurrences to the lines list
        lines.append("Occurrences:")
        for key, count in occurrences.items():
            lines.append(f"{key}: {count}")

        for line in lines:
            self.occurrences_text.insert(tk.END, line + "\n")
        self.occurrences_text.config(height=len(lines))

        # Bind the destroy method of the root window to a function that deletes the instance of Application
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)

    def close_window(self):
        try:
            self.root.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
        del self
