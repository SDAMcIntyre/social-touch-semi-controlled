import tkinter as tk
from tkinter import ttk

from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402


class SemiControlledValidatorApp:
    def __init__(self, scd: SemiControlledData, scd_visualiser:  SemiControlledDataVisualizer):
        # display the current Semi Controlled Data
        scd_visualiser.update(scd)

        self.root = tk.Tk()
        self.root.geometry("300x200")  # Width x Height
        self.root.title("Signal Validator")

        self.color_yes = 'green'
        self.color_no = 'red'
        self.text_yes = 'Valid Signal'
        self.text_no = 'Not Valid Signal'

        # Initial validation state (1 for YES, 0 for NO)
        self.signal_valid = tk.BooleanVar(value=True)
        self.current_color = self.color_yes
        self.current_text = self.text_yes

        # Button to toggle the signal validity
        self.toggle_button = tk.Button(self.root, text=self.current_text, bg=self.current_color, command=self.toggle_signal)
        self.toggle_button.pack(pady=10)

        # Button to confirm and close the interface
        self.confirm_button = ttk.Button(self.root, text="Confirm", command=self.confirm)
        self.confirm_button.pack(pady=10)

        # Label to display the estimated duration and expected value
        recording_ms = 1000 * (scd.md.time[-1] - scd.md.time[0])
        expected_ms = 1000 * scd.stim.get_singular_contact_duration_expected()
        info_text = "Recorded Duration: {0} ms\nExpected Value: {1}ms".format(recording_ms, expected_ms)

        self.info_label = ttk.Label(self.root, text=info_text)
        self.info_label.pack(pady=10)

    def toggle_signal(self):
        if self.current_color == self.color_yes:
            self.current_color = self.color_no
            self.current_text = self.text_no
            self.signal_valid.set(False)
        else:
            self.current_color = self.color_yes
            self.current_text = self.text_yes
            self.signal_valid.set(True)

        self.update_button()

    def update_button(self):
        self.toggle_button.config(bg=self.current_color, text=self.current_text)

    def confirm(self):
        # Print the value to the console (or handle it as needed)
        print("Signal Valid:", self.signal_valid.get())
        self.close_window()

    def __del__(self):
        self.close_window()

    def close_window(self):
        try:
            self.root.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
