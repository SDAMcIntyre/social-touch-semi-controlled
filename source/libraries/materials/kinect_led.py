import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup


# class of the LED green value of videos for the preprocessing
class SemiControlledDataLED:
    def __init__(self):
        self.session = []
        self.block_id = []
        self.file_path = []
        self.timeseries_filename = []
        self.metadata_filename = []

        self.time = []
        self.green_levels = []
        self.led_on = []

    def load_timeseries(self, led_files_info):
        self.session = led_files_info["session"]
        self.block_id = led_files_info["block_id"]
        self.file_path = led_files_info["file_path"]
        self.timeseries_filename = led_files_info["timeseries_filename"]
        self.metadata_filename = led_files_info["metadata_filename"]

        df = pd.read_csv(os.path.join(led_files_info["file_path"], led_files_info["timeseries_filename"]))
        df.dropna(inplace=True)  # remove lines that contains NaN values
        self.time = [round(num, 5) for num in df["time (second)"].values]
        self.green_levels = [round(num, 5) for num in df["green level"].values]
        self.led_on = df["LED on"].values

    def load_class_list_from_infos(self, led_files_info_list):
        data_led_list = []

        for led_files_info in led_files_info_list:
            scdl = SemiControlledDataLED()
            scdl.load_timeseries(led_files_info)
            data_led_list.append(scdl)

        return data_led_list

    def resample(self, new_time, show=False):
        if show:
            plt.figure()
            plt.plot(self.time, self.green_levels)
            plt.plot(self.time, self.led_on)

        # reset potential non start to zero.
        new_time = new_time - new_time[0]

        # Find the index of the first 1
        start = 0
        while start < len(self.led_on) and self.led_on[start] == 0:
            start += 1
        # Find the index of the last 1
        end = len(self.led_on) - 1
        while end >= 0 and self.led_on[end] == 0:
            end -= 1
        time_essential = self.time[start:end] - self.time[start]

        # display basic info
        print("start/end: original:({:.2f}, {:.2f}), essential:({:.2f}, {:.2f}), target:({:.2f}, {:.2f})"
              .format(self.time[0], self.time[-1], time_essential[0], time_essential[-1], new_time[0], new_time[-1]))
        print("nb. element ---> original:{:.2f}, target:{:.2f}, ratio:{:.6f} (expected ~0.03)".format(len(self.time), len(new_time), len(self.time)/len(new_time)))

        # /!\
        # /!\ FOR SOME REASON, TIME DON'T MATCH BETWEEN SHAN'S CSV AND KINECT VIDEO (MY EXTRACTION)
        # /!\
        # Artificially make them match
        # could include a PB bc the rows with NaN values have been removed during load_dataframe()
        new_time = np.linspace(self.time[0], self.time[-1], len(new_time))

        # Create interpolation functions
        interp_func_led_on = interp1d(self.time, self.led_on, kind='nearest')  # 'linear', 'quadratic', 'cubic'
        interp_func_green_levels = interp1d(self.time, self.green_levels, kind='linear')  # 'linear', 'quadratic', 'cubic'

        # Interpolate the values at the new time points
        self.led_on = interp_func_led_on(new_time)
        self.green_levels = interp_func_green_levels(new_time)

        # replace the old time vector by the new one
        self.time = new_time

        if show:
            plt.figure()
            plt.plot(self.time, self.green_levels)
            plt.plot(self.time, self.led_on)
            plt.show()
            WaitForButtonPressPopup()


