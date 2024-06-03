from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tkinter as tk
from tkinter import ttk

from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402


class SemiControlledDataVisualizer:
    def __init__(self, scd=None, plotVel=False, auto_positioning=False):
        # create the figures
        self.fig2D = DataVisualizer2D(2, "Depth and Area", auto_positioning=auto_positioning)
        self.figpos = DataVisualizer3D("Position", auto_positioning=auto_positioning)
        if plotVel:
            self.figvel = DataVisualizer3D("Velocity")

        # (mandatory for the update function) boolean to know if the velocity has to be displayed
        self.plotVel = plotVel

        if isinstance(scd, SemiControlledData):
            self.update(scd)

    def set_lim(self, fig_choice, limits):
        match fig_choice:
            case "Depth and Area":
                self.fig2D.set_lim(limits)
            case "Position":
                self.figpos.set_lim(limits)
            case "Velocity":
                if self.plotVel:
                    self.figvel.set_lim(limits)

    def update(self, scd: SemiControlledData):
        time = scd.md.time
        info_str = ("Stimulus Info\n"
                    f"Type: {scd.stim.type}\n"
                    f"Force: {scd.stim.force}\n"
                    f"Size: {scd.stim.size}\n"
                    f"Velocity: {scd.stim.vel} cm/s")

        self.fig2D.update(0, time, scd.contact.depth, 'Depth')
        self.fig2D.update(1, time, scd.contact.area, 'Area size')
        self.figpos.update(time, scd.contact.pos, info_str)
        if self.plotVel:
            self.figvel.update(time, scd.contact.vel, info_str)


class DataVisualizer2D:
    def __init__(self, nsubplot, title, auto_positioning=False):
        self.title = title
        self.fig, self.axs = plt.subplots(nsubplot, 1, figsize=(10, 12))
        self.fig.tight_layout(pad=5.0)

        if auto_positioning:
            # Get the current figure manager
            manager = plt.get_current_fig_manager()
            # Get the size of the screen
            screen_width, screen_height = manager.window.maxsize()
            # Set the position of the first figure to the right half of the screen
            windows_geometry_depth_n_area = [screen_width // 2, 0, screen_width // 2, 600]
            #if pos is not None:
            #    self.fig.canvas.manager.window.setGeometry(pos[0], pos[1], pos[2], pos[3])

        plt.subplots_adjust(hspace=0.5)
        plt.ion()  # Turn on interactive mode
        self.fig.show()

        self.limits = None

    def __del__(self):
        plt.close(self.fig)

    def set_lim(self, limits):
        self.limits = limits

    def update(self, ax_idx, time: list[float], data: list[float], data_name):
        self.axs[ax_idx].clear()

        self.axs[ax_idx].plot(time, data, label=data_name)
        self.axs[ax_idx].set_title(data_name)
        self.axs[ax_idx].set_xlabel('Time')
        self.axs[ax_idx].set_ylabel(data_name)
        self.axs[ax_idx].legend()

        if self.limits is not None:
            self.axs[ax_idx].set_ylim(self.limits[ax_idx][0], self.limits[ax_idx][1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class DataVisualizer3D:
    def __init__(self, title, auto_positioning=False):
        self.title = title
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()

        #if pos is not None:
        #    self.fig.canvas.manager.window.setGeometry(pos[0], pos[1], pos[2], pos[3])

        self.fig.show()

        self.limits = None
        self.cbar = None
        self.dot_size = 1

    def __del__(self):
        plt.close(self.fig)

    def set_lim(self, limits):
        self.limits = limits

    def update(self, time, data: list[float], info_str=None, colorsMap='viridis_r'):
        if self.fig.texts:
            for text in self.fig.texts:
                text.remove()
        self.ax.clear()

        # Scatter plot with color gradient based on time
        x, y, z = data
        sc = self.ax.scatter(x, y, z, c=time, cmap=colorsMap, s=self.dot_size)

        # Add a color bar
        if self.cbar is None:
            self.cbar = plt.colorbar(sc)
            self.cbar.set_label('Time')
            # Set the ticks location
            tick_locations = np.linspace(np.min(time), np.max(time), num=6)  # Adjust the number of ticks
            self.cbar.set_ticks(tick_locations)

        # Set labels
        # time / color
        tick_str = [f'{tick:.1f}' for tick in np.linspace(np.min(time), np.max(time), num=6)]
        self.cbar.set_ticklabels(tick_str)
        # x y z
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        # title
        self.ax.set_title(self.title)

        # Set limits if global limits have been
        if self.limits is not None:
            self.ax.set_xlim(self.limits[0][0], self.limits[0][1])
            self.ax.set_ylim(self.limits[1][0], self.limits[1][1])
            self.ax.set_zlim(self.limits[2][0], self.limits[2][1])

        if info_str is not None:
            self.fig.text(0, 0.5, info_str, verticalalignment='center', fontsize=10)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def display_one_by_one(scd_list):
    scd_visualizer = SemiControlledDataVisualizer()

    # set up uniform limits to compare the trials
    depth = np.concatenate([s.contact.depth for s in scd_list])
    area = np.concatenate([s.contact.area for s in scd_list])
    limits = [[min(depth), max(depth)], [min(area), max(area)]]
    scd_visualizer.set_lim("Depth and Area", limits)

    pos = np.concatenate([s.contact.pos for s in scd_list], axis=1)
    limits = [(min(_axis), max(_axis)) for _axis in pos]
    scd_visualizer.set_lim("Position", limits)

    vel = np.concatenate([s.contact.vel for s in scd_list], axis=1)
    limits = [(min(_axis), max(_axis)) for _axis in vel]
    scd_visualizer.set_lim("Velocity", limits)

    fig = Figure(figsize=(4, .5), dpi=100)
    current_index = 0

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Dataset Viewer")

    # Define button callback function
    def next_dataset():
        nonlocal current_index
        current_index = (current_index + 1) % len(scd_list)
        scd_visualizer.update(scd_list[current_index])

    # Add a button to the Tkinter window
    button = ttk.Button(root, text="Next", command=next_dataset)
    button.pack(pady=20)

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Initial plot
    scd_visualizer.update(scd_list[current_index])
    # Run the Tkinter main loop
    root.mainloop()


def display_attribute(df, selection=0):
    label_size = 20
    tick_size = 17
    legend_size = 14

    duos = [["estimated_velocity", "expected_velocity"],
            ["estimated_depth", "expected_depth"],
            ["estimated_area", "expected_area"]]
    current_feature = duos[selection][0]
    current_feature_expected = duos[selection][1]

    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
    fig.set_size_inches(8, 5, forward=True)

    # stroke
    try:
        idx_tap = (df['contact_type'].values == "stroke")
        df_current = df[idx_tap]
        ax = axes[0]
        palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
        sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
                     bins=50, palette=palette, multiple="stack", ax=ax)
        ax.set_title(current_feature + '_stroke', size=label_size)
        ax.set_xlabel('', fontsize=label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    except:
        pass

    # tap
    try:
        idx_tap = (df['contact_type'].values == "tap")
        df_current = df[idx_tap]
        ax = axes[1]
        palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
        sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
                     bins=50, palette=palette, multiple="stack", ax=ax)
        ax.set_title(current_feature + '_tap', size=label_size)
        ax.set_xlabel('', fontsize=label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    except:
        pass

