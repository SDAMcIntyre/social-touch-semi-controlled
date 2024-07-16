from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image
import pyglet
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import os

from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402


class SemiControlledDataVisualizer:
    def __init__(self, scd=None, auto_positioning=False):
        # create the figures
        self.fig2D_TTL = DataVisualizer2D(4, "TTLs", auto_positioning=auto_positioning)
        self.fig2D_global = DataVisualizer2D(5, "Neuron, Depth, and Area", auto_positioning=auto_positioning)
        
        self.figpos = DataVisualizer3D("Position", auto_positioning=auto_positioning)

        self.window_positioning()

        if isinstance(scd, SemiControlledData):
            self.update(scd)

    def __del__(self):
        del self.fig2D_TTL, self.fig2D_global, self.figpos

    def window_positioning(self):
        # Function to get screen size
        def get_screen_size():
            display = pyglet.canvas.Display()
            screen = display.get_default_screen()
            return screen.width, screen.height

        screen_width, screen_height = get_screen_size()

        # Set the positions and sizes of the figures
        fig1_width, fig1_height = screen_width // 2, screen_height
        fig2_width, fig2_height = screen_width // 2, screen_height * 3 // 5
        fig3_width, fig3_height = screen_width // 2, screen_height * 2 // 5

        # Helper function to move and resize the figures
        def move_resize_figure(fig, x, y, width, height):
            backend = plt.get_backend()
            if backend == 'TkAgg':
                fig.canvas.manager.window.wm_geometry(f"{width}x{height}+{x}+{y}")
            elif backend == 'Qt5Agg':
                fig.canvas.manager.window.setGeometry(x, y, width, height)
            elif backend == 'WXAgg':
                fig.canvas.manager.window.SetPosition((x, y))
                fig.canvas.manager.window.SetSize((width, height))
            else:
                print(f"Backend {backend} is not supported")

        # Position the figures
        move_resize_figure(self.figpos.fig, 0, 0, fig1_width, fig1_height)
        move_resize_figure(self.fig2D_global.fig, fig1_width, 0, fig2_width, fig2_height)
        move_resize_figure(self.fig2D_TTL.fig, fig1_width, fig2_height, fig3_width, fig3_height)

    def save(self, output_filename_abs):
        # set correct dimensions
        display = pyglet.canvas.Display()
        screen = display.get_default_screen()
        screenratio = screen.width / screen.height
        dpi = 100
        height = 1080 / dpi
        width = screenratio * height
        self.figpos.fig.set_size_inches(width / 2, height)
        self.fig2D_global.fig.set_size_inches(width / 2, height * 2 / 3)
        self.fig2D_TTL.fig.set_size_inches(width / 2, height * 1 / 3)

        # temporarily save the figures
        self.figpos.fig.savefig("fispos_tmp.png")
        self.fig2D_global.fig.savefig("global_tmp.png")
        self.fig2D_TTL.fig.savefig("ttl_tmp.png")

        # Load the temporary images
        figpos = Image.open("fispos_tmp.png")
        fig2D_global = Image.open("global_tmp.png")
        fig2D_TTL = Image.open("ttl_tmp.png")

        # Get the image dimensions
        width1, height1 = figpos.size
        width2, height2 = fig2D_global.size
        width3, height3 = fig2D_TTL.size

        # Create a new image with the combined height of the three images
        combined_image = Image.new('RGB', (width1 + width2, height1))

        # Get the dimensions of each image
        combined_image.paste(figpos, (0, 0))
        combined_image.paste(fig2D_global, (width1, 0))
        combined_image.paste(fig2D_TTL, (width1, height2))

        # Save the combined image
        combined_image.save(output_filename_abs)

        os.remove("fispos_tmp.png")
        os.remove("global_tmp.png")
        os.remove("ttl_tmp.png")


    def set_lim(self, fig_choice, limits):
        match fig_choice:
            case "Depth and Area":
                self.fig2D_global.set_lim(limits)
            case "Position":
                self.figpos.set_lim(limits)

    def update(self, scd: SemiControlledData, title=None):
        time = scd.md.time
        info_str = ("Neuron Info\n"
                    f"ID: {scd.neural.unit_id}\n"
                    f"Type: {scd.neural.unit_type}\n"
                    "Stimulus Info\n"
                    f"Type: {scd.stim.type}\n"
                    f"Force: {scd.stim.force}\n"
                    f"Size: {scd.stim.size}\n"
                    f"Velocity: {scd.stim.vel} cm/s")

        self.fig2D_TTL.update(0, time, scd.neural.TTL, 'TTL Nerve')
        self.fig2D_TTL.update(1, time, scd.contact.TTL, 'TTL contact')
        self.fig2D_TTL.update(2, time, scd.neural.TTL, 'both')
        self.fig2D_TTL.update(2, time, scd.contact.TTL, 'both', linestyle="--", reset=False)
        self.fig2D_TTL.update(3, time, scd.neural.TTL-scd.contact.TTL, 'diff')

        self.fig2D_global.update(0, time, scd.contact.pos_1D, 'Position (Principal Component)', showxlabel=False)
        self.fig2D_global.update(1, time, scd.contact.depth, 'Depth', showxlabel=False)
        self.fig2D_global.update(2, time, scd.contact.area, 'Area size', showxlabel=False)
        self.fig2D_global.update(3, time, scd.neural.iff, 'IFF', showxlabel=False)
        self.fig2D_global.update(4, time, scd.neural.spike, 'Spikes', showxlabel=True)

        self.figpos.update(time, scd.contact.pos, info_str)
        if title is not None:
            self.figpos.fig.suptitle(title)

    def add_vertical_lines(self, xlocs):
        self.fig2D_global.add_vertical_lines(xlocs)

        for ax in self.fig2D_global.axs:
            # Add red vertical lines at specified x-locations
            for xloc in xlocs:
                ax.axvline(x=xloc, color='red', linestyle='--', linewidth=1)


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

    def update(self, ax_idx, time: list[float], data: list[float], data_name, linestyle="-", reset=True, showxlabel=True):
        if reset:
            self.axs[ax_idx].clear()

        # Remove NaNs
        valid = ~(np.isnan(data))
        data = data[valid]
        time = time[valid]  # ensure time array is also filtered

        self.axs[ax_idx].plot(time, data, label=data_name, linestyle=linestyle)
        self.axs[ax_idx].set_title(data_name)
        if showxlabel:
            self.axs[ax_idx].set_xlabel('Time')
        self.axs[ax_idx].set_ylabel(data_name)
        self.axs[ax_idx].legend()

        if self.limits is not None:
            try:
                self.axs[ax_idx].set_ylim(self.limits[ax_idx][0], self.limits[ax_idx][1])
            except:
                pass

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_vertical_lines(self, xlocs):
        pass


class DataVisualizer3D:
    def __init__(self, title, auto_positioning=False):
        self.title = title
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()

        self.fig.show()

        self.limits = None
        self.cbar = None
        self.dot_size = 1

    def __del__(self):
        plt.close(self.fig)

    def set_lim(self, limits):
        self.limits = limits

    def update(self, time, data: list[float], info_str=None, colorsMap='viridis_r', withLine=True):
        if self.fig.texts:
            for text in self.fig.texts:
                text.remove()
        self.ax.clear()

        x, y, z = data

        # Remove NaNs
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x = x[valid]
        y = y[valid]
        z = z[valid]
        time = time[valid]  # ensure time array is also filtered

        # Scatter plot with color gradient based on time
        sc = self.ax.scatter(x, y, z, c=time, cmap=colorsMap, s=self.dot_size)

        if withLine:
            # Create line segments
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize time for color mapping
            norm = plt.Normalize(time.min(), time.max())
            lc = Line3DCollection(segments, cmap=colorsMap, norm=norm)
            lc.set_array(time)
            self.ax.add_collection(lc)

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


def display_scd_one_by_one(scd_list):
    scd_visualizer = SemiControlledDataVisualizer()

    # set up uniform limits to compare the trials
    pos = np.concatenate([s.contact.pos_1D for s in scd_list])
    depth = np.concatenate([s.contact.depth for s in scd_list])
    area = np.concatenate([s.contact.area for s in scd_list])
    limits = [[min(pos), max(pos)], [min(depth), max(depth)], [min(area), max(area)]]
    scd_visualizer.set_lim("Position, Depth and Area", limits)

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


def display_attribute(df, selection=0, title=None):
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
        if title is not None:
            ax.set_title(title, size=label_size)
        else:
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
        if title is not None:
            ax.set_title(title, size=label_size)
        else:
            ax.set_title(current_feature + '_tap', size=label_size)
        ax.set_xlabel('', fontsize=label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    except:
        pass

    plt.ion()
    plt.show(block=False)
    #plt.draw()
