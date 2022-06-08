import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from audio_management import AudioManager


class Disk:
    x: float = 0.0
    y: float = 0.5
    size: float = 100
    edgecolor: float(4, ) = (0, 0, 0, 1)
    facecolor: float(4, ) = (0, 0, 0, 1)


class Stimulus:
    speed: int = 5  # cm/sec
    length: int = 3  # cm
    duration: int = 3  # seconds
    step: float = 0.0  # ratio of the window frame


class Metronome:
    x, y = [0, 1]
    x_init, y_init = [0, 0.5]

    prev_traj_id = 0

    cm2inch = 0.393701
    screen_adj_x = 0.95
    screen_adj_y = 0.95
    frame_x = 3 * cm2inch * screen_adj_x
    frame_y = 3 * cm2inch * screen_adj_y

    start_t: float = 0.0  # time when the stimulus starts
    end_t: float = 0.0  # time when the stimulus starts

    def __init__(self, audioFolder):
        self.audiom = AudioManager(audioFolder)

        self.dot = Disk()
        self.stim = Stimulus()

        self.trajectory = None
        self.traj_len = 0
        self.traj_lenHalf = 0

        self.fig_frameDuration = 1  # width of the visual frame duration (milliseconds)
        self.fig_frameHz = 1000 / self.fig_frameDuration  # refresh rate of the window (Hz)
        self.fig = None

        self.scat = None
        self.ani = None

    def start_metronome(self):
        self.ani = animation.FuncAnimation(self.fig, self.update,
                                           init_func=self.init_anim,
                                           interval=self.fig_frameDuration)
        soundCh = self.audiom.play()
        time.sleep(self.audiom.currentCue.get_length())
        soundCh = self.audiom.playStopCue()
        plt.show()

    def update(self, i):
        t = time.time()
        elapsed_t = t - self.start_t
        if t > self.end_t:
            self.ani.event_source.stop()
            del self.ani
            plt.close()
        curr = int(1000*elapsed_t/self.fig_frameDuration)  # ms/frame dur
        traj_id = curr % self.traj_len
        self.dot.x = self.trajectory[traj_id]
        # Update the scatter collection with the new position.
        self.scat.set_offsets([self.dot.x, self.dot.y])

        # display short sound (audio metronome)
        if elapsed_t > self.audiom.countDownDuration:
            if self.prev_traj_id > traj_id:  # the loop has been made
                self.audiom.playMetronomeCue()
            elif self.prev_traj_id < self.traj_lenHalf <= traj_id:  # went through half of the period (bouncing back)
                self.audiom.playMetronomeCue()
        self.prev_traj_id = traj_id

    def init_anim(self):
        self.prev_traj_id = 0
        self.start_t = time.time()  # sec
        self.end_t = self.start_t + self.stim.duration + self.audiom.countDownDuration  # sec

    def init_metronome(self, contact, size, force, audio_version, speed=5, vertical=False, duration=3):
        audioFile = contact + "_" + size + "_" + force + "_" + str(speed) + "_" + audio_version
        self.audiom.setSound(audioFile, appendix=".mp3")

        self.init_fig(vertical)
        self.init_stim(speed, duration)
        self.init_traj()

    def init_fig(self, vertical):
        self.fig = plt.figure(figsize=([self.frame_x, self.frame_y]))

        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 1), ax.set_xticks([])
        ax.set_ylim(0, 1), ax.set_yticks([])
        # if vert: do something about vertical/horizontal

        if vertical:
            self.scat = ax.scatter(self.x_init, self.y_init,
                                   s=self.dot.size, lw=0.5,
                                   edgecolors=self.dot.edgecolor,
                                   facecolors=self.dot.facecolor)
        else:
            self.scat = ax.scatter(self.x_init, self.y_init,
                                   s=self.dot.size, lw=0.5,
                                   edgecolors=self.dot.edgecolor,
                                   facecolors=self.dot.facecolor)

    def init_stim(self, speed, duration):
        self.stim.speed = speed  # cm/s
        self.stim.duration = duration  # seconds
        self.stim.step = (self.stim.speed / self.stim.length) / self.fig_frameHz  # pixel/ms

    def init_traj(self):
        half_traj_1 = np.arange(0, 1, self.stim.step)
        half_traj_2 = np.arange(1, 0, -self.stim.step)
        self.trajectory = np.concatenate((half_traj_1, half_traj_2), axis=None)
        self.traj_len = len(self.trajectory)
        self.traj_lenHalf = int(self.traj_len/2)
