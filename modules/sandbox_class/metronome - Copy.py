import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import animation, artist
import time
import pygame


class DotParam:
    x: float = 0.0
    y: float = 0.5
    size: float = 100
    edgecolor: float(4, ) = (0, 0, 0, 1)
    facecolor: float(4, ) = (0, 0, 0, 1)


class Stimulus:
    speed: int = 5  # cm/sec
    length: int = 3  # cm
    n_rep: int = 3  # full period
    step: float = 0.0  # ratio of the window frame


class Metronome:
    x, y = [0, 1]
    x_init, y_init = [0, 0.5]

    prev_traj_id = 0

    cm2inch = 0.393701
    screen_adj_x = 0.95
    screen_adj_y = 0.95
    frame_x = 18 * cm2inch * screen_adj_x
    frame_y = 3 * cm2inch * screen_adj_y

    start_t: float = 0.0  # time when the stimulus starts
    rep: int = 0  # current number of repetition (end condition)

    countDownDuration = 3.0

    artists = None

    def __init__(self, audioFolder, imgFolder="../img"):
        # gridspec inside gridspec
        self.fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        self.fig.suptitle('Figure suptitle', fontsize='xx-large')

        gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 3], wspace=0.07, hspace=0.1)

        ax = np.zeros(4, dtype=object)
        # images divisions
        outer = gridspec.GridSpec(1, 2, wspace=0.07, hspace=0.1)
        for s in range(4):
            ax[s] = self.fig.add_subplot(gs[s])

        #ax[0] = fig.add_subplot(gs[0, 0])
        #outer[0].set_facecolor('0.75')
        #outer[0].suptitle('Current Stimulus', fontsize='x-large')

        # nested division
        n_img = 3
        inner_left = gridspec.GridSpecFromSubplotSpec(1, n_img, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        for i in range(n_img):
            self.stimArea = plt.subplot(self.fig, inner_left[i])
            self.stimArea.axis('off')
        self.stimArea[0].imshow(mpimg.imread(imgFolder + '/1f_stroke.png'))
        self.stimArea[1].imshow(mpimg.imread(imgFolder + '/force_light.png'))
        self.stimArea[2].imshow(mpimg.imread(imgFolder + '/palm_stroke.png'))

        inner_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        self.metronome_area = plt.subplot(self.fig, inner_right[0])
        self.metronome_area.axis('off')

        # stimulus information
        self.stim = Stimulus()

        # audio metronome
        pygame.mixer.pre_init()
        pygame.mixer.init()
        self.soundFileName_base = './' + audioFolder + '/'
        self.goCue = pygame.mixer.Sound('./' + audioFolder + '/go.wav')
        self.stopCue = pygame.mixer.Sound('./' + audioFolder + '/stop.wav')
        self.bitMetronome = pygame.mixer.Sound('./' + audioFolder + '/metronome_bit.mp3')

        # visual metronome
        self.fig_frameDuration = 1  # width of the visual frame duration (milliseconds)
        self.fig_frameHz = 1000 / self.fig_frameDuration  # refresh rate of the window (Hz)
        self.scat = None  # collection of artists (to make the dot moves)
        self.ani = None  # animation object for visual metronome
        # visual dot variables
        self.dot = DotParam()
        self.trajectory = None  # array of dot locations
        self.traj_len = 0  # length of the trajectory array
        self.traj_lenHalf = 0  # half the length to avoid repetitive calculation

        self._artists = []

    def start_metronome(self):
        self.ani = animation.FuncAnimation(self.fig, self.update,
                                           init_func=self.init_anim,
                                           interval=self.fig_frameDuration,
                                           blit=True)
        self.playGoCue()
        plt.show()
        print("woof?")
        time.sleep(self.countDownDuration)
        print("woof!")

    def update(self, i):
        elapsed_t = time.time() - self.start_t
        curr = int(1000 * elapsed_t / self.fig_frameDuration)  # ms/frame dur
        traj_id = curr % self.traj_len
        self.dot.x = self.trajectory[traj_id]
        self.scat.set_offsets([self.dot.x, self.dot.y])

        # if stimulus started, turn on audio metronome
        if elapsed_t > self.countDownDuration:
            # check if one extremity of the stimulus is reached
            if self.prev_traj_id > traj_id:  # the loop has been made in the traj array
                self.playMetronomeCue()
                self.rep += 1  # increments the number of period made
            elif self.prev_traj_id < self.traj_lenHalf <= traj_id:  # went through half of the period (bouncing back)
                self.playMetronomeCue()
        self.prev_traj_id = traj_id

        # exit condition
        if self.rep == self.stim.n_rep:
            self.playStopCue()
            #self.ani.event_source.stop()
            plt.close()
            print("woof")
            return []

        return self._artists

    def init_anim(self):
        self.rep = 0
        self.prev_traj_id = 0
        self.start_t = time.time()  # sec

        return self._artists

        # init_metronome:

    #   initialise the stimulus data, the window and the dot trajectory
    # input:
    #   - speed (cm/sec): speed of the motion
    #   - contactType (String): type of motion (tap/stroking)
    def init_metronome(self, speed=5, contactType="stroking"):
        if contactType == "tapping":
            vertical = True
            n_rep = 6
        elif contactType == "stroking":
            vertical = False
            n_rep = 3
        else:
            print("metronome.init_metronome: unknown type of contact")
        self.init_stim(speed, n_rep)
        self.init_fig(vertical)
        self.init_traj()

    def init_fig(self, vertical):
        # if vert: do something about vertical/horizontal
        if vertical:
            self.scat = self.metronome_area.scatter(self.x_init, self.y_init,
                                                    s=self.dot.size, lw=0.5,
                                                    edgecolors=self.dot.edgecolor,
                                                    facecolors=self.dot.facecolor)
        else:
            self.scat = self.metronome_area.scatter(self.x_init, self.y_init,
                                                    s=self.dot.size, lw=0.5,
                                                    edgecolors=self.dot.edgecolor,
                                                    facecolors=self.dot.facecolor)
        self._artists.append(self.metronome_area)

    def init_stim(self, speed, n_rep):
        self.stim.speed = speed  # cm/s
        self.stim.n_rep = n_rep  # number of period
        self.stim.step = (self.stim.speed / self.stim.length) / self.fig_frameHz  # pixel/ms

    def init_traj(self):
        x_min = -0.05
        x_max = 0.05
        ratio = 1 / (x_max - x_min)  # as it was previously tuned for a full window figure (no subplot)
        half_traj_1 = np.arange(x_min, x_max, self.stim.step / ratio)
        half_traj_2 = np.arange(x_max, x_min, -self.stim.step / ratio)
        self.trajectory = np.concatenate((half_traj_1, half_traj_2), axis=None)
        self.traj_len = len(self.trajectory)
        self.traj_lenHalf = int(self.traj_len / 2)

    def playGoCue(self):
        return self.goCue.play()

    def playStopCue(self):
        return self.stopCue.play()

    def playMetronomeCue(self):
        return self.bitMetronome.play()
