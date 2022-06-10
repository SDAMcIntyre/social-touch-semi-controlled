import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
import time

from modules.metronome import Metronome


class StimulusController:
    gesture: str
    size: str
    force: str
    speed: int = 5  # cm/sec

    distance_gesture: int  # cm
    n_rep_threshold: int  # nb of full period
    min_time: int  # second

    def __init__(self, prep_duration, distance_gesture=3):
        self.min_n_rep = 3  # minimum of period
        self.min_time = 6  # seconds
        self.distance_gesture = distance_gesture
        self.min_time += prep_duration

    def isOver(self, n_period_curr, time_curr):
        return time_curr >= self.min_time and n_period_curr >= self.n_rep_threshold

    def defineNewStimulus(self, gesture, size, force, speed):
        self.gesture = gesture
        self.size = size
        self.force = force
        self.speed = speed

        if gesture == "tap":
            self.n_rep_threshold = 6
        else:
            self.n_rep_threshold = 3

    def get_distance_gesture(self):
        return self.distance_gesture


class StimulusDisplay:
    extension = ".png"

    def __init__(self, imgFolder="../img"):
        self.imgFolder = imgFolder + "/"

    def initialise(self, imgAxs, gesture, size, force, speed):
        # nested division
        self.define_block(imgAxs[0], gesture, size)
        self.define_force(imgAxs[1], force)
        self.define_speed(imgAxs[2], speed)

    def define_block(self, imgAx, gesture, size):
        if size == 'one finger tip':
            size = '1f'
        elif size == 'two finger pads':
            size = '2f'
        else:
            size = 'palm'

        fn = size + "_" + gesture + self.extension
        imgAx.imshow(mpimg.imread(self.imgFolder + fn))

    def define_force(self, imgAx, force):
        fn = "force_" + force + self.extension
        imgAx.imshow(mpimg.imread(self.imgFolder + fn))

    def define_speed(self, imgAx, speed):
        edgecolor: float(4, ) = (0, 0, 0, 1)
        facecolor: float(4, ) = (0, 0, 0, 0.1)
        imgAx.add_patch(mpatches.Circle((0.5, 0.5), 0.5, ec=edgecolor, color=facecolor))
        imgAx.text(0.5, 0.5, str(speed) + "cm/sec", ha="center", va="center", family='sans-serif', size=16)


class ExpertInterface:
    cm2inch = 0.393701
    sec2ms = 1000
    screen_adj_x = 0.95
    screen_adj_y = 0.95
    frame_x = 18 * cm2inch * screen_adj_x
    frame_y = 3 * cm2inch * screen_adj_y

    start_t: float = 0.0  # time when the stimulus starts
    rep: int = 0  # current number of repetition (end condition)

    stimController = StimulusController
    images = StimulusDisplay
    metr = Metronome
    fig = object

    artists = []

    def __init__(self, audioFolder="../cues", imgFolder="../img"):
        self.audioFolder = audioFolder
        self.imgFolder = imgFolder

        # visual metronome
        self.frameDuration = 1  # milliseconds (width of the visual frame duration)
        self.frameHz = self.sec2ms / self.frameDuration  # Hz (refresh rate of the window)

        self.metr = Metronome(audioFolder, self.frameHz)  # visual/auditive metronome
        self.images = StimulusDisplay(imgFolder)  # stimulus information
        self.stimController = StimulusController(self.metr.audio.durationGo)  # stimulus information

        self.imgAxs, self.metrAx = self.__init_fig(self.stimController.get_distance_gesture())

        self.ani = None  # animation object for visual metronome
        self.artists = []

    '''
         ------------------       public functions       ------------------
    '''

    # initialise the stimulus data, the window and the dot trajectory
    # input:
    #   - speed (cm/sec): speed of the motion
    #   - contactType (String): type of motion (tap/stroking)
    def initialise(self, gesture, size, force, speed):
        self.stimController.defineNewStimulus(gesture, size, force, speed)
        gestDistance = self.stimController.get_distance_gesture()
        self.images.initialise(self.imgAxs, gesture, size, force, speed)
        self.metr.initialise(self.metrAx, gesture, speed, gestDistance, self.frameHz)
        self.artists.append(self.metr.get_ball())

    def start_sequence(self):
        self.ani = animation.FuncAnimation(self.fig, self.__update,
                                           init_func=self.__init_anim,
                                           interval=self.frameDuration,
                                           repeat=False,
                                           blit=True)
        self.metr.start()
        plt.show()

    '''
         ------------------       PRIVATE FUNCTIONS       ------------------
    '''

    def __update(self, i):
        elapsed_t = time.time() - self.start_t
        self.metr.update(elapsed_t)
        if self.stimController.isOver(self.metr.get_n_period(), elapsed_t):
            print("exit condition raised!")
            self.metr.end()
            self.__stop_interface()
            return []
        else:
            return self.artists

    # careful! Can be called twice by FuncAnimation
    # see https://stackoverflow.com/questions/49451405/matplotlibs-funcanimation-calls-init-func-more-than-once
    def __init_anim(self):
        self.fig.canvas.draw_idle()
        self.start_t = time.time()  # sec
        return self.artists

    def __init_fig(self, distGesture):
        ax = np.zeros(4, dtype=object)

        # gridspec inside gridspec
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        fig.suptitle('Social Touch Semi-controlled', fontsize='xx-large', ha='right')
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 3], wspace=0.07, hspace=0.1)

        # images divisions

        for s in range(4):
            ax[s] = fig.add_subplot(gs[s])
            if s != 3:
                ax[s].get_xaxis().set_visible(False)
                ax[s].get_yaxis().set_visible(False)
            else:
                ax[s].set_xlim([0, distGesture])
                ax[s].set_ylim([0, distGesture])

        ax[2].set_xlim([0, 1])
        ax[2].set_ylim([0, 1])
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].axis('off')

        self.fig = fig

        return ax[0:2 + 1], ax[3]

    def __stop_interface(self):
        self.fig.canvas.draw_idle()
        self.ani.event_source.stop()
        self.__clear_figure()
        plt.close('all')

        # plt.close()

    def __clear_figure(self):
        self.ani = None  # animation object for visual metronome
        self.artists = []
        for ax in self.fig.get_axes():
            ax.clear()
