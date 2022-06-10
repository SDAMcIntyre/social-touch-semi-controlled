import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import animation, artist
import time
import pygame


class StimulusController:
    gesture: str
    size: str
    force: str
    speed: int = 5  # cm/sec

    distance_gesture: int = 3  # cm
    n_rep_threshold: int = 3  # full period
    min_time: int = 3  # second

    def __init__(self, distance_gesture=3):
        self.min_n_rep = 3  # minimum of period
        self.min_time = 3  # seconds
        self.distance_gesture = distance_gesture

    def isOver(self, n_period_curr, time_curr):
        return time_curr >= self.min_time and n_period_curr >= self.n_rep_threshold

    def defineNewStimulus(self, gesture, size, force, speed):
        self.gesture = gesture
        self.size = size
        self.force = force
        self.speed = speed

    def get_distance_gesture(self):
        return self.distance_gesture


class StimulusDisplay:

    def __init__(self, imgFolder="../img"):
        self.imgFolder = imgFolder

    def init_img(self, imgAxs):
        # nested division
        imgAxs[0].imshow(mpimg.imread(self.imgFolder + '/1f_stroke.png'))
        imgAxs[1].imshow(mpimg.imread(self.imgFolder + '/force_light.png'))
        imgAxs[2].imshow(mpimg.imread(self.imgFolder + '/palm_stroke.png'))


class AudioCues:
    def __init__(self, audioFolder):
        pygame.mixer.pre_init()
        pygame.mixer.init()

        self.audioFolder = audioFolder
        self.soundFileName_base = './' + audioFolder + '/'
        self.goCue = pygame.mixer.Sound(self.soundFileName_base + 'go.wav')
        self.stopCue = pygame.mixer.Sound(self.soundFileName_base + 'stop.wav')
        self.bitMetronome = pygame.mixer.Sound(self.soundFileName_base + 'metronome_bit.mp3')

        self.durationGo = self.goCue.get_length()
        self.durationStop = self.stopCue.get_length()


class Metronome:
    x: float = 0.0
    y: float = 1.0
    x_init: float = 0.0
    y_init: float = 0.5

    size: float = 100
    edgecolor: float(4, ) = (0, 0, 0, 1)
    facecolor: float(4, ) = (0, 0, 0, 1)

    audio: AudioCues
    ball: matplotlib.axes.Axes.scatter

    vertical: bool
    step: float
    prev_id: int
    n_period: int

    frameHz: float

    def __init__(self, audioFolder, frameHz):
        self.audio = AudioCues(audioFolder)  # audio metronome
        self.frameHz = frameHz  # window refresh rate

        self.trajectory = []  # array of dot locations
        self.traj_len = 0  # length of the trajectory array
        self.traj_lenHalf = 0  # half the length to avoid repetitive calculation

        self.vertical = False
        self.step = 0.0
        self.prev_id = 0
        self.n_period = 0

    def start(self):
        self.n_period = 0
        self.prev_id = 0
        self.audio.goCue.play()

    # provides the window's area where to draw the metronome
    #  - vertical
    def init_metronome(self, ax, gesture, speed, distance, frameHz):
        self.vertical = (gesture == "tapping")
        # if vert: do something about vertical/horizontal
        if self.vertical:
            self.x_init, self.y_init = [0.5, 0]
        else:
            self.x_init, self.y_init = [0, 0.5]
        self.define_step(speed, distance, frameHz)
        self.init_traj()

        self.init_ball(ax)

    # provides the window's area where to draw the metronome
    def init_ball(self, ax):
        ax.clear()
        self.ball = ax.scatter(self.x_init, self.y_init,
                               s=self.size, lw=0.5,
                               edgecolors=self.edgecolor,
                               facecolors=self.facecolor)
        return self.ball

    # define the trajectory baed on the steps
    def init_traj(self):
        x_min = -0.05
        x_max = 0.05
        ratio = 1 / (x_max - x_min)  # as it was previously tuned for a full window figure (no subplot)
        half_traj_1 = np.arange(x_min, x_max, self.step / ratio)
        half_traj_2 = np.arange(x_max, x_min, -self.step / ratio)
        traj_move = np.concatenate((half_traj_1, half_traj_2), axis=None)
        self.traj_len = len(self.trajectory)
        self.traj_lenHalf = int(self.traj_len / 2)

        traj_idle = np.zeros(self.traj_len)
        if self.vertical:
            self.trajectory = np.c_[traj_idle, traj_move]
        else:
            self.trajectory = np.c_[traj_move, traj_idle]

    # define the steps for the ball
    #  - speed: speed of the stimulus (cm/sec)
    #  - distance: distance to travel on the skin (cm)
    #  - refresh rate of the ball (based on FuncAnimation frame duration)
    def define_step(self, speed, distance, frameHz):
        self.step = (speed / distance) / frameHz  # pixel/ms

    # return the corresponding id based on the time of the stimulation
    def get_traj_id(self, elapsed_t):
        curr = int(self.frameHz * elapsed_t)  # ms/frame dur
        return curr % self.traj_len

    def update_metronome(self, elapsed_t):
        id_curr = self.get_traj_id(elapsed_t)
        self.ball.set_offsets(self.trajectory[id_curr])

        # if stimulus started, turn on audio metronome
        if elapsed_t > self.audio.durationGo:
            # check if one extremity of the stimulus is reached
            if self.prev_id > id_curr:  # the loop has been made in the traj array
                self.audio.bitMetronome.play()
                self.n_period += 1  # increments the number of period made
            elif self.prev_id < self.traj_lenHalf <= id_curr:  # went through half of the period (bouncing back)
                self.audio.bitMetronome.play()
        self.prev_id = id_curr

    # End routine called when the stimulus is finished
    def end(self):
        self.audio.stopCue.play()
        time.sleep(self.audio.durationStop)

    def get_n_period(self):
        return self.n_period

    def get_ball(self):
        return self.ball


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

        self.stimController = StimulusController()  # stimulus information
        self.images = StimulusDisplay(imgFolder)  # stimulus information
        self.metr = Metronome(audioFolder)  # visual/auditive metronome

        self.init_fig()

        # visual metronome
        self.frameDuration = 1  # milliseconds (width of the visual frame duration)
        self.frameHz = self.sec2ms / self.frameDuration  # Hz (refresh rate of the window)

        self.ani = None  # animation object for visual metronome
        self.artists = []

    '''
         ------------------       public functions       ------------------
    '''
    # initialise the stimulus data, the window and the dot trajectory
    # input:
    #   - speed (cm/sec): speed of the motion
    #   - contactType (String): type of motion (tap/stroking)
    def init_interface(self, gesture, size, force, speed):
        self.stimController.defineNewStimulus(gesture, size, force, speed)
        gestDistance = self.stimController.get_distance_gesture()
        self.clear_figure()
        imgAxs, metrAx = self.init_fig()
        self.images.init_img(imgAxs)
        self.metr.init_metronome(metrAx, gesture, speed, gestDistance, self.frameHz)
        self.artists.append(self.metr.get_ball())

    def start_sequence(self):
        self.ani = animation.FuncAnimation(self.fig, self.__update,
                                           init_func=self.__init_anim,
                                           interval=self.frameDuration,
                                           blit=True)
        plt.show()

    '''
         ------------------       PRIVATE FUNCTIONS       ------------------
    '''
    def __update(self, i):
        elapsed_t = time.time() - self.start_t
        self.metr.update_metronome(elapsed_t)

        # exit condition
        if self.stimController.isOver(self.metr.get_n_period(), elapsed_t):
            print("exit condition raised!")
            self.metr.end()
            self.__stop_interface()
            return []
        else:
            return self.artists

    def __init_anim(self):
        self.fig.canvas.draw_idle()
        self.metr.start()
        self.start_t = time.time()  # sec
        return self.artists

    def __init_fig(self):
        ax = np.zeros(4, dtype=object)

        # gridspec inside gridspec
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        fig.suptitle('Social Touch Semi-controlled', fontsize='xx-large')
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 3], wspace=0.07, hspace=0.1)

        # images divisions
        for s in range(4):
            ax[s] = fig.add_subplot(gs[s])

        ax[3].axis('off')
        self.fig = fig

        return ax[0:2], ax[3]

    def __stop_interface(self):
        self.fig.canvas.draw_idle()
        # self.ani.event_source.stop()
        self.__clear_figure()
        # plt.close('all')
        plt.close()

    def __clear_figure(self):
        self.ani = None  # animation object for visual metronome
        self.artists = []
