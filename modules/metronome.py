import matplotlib
import numpy as np
import pygame
import time


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

    '''
         ------------------       public functions       ------------------
    '''
    def start(self):
        self.n_period = 0
        self.prev_id = 0
        self.audio.goCue.play()

    # provides the window's area where to draw the metronome
    #  - vertical
    def initialise(self, ax, gesture, speed, distance, frameHz):
        self.vertical = (gesture == "tap")
        # if vert: do something about vertical/horizontal
        if self.vertical:
            self.x_init, self.y_init = [0.5, 0]
        else:
            self.x_init, self.y_init = [0, 0.5]
        self.__define_step(speed, distance, frameHz)
        self.__init_traj(distance)
        self.__init_ball(ax)

    # return the corresponding id based on the time of the stimulation
    def get_traj_id(self, elapsed_t):
        curr = int(self.frameHz * elapsed_t)  # ms/frame dur
        return curr % self.traj_len

    def update(self, elapsed_t):
        id_curr = self.get_traj_id(elapsed_t)
        self.ball.set_offsets(self.trajectory[id_curr])

        if self.prev_id > id_curr:  # the loop has been made in the traj array
            self.audio.bitMetronome.play()
        elif self.prev_id < self.traj_lenHalf <= id_curr:  # went through half of the period (bouncing back)
            self.audio.bitMetronome.play()

        # if stimulus started, turn on audio metronome
        if elapsed_t > self.audio.durationGo:
            # check if one extremity of the stimulus is reached
            if self.prev_id > id_curr:  # the loop has been made in the traj array
                self.n_period += 1  # increments the number of period made
        self.prev_id = id_curr

        #return self.ball

    # End routine called when the stimulus is finished
    def end(self):
        self.audio.stopCue.play()
        time.sleep(self.audio.durationStop)

    def get_n_period(self):
        return self.n_period

    def get_ball(self):
        return self.ball

    '''
         ------------------       PRIVATE FUNCTIONS       ------------------
    '''
    # provides the window's area where to draw the metronome
    def __init_ball(self, ax):
        #ax.clear()
        self.ball = ax.scatter(self.x_init, self.y_init,
                               s=self.size, lw=0.5,
                               edgecolors=self.edgecolor,
                               facecolors=self.facecolor)
        return self.ball

    # define the trajectory baed on the steps
    def __init_traj(self, distGesture):
        min_val = 0
        max_val = distGesture
        ratio = 1 / (max_val - min_val)  # as it was previously tuned for a full window figure (no subplot)
        half_traj_1 = np.arange(min_val, max_val, self.step / ratio)
        half_traj_2 = np.arange(max_val, min_val, -self.step / ratio)
        traj_move = np.concatenate((half_traj_1, half_traj_2), axis=None)
        self.traj_len = len(traj_move)
        self.traj_lenHalf = int(self.traj_len / 2)

        traj_idle = np.ones(self.traj_len) * (max_val/2)
        if self.vertical:
            self.trajectory = np.c_[traj_idle, traj_move]
        else:
            self.trajectory = np.c_[traj_move, traj_idle]

    # define the steps for the ball
    #  - speed: speed of the stimulus (cm/sec)
    #  - distance: distance to travel on the skin (cm)
    #  - refresh rate of the ball (based on FuncAnimation frame duration)
    def __define_step(self, speed, distance, frameHz):
        self.step = (speed / distance) / frameHz  # pixel/ms