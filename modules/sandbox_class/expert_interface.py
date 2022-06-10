import numpy as np
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time


class ExpertInterface:
    stimArea = None
    metronomeArea = None
    fig = None

    def __init__(self, imgFolder="../img"):
        # gridspec inside gridspec
        self.fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        self.fig.suptitle('Figure suptitle', fontsize='xx-large')

        subfigs = self.fig.subfigures(1, 2, wspace=0.07)
        subfigs[0].set_facecolor('0.75')
        subfigs[0].suptitle('Current Stimulus', fontsize='x-large')

        self.stimArea = subfigs[0].subplots(1, 3, sharey=True)
        for a in self.stimArea:
            a.axis('off')
        self.stimArea[0].imshow(mpimg.imread(imgFolder+'/1f_stroke.png'))
        self.stimArea[1].imshow(mpimg.imread(imgFolder+'/force_light.png'))
        self.stimArea[2].imshow(mpimg.imread(imgFolder+'/palm_stroke.png'))

        right = subfigs[1].subplots(1, 1, sharex=True)
        right.axis('off')

    def getFigure(self):
        return self.fig