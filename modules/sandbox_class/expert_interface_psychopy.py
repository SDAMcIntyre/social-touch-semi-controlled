import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

from psychopy import visual, core, event


class StimuliInfo:
    types: str(2, )
    sizes: str(3, )
    forces: str(3, )
    speeds: int(8, )


class ContactWindowInfo:
    blocks: float(2, )
    forces: float(2, )
    speeds: float(2, )


class ExpertInterface:
    image_block = None
    image_force = None
    text_speed = None

    # parameter information
    contacts = None
    iLoc = None
    scale = 1

    def __init__(self, folderImg, contact_types, contact_sizes, contact_forces, contact_speeds, screen=0):
        self.win = visual.Window(size=[900, 500], pos=[20, 40], screen=screen)

        self.imgFolder = folderImg

        # basic information about the values of the experiment parameters
        self.contacts = StimuliInfo()
        self.contacts.types = contact_types
        self.contacts.sizes = contact_sizes
        self.contacts.forces = contact_forces
        self.contacts.speed = contact_speeds

        # Location of each paramaters' information
        self.iLoc = ContactWindowInfo()
        self.iLoc.blocks = [-0.5, 0.4]
        self.iLoc.forces = [0.2, 0.4]
        self.iLoc.speeds = [0.75, 0.4]

        img_force = folderImg + "/force_" + self.contacts.forces[0] + ".png"
        text_speed = str(self.contacts.speed[0]) + " cm/s"

        self.update_block(self.contacts.sizes[0], self.contacts.types[0], flip=False)
        self.update_force(img_force, flip=False)
        self.update_speed(text_speed, flip=False)

        self.win.flip()

    def update_block(self, size_str, type_str, flip=True):
        img_path = self.imgFolder + "/" + size_str + "_" + type_str + ".png"
        self.image_block = visual.ImageStim(self.win, image=img_path,
                                            pos=self.iLoc.blocks,
                                            opacity=1.0)
        ratio = self.image_block.size[0] / self.image_block.size[1]
        self.image_block.setSize((self.scale * ratio, self.scale))
        self.image_block.setAutoDraw(True)
        self.image_block.draw()
        if flip:
            self.win.flip()

    def update_force(self, img_path, flip=True):
        self.image_force = visual.ImageStim(self.win, image=img_path,
                                            pos=self.iLoc.forces,
                                            opacity=1.0)
        ratio = self.image_force.size[0] / self.image_force.size[1]
        self.image_force.setSize((self.scale * ratio, self.scale))
        self.image_force.setAutoDraw(True)
        self.image_force.draw()
        if flip:
            self.win.flip()

    def update_speed(self, text, flip=True):
        self.text_speed = visual.TextStim(self.win, text,
                                          pos=self.iLoc.speeds,
                                          opacity=1.0)
        self.text_speed.setAutoDraw(True)
        self.text_speed.draw()
        if flip:
            self.win.flip()

    def flip(self):
        self.win.flip()