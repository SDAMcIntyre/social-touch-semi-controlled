import sys

# insert at 1, 0 is the script path (or '' in REPL)
import time
from psychopy import visual, core, event
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../modules')
from expert_interface import ExpertInterface


def metronome_call():
    # contact = "tapping"
    contact = "stroking"
    size = "fingertip"
    force = "moderate"
    audio_version = "long"
    m = Metronome(audioFolder="../cues")

    # for speed, duration in [[10, 10], [3, 10], [24, 10]]:
    for speed in [10, 24]:
        print("speed: ", speed, "cm/sec")
        m.init_metronome(speed=speed, contactType=contact)
        m.start_metronome()

    print("metronome_call...end")


def ui_call2():
    folderImg = "../img"
    types = ["tap", "stroke"]
    sizes = ["1f", "2f", "palm"]
    forces = ["light", "medium", "strong"]
    speeds = [1.0, 3.0, 6.0, 9.0, 15.0, 18.0, 21.0, 24.0]  # cm/s

    ui = ExpertInterface(folderImg, types, sizes, forces, speeds, screen=1)
    event.waitKeys()  # press space to continue
    ui.update_block(sizes[2], types[1])
    event.waitKeys()  # press space to continue


def ui_call():
    #gesture = "stroke"
    gesture = "tap"
    size = "1 finger tip"
    force = "moderate"
    audio_version = "long"

    for speed, gesture in [[3,"stroke"], [24,"tap"], [10,"stroke"]]:
        print("gesture[" + gesture + "]\t\t", end="")
        print("size[" + size + "]\t\t", end="")
        print("force[" + force + "]\t\t", end="")
        print("speed[" + str(speed) + "]")
        ei = ExpertInterface(audioFolder="../cues", imgFolder="../img")
        ei.initialise(gesture, size, force, speed)
        ei.start_sequence()
        del ei


if __name__ == '__main__':
    p = 2
    if p == 1:
        metronome_call()
    elif p == 2:
        ui_call()
    elif p == 3:
        w = 10
        h = 10
        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(1, columns * rows + 1):
            img = np.random.randint(10, size=(h, w))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()
