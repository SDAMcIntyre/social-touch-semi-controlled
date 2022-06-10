import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

x, y = [0, 1]
backward, forward = [0, 1]

# Create rain data
dot = np.zeros(1, dtype=[('position', float, (2,)),
                         ('size', float),
                         ('growth', float),
                         ('edgecolor', float, (4,)),
                         ('facecolor', float, (4,)),
                         ('direction', int)
                         ])

stim = np.zeros(1, dtype=[('speed', int),
                          ('length', int),
                          ('step', float),
                          ('start_t', float),
                          ('time', float),
                          ('frameDuration', int),
                          ('frameHz', float),
                          ('traj_len', int),
                          ])



# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

# Initialize the raindrops in random positions and with
# random growth rates.
dot['position'] = [0, 0.5]
dot['size'] = 20
dot['edgecolor'] = (0, 0, 0, 1)
dot['facecolor'] = (0, 0, 0, 1)
dot['direction'] = forward

stim['speed'] = 3  # cm/s
stim['length'] = 10  # cm

stim['frameDuration'] = 1  # milliseconds
stim['frameHz'] = 1000/stim['frameDuration']  # hz

stim['step'] = (stim['speed']/stim['length']) / stim['frameHz']  # pixel/ms
stim['time'] = time.time()  # cm

# define in advance the location of the dot in function of the visual frame duration
half_traj_1 = np.arange(0, 1, stim['step'])
half_traj_2 = np.arange(1, 0, -stim['step'])
trajectory = np.concatenate((half_traj_1, half_traj_2), axis=None)

stim["traj_len"] = len(trajectory)

print("frameDuration: ", stim['frameDuration'], "ms")
print("frameHz: ", stim['frameHz'], "Hz")
print("step: ", stim['step'])
print(len(trajectory))

# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(dot['position'][0, x], dot['position'][0, y],
                  s=dot['size'], lw=0.5,
                  edgecolors=dot['edgecolor'],
                  facecolors=dot['facecolor'])


def init():
    stim['start_t'] = time.time()  # sec
    stim['time'] = time.time()  # sec


def update(frame_number):
    curr = int(1000*(time.time()-stim['start_t']) / stim['frameDuration']) % stim["traj_len"]
    #print(curr)

    s = stim['step']
    if dot['direction'] == forward:
        dot['position'][0, x] += s
        if (dot['position'][0, x] + s) >= 1:
            dot['direction'] = backward
    else:
        dot['position'][0, x] -= s
        if (dot['position'][0, x] - s) <= 0:
            dot['direction'] = forward

    # Update the scatter collection with the new position.
    scat.set_offsets(dot['position'])


# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, init_func=init, interval=stim['frameDuration'])
prev = time.time()
plt.show()
