import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Fixing random state for reproducibility
np.random.seed(19680801)

x, y = [0, 1]
backward, forward = [0, 1]

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

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
                          ('step', float)
                          ])

# Initialize the raindrops in random positions and with
# random growth rates.
dot['position'] = [0, 0.5]
dot['size'] = 20
dot['edgecolor'] = (0, 0, 0, 1)
dot['facecolor'] = (0, 0, 0, 1)

dot['direction'] = forward

stim['speed'] = 3  # cm/s

stim['length'] = 10  # cm
stim['step'] = stim['speed']/(100*stim['length'])  # pixel/ms


# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(dot['position'][0, x], dot['position'][0, y],
                  s=dot['size'], lw=0.5,
                  edgecolors=dot['edgecolor'],
                  facecolors=dot['facecolor'])


def update(frame_number):
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
animation = FuncAnimation(fig, update, interval=1)
plt.show()
