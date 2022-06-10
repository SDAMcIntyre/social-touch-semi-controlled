import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# Make a horizontal slider to control the frequency.
ax = plt.axes()
metronome_slider = Slider(
    ax=ax,
    label='Pace',
    valmin=0.1,
    valmax=30,
    valinit=0,
)


def animation_function(i):
    metronome_slider.valinit = i
    metronome_slider.reset()
    return metronome_slider,


animation = FuncAnimation(fig,
                          func=animation_function,
                          frames=np.arange(0, 30, 1),
                          interval=100)
plt.show()