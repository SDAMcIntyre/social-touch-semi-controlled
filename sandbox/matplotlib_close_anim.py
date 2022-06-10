import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

grid_size = [10, 10]
grid = np.random.randint(low=0, high=256, size=grid_size, dtype=np.uint8)
im = ax.imshow(grid, cmap='gray', vmin=0, vmax=255)


# Animation settings
def animate(frame):
    if frame == FRAMES_NUM:
        print(f'{frame} == {FRAMES_NUM}; closing!')
        plt.close(fig)
    else:
        print(f'frame: {frame}') # Debug: May be useful to stop
        grid = np.random.randint(low=0, high=256, size=grid_size, dtype=np.uint8)
        im.set_array(grid)


INTERVAL = 100
FRAMES_NUM = 10

anim = animation.FuncAnimation(fig, animate, interval=INTERVAL,
                               frames=FRAMES_NUM+1, repeat=False)

plt.show()

print("error")