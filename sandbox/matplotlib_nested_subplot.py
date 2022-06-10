import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

channelArrangement = [16, 17, 18, 19, 22, 25, 28, 29, 31]

fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

# Left part
i = 0
inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
for c in range(3):
    ax = plt.Subplot(fig, inner[c])
    t = ax.text(0.5, 0.5, 'outer=%d\ncol=%d' % (i, c))
    ax.set_xticks([])
    ax.set_yticks([])
    t.set_ha('center')
    fig.add_subplot(ax)


# Right part
i = 1
inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
ax = plt.Subplot(fig, inner[0])
t = ax.text(0.5, 0.5, 'outer=%d\ncol=%d' % (i, 1))
ax.set_xticks([])
ax.set_yticks([])
t.set_ha('center')
fig.add_subplot(ax)

plt.show()
