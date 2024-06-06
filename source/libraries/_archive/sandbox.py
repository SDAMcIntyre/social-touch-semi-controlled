
import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt

def load_signals(neuronal_path, other_signal_path):
    # Charger les signaux à partir des fichiers (à adapter selon le format des fichiers)
    neuronal_signal = pd.read_csv(neuronal_path).values.flatten()
    other_signal = pd.read_csv(other_signal_path).values.flatten()
    return neuronal_signal, other_signal

def convert_position_to_velocity(position_signal, sampling_rate):
    # Calculer la vitesse à partir de la position (dérivée première)
    velocity = np.diff(position_signal) * sampling_rate
    return velocity

def compute_cross_correlation(signal1, signal2):
    # Calculer la corrélation croisée
    correlation = correlate(signal1, signal2, mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal1))
    return correlation, lags

def find_best_lag(correlation, lags):
    # Trouver le décalage qui maximise la corrélation
    best_lag = lags[np.argmax(correlation)]
    return best_lag

def adjust_signal(signal, lag):
    # Ajuster le signal neuronal en fonction du décalage
    if lag > 0:
        adjusted_signal = np.pad(signal, (lag, 0), 'constant')[:len(signal)]
    elif lag < 0:
        adjusted_signal = np.pad(signal, (0, -lag), 'constant')[-lag:]
    else:
        adjusted_signal = signal
    return adjusted_signal

def main(neuronal_path, other_signal_path, signal_type, sampling_rate=1000):
    neuronal_signal, other_signal = load_signals(neuronal_path, other_signal_path)

    if signal_type == 'caresse':
        other_signal = convert_position_to_velocity(other_signal, sampling_rate)

    correlation, lags = compute_cross_correlation(neuronal_signal, other_signal)
    best_lag = find_best_lag(correlation, lags)

    adjusted_neuronal_signal = adjust_signal(neuronal_signal, best_lag)

    # Affichage des résultats
    plt.figure()
    plt.plot(neuronal_signal, label='Neuronal Signal')
    plt.plot(adjusted_neuronal_signal, label='Adjusted Neuronal Signal')
    plt.legend()
    plt.show()

    print(f"Best lag: {best_lag}")

# Exemple d'utilisation :
# main('neuronal_signal.csv', 'other_signal.csv', 'caresse')

# Pour le signal de tapping, on utiliserait :
# main('neuronal_signal.csv', 'other_signal.csv', 'tapping')






### qs
import seaborn as sns
import matplotlib.pyplot as plt

label_size = 20
tick_size = 17
legend_size = 14


choice = 2
duos = [["estimated_velocity", "expected_velocity"],
        ["estimated_depth", "expected_depth"],
        ["estimated_area", "expected_area"]]
current_feature = duos[choice][0]
current_feature_expected = duos[choice][1]

fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
fig.set_size_inches(8, 5, forward=True)

# stroke
idx_tap = (df['contact_type'].values == "stroke")
df_current = df[idx_tap]
ax = axes[0]
palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
             bins=50, palette=palette, multiple="stack", ax=ax)
ax.set_title(current_feature + '_stroke', size=label_size)
ax.set_xlabel('', fontsize=label_size)
ax.yaxis.label.set_size(label_size)
ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
ax.yaxis.set_tick_params(labelsize=tick_size)


# tap
idx_tap = (df['contact_type'].values == "tap")
df_current = df[idx_tap]
ax = axes[1]
palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
             bins=50, palette=palette, multiple="stack", ax=ax)
ax.set_title(current_feature + '_tap', size=label_size)
ax.set_xlabel('', fontsize=label_size)
ax.yaxis.label.set_size(label_size)
ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
ax.yaxis.set_tick_params(labelsize=tick_size)






[velocity_cm, depth_cm, area_cm] = scdm.estimate_contact_averaging()
[contact_types, velocity_cm_exp, depth_cm_exp, area_cm_exp] = scdm.get_contact_expected()
idx_tap = [index for index, value in enumerate(contact_types) if value == 'tap']
idx_stroke = [index for index, value in enumerate(contact_types) if value == 'stroke']
depth_cm_exp[idx_tap].unique()



df_plot = df_contact[df_contact['stimulus'].values == stim]


fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
fig.set_size_inches(8, 5, forward=True)
for stim in ['stroke', 'tap']:
    ax = axes[0 if stim == 'stroke' else 1]
    df_plot = df_contact[df_contact['stimulus'].values == stim]
    palette = sns.color_palette('Set2', n_colors=len(df_plot[contact_conditions[c_feat]].unique()))
    sns.histplot(df_plot, x=c_feat, hue=contact_conditions[c_feat],
                 bins=50, palette=palette, multiple="stack", ax=ax)
    ax.set_title(c_feat + '_' + stim, size=label_size)
    ax.set_xlabel('', fontsize=label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
    ax.yaxis.set_tick_params(labelsize=tick_size)
    if stim == 'tap':
        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels_ = [t.get_text() for t in old_legend.get_texts()]
        labels = ['group ' + labels_[i] + ': ' + vel_ranges[i] for i in range(len(labels_))]
        title = old_legend.get_title().get_text()
        ax.legend(handles, labels, title=title, frameon=False, loc=0, fontsize=legend_size, title_fontsize=legend_size, handletextpad=0.3,
                  markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
    else:
        ax.legend_.remove()

plt.tight_layout()




# automatic filtering
# 1. Based on the duration
duration_recorded = []
duration_expected = []
for scd in scdm.data_periods:
    duration_recorded.append(1000 * (scd.md.time[-1] - scd.md.time[0]))
    duration_expected.append(1000 * scd.stim.get_singular_contact_duration_expected())
# Extract the indices of elements that is below the boundary
incorrect_idx = [idx for idx, dur in enumerate(duration_recorded) if dur < 50]
signal_valid_list[incorrect_idx] = False

plt.hist(duration_recorded)
plt.hist(duration_recorded, bins=200, edgecolor='black')













# define the euclidian distance based on the first location
pos_eucl = np.sqrt(np.sum(scd.contact.pos[0:1, :] ** 2, axis=0))
# detrend the signal to avoid any artifact during the smoothing
pos_eucl_det = signal.detrend(pos_eucl)
# smoothing
pos_eucl_det_smooth = self.get_smooth_signal(pos_eucl_det, scd, nframe=5, method="blind")

# define the minimum distance between expected period/stimulation
nb_period_expected = scd.stim.get_n_period_expected()
# stroke: two stimulations are contained per period
nb_stimulation_expected = 2 * nb_period_expected
min_dist_peaks = .6 * scd.md.nsample/nb_stimulation_expected

# find peaks
pos_peaks, _ = signal.find_peaks(abs(pos_eucl_det_smooth), distance=min_dist_peaks)
fig, ax = plt.subplots(1, 1)
plt.plot(pos_eucl_det_smooth, label='Signal')
plt.plot(pos_peaks, pos_eucl_det_smooth[pos_peaks], 'r.', label='Peaks')
ax.set_title("Found peaks on smoothed signal")

scd_period_list = []
if show:
    visualiser = SemiControlledDataVisualizer()
for i in range(len(pos_peaks) - 1):
    interval = np.arange(1 + pos_peaks[i], pos_peaks[i + 1], dtype=int)
    scd_interval = scd.get_data_idx(interval)
    scd_period_list.append(scd_interval)
    if show:
        visualiser.update(scd_interval)
        #time.sleep(.5)
        WaitForButtonPressPopup()





p = scd.contact.pos


p_updated = p
nsample = np.shape(p_updated)[1]
xyz_mean = np.mean(p_updated, axis=1)
p_updated = p - np.transpose(np.tile(xyz_mean, (nsample, 1)))

weights = np.repeat(1.0, 7) / 7
p_updated[0,:] = np.convolve(p_updated[0,:], weights, 'same')
p_updated[1,:] = np.convolve(p_updated[1,:], weights, 'same')
p_updated[2,:] = np.convolve(p_updated[2,:], weights, 'same')

plt.plot(p[0,:])
plt.plot(p[1,:])
plt.plot(p[2,:])
plt.plot(p_updated[0,:])
plt.plot(p_updated[1,:])
plt.plot(p_updated[2,:])






##
from numpy.fft import fft
import math

p_eucl = [sum(np.square(coord)) for coord in np.transpose(p_updated)]

window_size = int(Fs/30 * 1)
weights = np.repeat(1.0, window_size) / window_size
p_updated_smooth = np.empty(p_updated.shape)
p_updated_smooth[0,:] = np.convolve(p_updated[0,:], weights, 'same')
p_updated_smooth[1,:] = np.convolve(p_updated[1,:], weights, 'same')
p_updated_smooth[2,:] = np.convolve(p_updated[2,:], weights, 'same')
p_eucl_smooth = [np.sqrt(sum(np.square(coord))) for coord in np.transpose(p_updated_smooth)]
plt.plot(p_eucl_smooth)


window_size = 1000
weights = np.repeat(1.0, window_size) / window_size
for idx in [0,1,2]:
    t = scd.md.time
    x = p_updated[idx,:]
    x = np.convolve(x, weights, 'same')

    Fs = 1/np.mean(np.diff(t))

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/Fs
    freq = n/T
    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)

    plt.subplot(122)
    plt.plot(t, x, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()




fft_result = fft(p_updated)
frequencies = np.fft.fftfreq(nsample)
for i in range(3):
    plt.plot(frequencies, np.abs(fft_result[i,:]), label=f'Row {i + 1}')

plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.legend()



##

p = scd.contact.pos

p_updated = p
nsample = np.shape(p_updated)[1]
xyz_mean = np.mean(p_updated, axis=1)
p_updated = p - np.transpose(np.tile(xyz_mean, (nsample, 1)))

fft_result = np.fft.fft2(p_updated)
frequencies = np.fft.fftfreq(nsample)
for i in range(3):
    plt.plot(frequencies, np.abs(fft_result[i]), label=f'Row {i + 1}')

plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.legend()

scd.contact.pos = p_updated
SemiControlledDataVisualizer(scd)



























