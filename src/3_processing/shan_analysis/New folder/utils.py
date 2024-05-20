import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from params import *

def find_consecutive_zeros(data):
    iszero = np.concatenate(([0], np.equal(data, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    diff = ranges[:, 1] - ranges[:, 0]
    if len(diff) <= 0: return None
    range_max = ranges[diff == diff.max()][0]
    if range_max[1] - range_max[0] > len(data) / 2 and range_max[0] < 10:
        return range_max
    else:
        return None

def plot_raw_data(df):
    fig, axes = plt.subplots(7, sharex=True)
    fig.set_size_inches(8, 8, forward=True)
    y_list = ['IFF'] + contact_feat
    for i in range(7):
        sns.lineplot(df, x='t', y=y_list[i], ax=axes[i])

def plot_downsampled_data(df):
    fig, axes = plt.subplots(7, sharex=True)
    fig.set_size_inches(8, 8, forward=True)
    y_list = ['spike_binned'] + contact_feat
    for i in range(7):
        sns.lineplot(df, x='t', y=y_list[i], ax=axes[i])

def psth(df, bin_size):
    df_t = df['t'].values
    df_spike = df['spike'].values
    spike_times = df_t[df_spike == 1]

    n_bins = int(np.ceil(spike_times[-1] / bin_size))
    bins = np.arange(0, round((n_bins + 1) * bin_size, 2), bin_size)
    t = np.arange(bin_size / 2, n_bins * bin_size, bin_size)

    spike_binned, _ = np.histogram(spike_times, bins=bins)
    if len(t) != len(spike_binned):
        t = t[:min(len(t), len(spike_binned))]
        spike_binned = spike_binned[:min(len(t), len(spike_binned))]
    df_new = pd.DataFrame({'t': t, 'spike_binned': spike_binned})

    return df_new

def nerual_psth_smoothed_contact(df, bin_size, norm_method = 'z_score'):

    # plot_raw_data(df)

    # df['t'] = list(np.array(range(len(df.index))) * 0.001)

    df_t = df['t'].values
    df_spike = df['spike'].values
    spike_times = df_t[df_spike == 1]

    n_bins = int(np.ceil(spike_times[-1] / bin_size))
    bins = np.arange(0, round((n_bins + 1) * bin_size, 2), bin_size)
    t = np.arange(bin_size / 2, n_bins * bin_size, bin_size)

    ### neural psth
    spike_binned, _ = np.histogram(spike_times, bins=bins)
    if len(t) != len(spike_binned):
        t = t[:min(len(t), len(spike_binned))]
        spike_binned = spike_binned[:min(len(t), len(spike_binned))]
    df_new = pd.DataFrame({'t': t, 'spike_binned': spike_binned})

    ### contact smooth
    df_contact = df.loc[:, ['t'] + contact_feat]
    for c_i in contact_feat:
        c_smoothed = gaussian_filter1d(np.abs(df_contact[c_i].values), sigma=50)
        f = interp1d(df_contact['t'].values, c_smoothed, fill_value='extrapolate')
        c_downsampled = f(t)
        df_new[c_i] = c_downsampled
    # plot_downsampled_data(df_new)

    ### remove long non-contact intervals
    zeros_idx = find_consecutive_zeros(df_new['spike_binned'])
    if zeros_idx is not None:
        df_new = df_new.drop(range(zeros_idx[-1] - 1))
    df_new = df_new.reset_index()

    ### normalization
    for c_i in contact_feat:
        c_feat = df_new[c_i].values
        upper = np.nanpercentile(c_feat, 98)
        c_feat = np.where(c_feat<=upper,c_feat,upper)
        df_new[c_i] = c_feat

    if norm_method == 'z_score':
        X_norm = StandardScaler().fit_transform(df_new[contact_feat].values)
        y_scaler = StandardScaler()
    if norm_method == 'min_max':
        X_norm = MinMaxScaler().fit_transform(df_new[contact_feat].values)
        y_scaler = MinMaxScaler()
    y = df_new['spike_binned'].values.reshape(-1, 1)
    y_norm = y_scaler.fit_transform(y)
    df_norm = pd.DataFrame(data=X_norm, columns=contact_feat)
    df_norm['spike_binned'] = y_norm
    df_norm['t'] = df_new['t']
    # plot_downsampled_data(df_norm)

    return df_norm, y, y_scaler


if __name__ == '__main__':
    x = find_consecutive_zeros([0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])
    print(x)
