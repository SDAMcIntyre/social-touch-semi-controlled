import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from numpy.linalg import svd
import scipy
from scipy import interpolate, stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from varname import nameof

from utils import nerual_psth_smoothed_contact
from params_semi_control_IFF import *
from semi_control_IFF import adapt_contact_condition


def stars(p):
  if p < 0.0001:
      return "****"
  elif (p < 0.001):
      return "***"
  elif (p < 0.01):
      return "**"
  elif (p < 0.05):
      return "*"
  else:
      return "-"

def combine_neural_contact_data(new_axes = False, per_repeat=False):
    subtype_dict = {'ST13-unit1': 'SAII', 'ST13-unit2': 'CT', 'ST13-unit3': 'HFA',
                    'ST14-unit1': 'SAI', 'ST14-unit2': 'HFA', 'ST14-unit3': 'SAII', 'ST14-unit4': 'Field',
                    'ST15-unit1': 'Field', 'ST15-unit2': 'Field',
                    'ST16-unit1': 'SAII', 'ST16-unit2': 'SAII', 'ST16-unit3': 'CT', 'ST16-unit4': 'Field', 'ST16-unit5': 'SAI',
                    'ST17-unit1': 'SAI',
                    'ST18-unit1': 'Field', 'ST18-unit2': 'SAI', 'ST18-unit3': 'Field', 'ST18-unit4': 'SAII'}

    dir_temp = 'new_axes_3Dposition' if new_axes else 'manual_axes'
    data_dir = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/' + dir_temp
    if per_repeat:
        data_dir = 'D:/OneDrive - University of Virginia/Projects/TSSingleUnitModel/matlab/combined_data/'
    os.chdir(data_dir)
    df_all = pd.DataFrame()
    for name in os.listdir():  # iterate through all file
         if Path(name).is_file() and name.split('-')[-1].split('.')[0] == 'semicontrol':
            print(name)
            df_i = pd.read_csv(name)
            df_i['type'] = subtype_dict['-'.join(name.split('-')[3:5])]
            df_i['unit'] = '-'.join(name.split('-')[3:5])
            df_i['repeat_id_new'] = df_i['repeat_id']
            if per_repeat:
                for repeat in df_i['repeat_id'].unique():
                    if ~np.isnan(repeat) and repeat != 0:
                        print(repeat)
                        index = list(df_i[df_i['repeat_id'] == repeat].index)
                        indexs = index[:]
                        print(indexs, len(indexs))
                        for ii in range(len(indexs) // 2):
                            start = indexs[ii * 2]
                            end = indexs[ii * 2 + 1]
                            if end - start > 5000:
                                indexs.insert(ii * 2 + 1, start+1)
                        for ii in range(len(indexs) // 2):
                            start = indexs[ii * 2]
                            end = indexs[ii * 2 + 1]
                            print(end-start)
                            df_i.loc[(df_i.index >= start) & (df_i.index <= end), 'repeat_id_new'] = repeat
            df_all = pd.concat([df_all, df_i])
            df_i.to_csv('D:/OneDrive - University of Virginia/Projects/TSSingleUnitModel/matlab/processed_data/' + name)
    df_ave = df_all.groupby(['block_id', 'trial_id', 'unit', 'type', 'stimulus', 'vel', 'finger', 'force']).mean().reset_index()
    df_count = df_ave.groupby(['type', 'stimulus', 'vel', 'finger', 'force']).size().reset_index()
    df_stimulus = df_ave.groupby(['type', 'unit', 'block_id', 'finger', 'stimulus']).size().reset_index()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df_count)

    dir_temp = 'D:/OneDrive - University of Virginia/Projects/TSSingleUnitModel/python/data/semi_control_IFF/'
    if per_repeat:
        df_all.to_pickle(dir_temp + 'data_all_per_repeat.pkl')
        df_stimulus.to_csv(dir_temp + 'stimulus_info_per_repeat.csv', index=False, header=True)
    else:
        df_all.to_pickle(dir_temp + 'data_all.pkl')
        df_stimulus.to_csv(dir_temp + 'stimulus_info.csv', index = False, header=True)


def IFF_interpolation(IFF):
    t = np.arange(len(IFF))
    good = np.where(IFF != 0, 1, 0)
    print(good.size)
    print(np.sum(good))
    good = np.where(good == 1)

    # replace zeros as the next IFF value
    f = interpolate.interp1d(t[good], IFF[good], kind='next', bounds_error=False, fill_value=0)
    IFFinterp = np.where(IFF != 0, IFF, f(t))
    IFFinterp[np.isnan(IFFinterp)] = 0

    return IFFinterp

def plot_example_TS():
    def find_amplitude(vel, dis):
        return np.pi * dis / (2 * dis / vel)

    for u_i in range(2):
        if u_i == 0:
            dir_temp = 'new_axes_3Dposition/2022-06-17-ST16-unit5-semicontrol.csv'
            data_dir = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/' + dir_temp
            df = pd.read_csv(data_dir)
            df_block = df[df['block_id'] == 8].reset_index(drop=True)
            r = [0, 14.5]
            df_plot_0 = df_block[(df_block.index >= r[0]*1000) & (df_block.index <= r[1]*1000)]

            df_block = df[df['block_id'] == 3]
            r = [[216, 223.8], [239.5, 247], [263, 270.7], [286,294]] #, [207, 216][237.5, 247.2]
            df_plot_1 = df_block[((df_block['t'] >= r[0][0]) & (df_block['t'] <= r[0][1])) |
                               ((df_block['t'] >= r[1][0]) & (df_block['t'] <= r[1][1])) |
                               ((df_block['t'] >= r[2][0]) & (df_block['t'] <= r[2][1])) |
                               ((df_block['t'] >= r[3][0]) & (df_block['t'] <= r[3][1])) ]

            df_plot = pd.concat([df_plot_0, df_plot_1], ignore_index=True)

            # df_plot['IFF_interp'] = IFF_interpolation(df_plot['IFF'].values)
            # df_plot.replace(0, np.nan, inplace=True)
            # df_plot['t'] = df_plot.index * 0.001
            print(df_plot, len(df_plot.index), len(df_plot_1.index))
        if u_i == 1:
            dir_temp = 'new_axes_3Dposition/2022-06-15-ST14-unit2-semicontrol.csv'
            data_dir = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/' + dir_temp
            df = pd.read_csv(data_dir)
            df_plot_0 = df[(df['block_id'] == 6) & (df['trial_id'] == 2)]
            df_plot_0['velLong'] = - df_plot_0['velLong']

            df_plot_1 = df[(df['block_id'] == 3) & ((df['trial_id'] == 3) | (df['trial_id'] == 5) |
                          (df['trial_id'] == 8) | (df['trial_id'] == 11))]

            df_plot = pd.concat([df_plot_0, df_plot_1], ignore_index=True)

            # df_plot['IFF_interp'] = IFF_interpolation(df_plot['IFF'].values)
            # df_plot.replace(0, np.nan, inplace=True)
            # df_plot['t'] = df_plot.index * 0.001
            print(df_plot, len(df_plot_0.index), len(df_plot_1.index))


        fig, axes = plt.subplots(2, 1, sharex='all')
        fig.set_size_inches(8, 5)

        y_list = ['velLong', 'IFF']
        for i in range(2):
            ax = axes[i]
            if i == 0:
                # df_plot.replace({y_list[i]: {0: np.nan}}, inplace=True)
                # yhat = df_plot[y_list[i]].rolling(500, center=True, min_periods=5).mean().fillna(0)
                vels, repeats = [1, 3, 9, 18, 24], [4, 6, 18, 36, 48]
                gaps = [[1200, 1800, 1700, 2000, 2000, 2000], [1800, 1600, 1600, 1300, 2300, 2000]]
                yhat = []
                for i_r in range(5):
                    t_total = int(3 / vels[i_r] * 1000) * repeats[i_r]
                    x = np.array(range(t_total))
                    x_unit = np.pi / int(3 / vels[i_r] * 1000)
                    cosine = list(np.sin(x * x_unit) * find_amplitude(vels[i_r], 3))
                    yhat = yhat + [0] * gaps[u_i][i_r] + cosine
                yhat = yhat + [0] * gaps[u_i][-1]

                sns.lineplot(x=list(range(len(yhat))), y=yhat, color='grey', linewidth=1.5, alpha=0.4, label='Designed Vel', ax=ax)
                sns.lineplot(x=df_plot.index, y=df_plot[y_list[i]], color='seagreen', linewidth=1, alpha=0.4,
                             label='Tracked Vel', ax=ax)
            if i == 1:
                sns.lineplot(x=df_plot.index, y=df_plot[y_list[i]], color='tab:blue', linewidth=1, alpha=0.2, label='Raw IFF',
                             ax=ax)

                df_plot.replace({y_list[i]: {0: np.nan}}, inplace=True)
                yhat = df_plot[y_list[i]].rolling(500, center=True, min_periods=5).mean().fillna(0)
                sns.lineplot(x=df_plot.index, y=yhat, color='grey', linewidth=1.5, alpha=1, label='Moving Average', ax=ax)
            ax.set_xlim(None, 50000)
            ax.xaxis.set_tick_params(labelsize=tick_size)
            ax.yaxis.set_tick_params(labelsize=tick_size)
            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)
            handles, label_default = ax.get_legend_handles_labels()
            ax.legend(handles, label_default, frameon=False, loc=0, fontsize=legend_size, handletextpad=0.3,
                      markerscale=0.3, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
        sns.despine(trim=True)
        plt.tight_layout()

        fig.savefig(plot_dir+'raw_timeseries_'+str(u_i), dpi=800)
    plt.show()



def extract_features_from_iff(iff_raw, iff, interval):

    if len(iff_raw) < 5:
        feats = [np.nan] * 11
    else:
        iff_mean = np.mean(iff_raw)
        iff_variation = scipy.stats.variation(iff_raw)
        # iff_variation = np.std(iff_raw) /np.mean(iff_raw)
        n_spike = len(iff_raw)
        peak = np.max(iff_raw)

        spike = np.where(iff == 0, 0, 1)
        max_n_spike = max(np.convolve(spike,np.ones(interval,dtype=int),'valid'))

        iff_q = np.quantile(iff_raw, 0.75)
        iff_larger_than_q = iff_raw[iff_raw > iff_q]
        iff_high_qtr = np.mean(iff_larger_than_q)

        value, counts = np.unique(iff_raw, return_counts=True)
        iff_entropy = scipy.stats.entropy(counts)

        freqs = np.fft.fftfreq(iff.size, 0.001)
        ps = np.abs(np.fft.fft(iff))
        idx = np.argsort(freqs)
        freqs, ps = freqs[idx], ps[idx]
        x_mask = (freqs > 0) & (freqs <= 30)
        freqs_seg, ps_seg = freqs[x_mask], ps[x_mask]
        idx = np.argmax(ps_seg)
        freq_max, freq_amp_max = freqs_seg[idx], ps_seg[idx]
        freq_centroid = np.sum(ps_seg * freqs_seg) / np.sum(ps_seg)
        freq_entropy = scipy.stats.entropy(ps)


        feats = [iff_mean, iff_variation, n_spike, max_n_spike, iff_high_qtr, peak, iff_entropy, \
                 freq_max, freq_amp_max, freq_centroid, freq_entropy]
    names = ['iff_mean', 'iff_variation', 'n_spike', 'max_n_spike', 'iff_high_qtr', 'peak', 'iff_entropy', \
             'freq_max', 'freq_amp_max', 'freq_centroid', 'freq_entropy']

    return feats, names

def extract_neural_features(interval=0):

    data_all = pd.read_pickle(data_dir+'data_all.pkl')
    data_all['trial_id'] = data_all['block_id'].astype(str) + '_' + data_all['trial_id'].astype('Int32').astype(str)

    features = {}
    labels = {}
    df_feature = pd.DataFrame()
    for unit_name in data_all['unit'].unique():
        # print(unit_name)
        data_unit = data_all.loc[data_all['unit'] == unit_name]
        unit_type = data_unit['type'].values[0]

        # print(unit_type)
        if not unit_type in features.keys():
            features[unit_type] = []
            labels[unit_type] = []
        for trial_id in data_unit['trial_id'].unique():
            if trial_id.split('_')[-1] == '<NA>': continue
            data_trial = data_unit.loc[data_unit['trial_id']== trial_id]
            trial_row0 = data_trial.iloc[[0]]
            df_temp = trial_row0[['unit', 'trial_id', 'type', 'stimulus', 'vel', 'finger', 'force']]
            iff = data_trial['IFF'].values
            iff_raw = iff[iff != 0]
            if len(iff) < 2: continue

            feats, names = extract_features_from_iff(iff_raw, iff, interval)
            df_temp.loc[:, names] = feats
            df_feature = pd.concat([df_feature,df_temp],ignore_index=True)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
    #     print(df_feature)
    df_feature.to_csv(data_dir+'neural_features.csv', index = False, header=True)
    return df_feature

def extract_nerual_features_per_repeat(interval=0):

    data_all = pd.read_pickle(data_dir+'data_all_per_repeat.pkl')
    data_all['trial_id'] = data_all['block_id'].astype(str) + '_' + data_all['trial_id'].astype('Int32').astype(str)

    df_feature = pd.DataFrame()
    for unit_name in data_all['unit'].unique():
        # print(unit_name)
        data_unit = data_all.loc[data_all['unit'] == unit_name]

        for trial_id in data_unit['trial_id'].unique():
            if trial_id.split('_')[-1] == '<NA>': continue
            data_trial = data_unit.loc[data_unit['trial_id']== trial_id]
            for repeat in data_trial['repeat_id_new'].unique():
                if np.isnan(repeat) or repeat == 0: continue
                data_repeat = data_trial.loc[data_trial['repeat_id_new'] == repeat]
                repeat_row0 = data_repeat.iloc[[0]]
                df_temp = repeat_row0[['unit', 'trial_id', 'type', 'stimulus', 'vel', 'finger', 'force', 'repeat_id_new']]
                iff = data_repeat['IFF'].values
                iff = iff[~np.isnan(iff)]
                iff_raw = iff[iff != 0]
                # if len(iff_raw) < 5: continue

                feats, names = extract_features_from_iff(iff_raw, iff, interval)
                df_temp.loc[:, names] = feats
                df_feature = pd.concat([df_feature,df_temp],ignore_index=True)

        fig = plt.figure()
        plt.plot(range(len(data_unit.index)), data_unit['IFF'].values, data_unit['repeat_id_new'])
    plt.show()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
    #     print(df_feature)
    df_feature.to_csv(data_dir+'neural_features_per_repeat.csv', index = False, header=True)
    return df_feature


def plot_neural_features(adapt_method='actual', split_gesture=False, per_repeat=False):
    """
    :param adapt_method:
        'actual' for using raw data
        'VarGaussianMix' for changing velocity labels by clustering
    :param split_gesture:
        True for clustering for tapping and stroking separately
        False for clustering tapping and stroking data combined
    :param per_repeat:
        True for analysis per repeat
        False for analysis per trial
    :return: None plot instead
    """

    sns.set(style="ticks", font='Arial', font_scale=2.7)

    # df_feature = pd.read_csv(data_dir+'neural_features.csv')
    _, df_feature = adapt_contact_condition(adapt_method, split_gesture, per_repeat=per_repeat)

    df_feature = df_feature[df_feature['finger'] != ' two finger pads']
    df_feature.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                        'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}},
                       inplace=True)
    df_feature.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                        'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}},
                       inplace=True)

    split_str = 'split_ges' if split_gesture else 'combine_ges'
    plot_name = '_'.join([adapt_method, split_str])
    if per_repeat:
        dir_temp = plot_dir + 'per_repeat/stats_label_adapt/' + plot_name
    else:
        dir_temp = plot_dir + 'stats_label_adapt/' + plot_name
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    hue_order = {'vel': [1, 3, 9, 18, 24], 'finger': ['1f', 'wh'],
                 'force':['lf', 'mf', 'sf']}
    if adapt_method == 'VarGaussianMix':
        df_feature = df_feature[df_feature['vel'].notna()]
        hue_order['vel'] = np.sort(df_feature['vel'].unique())

    # y_limits = {'iff_mean': 200,'iff_variation': 1.5,'max_n_spike': 110,'peak': 550,'iff_entropy': 10}

    for stimulus in ['tap', 'stroke']:
        df_stimulus = df_feature[df_feature['stimulus'] == stimulus]
        for var in ['vel', 'finger', 'force']:
            fig = plt.figure()
            fig.set_size_inches(16, 6, forward=True)  #16,6,
            feat_i = 0
            for feat in ['iff_mean','n_spike','peak','max_n_spike','iff_variation','iff_entropy']:
            # for feat in ['n_spike', 'max_n_spike', 'freq_max', 'freq_amp_max', 'freq_centroid', 'freq_entropy']:
                ax = fig.add_subplot(2, 3, feat_i+1)  #2,3,i+1

                sns.boxplot(data=df_stimulus, x='type', y=feat, hue=var, showfliers=False,
                    order=unit_order, hue_order=hue_order[var], palette='flare')#,**PROPS)
                # sns.pointplot(data=feature_df, x=plot_var, y=feat,markers='D', #zorder=500,
                #     color='orange', order=orders[plot_var])

                ax.set_xlabel('', fontsize=label_size)
                # if feat == 'max_n_spike': ax.set_ylabel('iff_high_qtr')
                if feat == 'peak': ax.set_ylabel('iff_peak')
                ax.yaxis.label.set_size(label_size)
                ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
                ax.yaxis.set_tick_params(labelsize=tick_size)
                # ax.set_ylim([0, None])#y_limits[feat]])
                # if feat_i < 2: ax.set_xtick_labels([])
                if feat_i == 0:
                    handles, label_default= ax.get_legend_handles_labels()
                    ax.legend(handles, label_default,frameon=False, loc=0, fontsize=legend_size,handletextpad=0.3,
                        markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
                else:
                    ax.legend().remove()
                sns.despine(trim=True)
                plt.tight_layout()
                feat_i += 1

            fig.suptitle(stimulus + '_' + var, fontsize=label_size)
            # plt.tight_layout()
            plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.93, hspace=0.274, wspace=0.274)
            plt.savefig(dir_temp+'/neural_'+stimulus + '_' + var+'.png', dpi=600)


    plt.show()


def extract_features_from_contact(data_trial):
    feats = []
    features = ['area', 'depth', 'velAbsRaw', 'velLongRaw', 'velVertRaw', 'velLatRaw']
    features = ['area', 'depth', 'velAbs', 'velLong', 'velVert', 'velLat']
    for feat in features:
        data_feat = data_trial[feat].values
        data_feat = data_feat[data_feat != 0]
        # if feat in ['velLongRaw', 'velLatRaw', 'velVertRaw']:

        feat_mean = np.nanmean(np.abs(data_feat))
        feat_cv = np.nanstd(np.abs(data_feat)) /feat_mean

        feats.append(feat_mean)
        feats.append(feat_cv)
    return feats

def extract_contact_features():

    data_all = pd.read_pickle(data_dir+'data_all.pkl')
    data_all['trial_id'] = data_all['block_id'].astype(str) + '_' + data_all['trial_id'].astype('Int32').astype(str)

    # features = {}
    # labels = {}
    df_feature = pd.DataFrame()
    for unit_name in data_all['unit'].unique():
        # print(unit_name)
        data_unit = data_all.loc[data_all['unit'] == unit_name]
        unit_type = data_unit['type'].values[0]

        # print(unit_type)
        # if not unit_type in features.keys():
        #     features[unit_type] = []
        #     labels[unit_type] = []
        for trial_id in data_unit['trial_id'].unique():
            if trial_id.split('_')[-1] == '<NA>': continue
            data_trial = data_unit.loc[data_unit['trial_id']== trial_id]
            trial_row0 = data_trial.iloc[[0]]
            df_temp = trial_row0[['unit', 'trial_id', 'type', 'stimulus', 'vel', 'finger', 'force']]

            feats = extract_features_from_contact(data_trial)
            duration = len(data_trial[data_trial['area'] > 0].index) / 1000.
            length = len(data_trial.index) / 1000.
            feats = feats + [duration, length]
            names = []
            for n in features:
                names.append(n+'_mean')
                names.append(n+'_cv')
            names.append('duration')
            names.append('length')
            df_temp.loc[:, names] = feats
            df_feature = pd.concat([df_feature,df_temp],ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df_feature)
    df_feature.to_csv(data_dir+'contact_features.csv', index = False, header=True)
    return df_feature

def extract_contact_features_per_repeat():

    data_all = pd.read_pickle(data_dir+'data_all_per_repeat.pkl')
    data_all['trial_id'] = data_all['block_id'].astype(str) + '_' + data_all['trial_id'].astype('Int32').astype(str)

    df_feature = pd.DataFrame()
    for unit_name in data_all['unit'].unique():
        # print(unit_name)
        data_unit = data_all.loc[data_all['unit'] == unit_name]

        for trial_id in data_unit['trial_id'].unique():
            if trial_id.split('_')[-1] == '<NA>': continue
            data_trial = data_unit.loc[data_unit['trial_id']== trial_id]
            for repeat in data_trial['repeat_id_new'].unique():
                if np.isnan(repeat) or repeat == 0: continue
                data_repeat = data_trial.loc[data_trial['repeat_id_new'] == repeat]
                repeat_row0 = data_repeat.iloc[[0]]
                df_temp = repeat_row0[['unit', 'trial_id', 'type', 'stimulus', 'vel', 'finger', 'force', 'repeat_id_new']]

                feats = extract_features_from_contact(data_repeat)
                duration = len(data_repeat[data_repeat['area'] > 0].index) / 1000.
                length = len(data_repeat.index) / 1000.
                feats = feats + [duration, length]
                names = []
                for n in features:
                    names.append(n+'_mean')
                    names.append(n+'_cv')
                names.append('duration')
                names.append('length')
                df_temp.loc[:, names] = feats
                df_feature = pd.concat([df_feature,df_temp],ignore_index=True)

        fig = plt.figure()
        plt.plot(range(len(data_unit.index)), data_unit['velAbs'].values, data_unit['repeat_id_new'])
    plt.show()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df_feature)
    df_feature.to_csv(data_dir+'contact_features_per_repeat.csv', index = False, header=True)
    return df_feature

def plot_contact_features(adapt_method='actual', split_gesture=False, per_repeat=False):
    """
    :param adapt_method:
        'actual' for using raw data
        'VarGaussianMix' for changing velocity labels by clustering
    :param split_gesture:
        True for clustering for tapping and stroking separately
        False for clustering tapping and stroking data combined
    :param per_repeat:
        True for analysis per repeat
        False for analysis per trial
    :return: None plot instead
    """

    sns.set(style="ticks", font='Arial', font_scale=2.7)

    # df_feature = pd.read_csv(data_dir+'contact_features.csv')
    df_feature, _ = adapt_contact_condition(adapt_method, split_gesture, per_repeat=per_repeat)
    df_feature = df_feature[df_feature['finger'] != ' two finger pads']
    df_feature.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                        'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}},
                        inplace=True)
    df_feature.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                        'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}},
                        inplace=True)

    split_str = 'split_ges' if split_gesture else 'combine_ges'
    plot_name = '_'.join([adapt_method, split_str])
    if per_repeat:
        dir_temp = plot_dir + 'per_repeat/stats_label_adapt/' + plot_name
    else:
        dir_temp = plot_dir + 'stats_label_adapt/' + plot_name
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)


    hue_order = {'vel': [1, 3, 9, 18, 24], 'finger': ['1f', 'wh'],
                 'force':['lf', 'mf', 'sf']}
    if adapt_method == 'VarGaussianMix':
        df_feature = df_feature[df_feature['vel'].notna()]
        hue_order['vel'] = np.sort(df_feature['vel'].unique())

    # y_limits = {'iff_mean': 200,'iff_variation': 1.5,'max_n_spike': 110,'peak': 550,'iff_entropy': 5}

    for name in ['_mean']: #, '_cv'
        for stimulus in ['tap', 'stroke']:
            df_stimulus = df_feature[df_feature['stimulus'] == stimulus]
            for var in ['vel', 'finger', 'force']:
                fig = plt.figure()
                fig.set_size_inches(16, 6, forward=True)  #16,6,
                feat_i = 0
                names = [n+name for n in features]
                for feat in names:
                    ax = fig.add_subplot(2, 3, feat_i+1)  #2,3,i+1

                    sns.boxplot(data=df_stimulus, x='type', y=feat, hue=var, showfliers=False,
                        order=unit_order, hue_order=hue_order[var], palette='flare')#,**PROPS)
                    # sns.pointplot(data=feature_df, x=plot_var, y=feat,markers='D', #zorder=500,
                    #     color='orange', order=orders[plot_var])

                    ax.set_xlabel('', fontsize=label_size)
                    ax.yaxis.label.set_size(label_size)
                    ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
                    ax.yaxis.set_tick_params(labelsize=tick_size)
                    # ax.set_ylim([0, y_limits[feat]])
                    # if feat_i < 2: ax.set_xtick_labels([])
                    if feat_i == 0:
                        handles, label_default= ax.get_legend_handles_labels()
                        ax.legend(handles, label_default,frameon=False, loc=0, fontsize=legend_size,handletextpad=0.3,
                            markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
                    else:
                        ax.legend().remove()
                    sns.despine(trim=True)
                    plt.tight_layout()
                    feat_i += 1

                fig.suptitle(stimulus + '_' + var, fontsize=label_size)
                # plt.tight_layout()
                plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.93, hspace=0.274, wspace=0.274)
                plt.savefig(dir_temp+'/contact_'+stimulus+'_'+var+name+'.png', dpi=600)

    plt.show()

def plot_contact_IFF_reg():
    df_all = pd.read_pickle(data_dir + 'data_all.pkl')
    df_all = df_all[(df_all['IFF'] > 0) & (df_all['area'] > 0)]

    feats = ['velAbs', 'area', 'depth', 'velLong', 'velLat', 'velVert']

    for feat in feats:
        for p_i in ['stroke', 'tap', 'all']:
            if p_i != 'all':
                df_plot = df_all[df_all['stimulus'] == p_i]
            else:
                df_plot = df_all

            fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
            fig.set_size_inches(16, 9)
            fig.suptitle('_'.join([p_i, feat]), fontsize=label_size)

            for i in range(len(unit_order)):
                df_unit = df_plot[df_plot['type'] == unit_order[i]]
                ax = axes[i//3, i%3]

                units = df_unit['unit'].unique()
                palette = sns.color_palette('Set2', n_colors=len(units))
                for i_u in range(len(units)):
                    df_unit_ = df_unit[df_unit['unit'] == units[i_u]]
                    sns.regplot(data=df_unit_, x=feat, y='IFF', color=palette[i_u], lowess=True, marker='.', ci=None, ax=ax)
                sns.regplot(data=df_unit, x=feat, y='IFF', lowess=True, scatter=False, ci=None, ax=ax,
                            line_kws=dict(alpha=0.2, color='grey', linewidth=20))
                corr = stats.spearmanr(df_unit[feat].values, df_unit['IFF'].values)
                x, y = df_unit[feat].values.reshape(-1, 1), df_unit['IFF'].values.reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                r2 = r2_score(y, y_pred)
                if corr.pvalue < 0.05:
                    ax.set_title(unit_order[i] + '\nr=' + str(round(corr.statistic, 2)) + stars(corr.pvalue) \
                                 + ' r2=' + str(round(r2, 2)), color='orange', fontsize=label_size)
                else:
                    ax.set_title(unit_order[i], fontsize=label_size)
                ax.xaxis.set_tick_params(labelsize=tick_size)
                ax.yaxis.set_tick_params(labelsize=tick_size)
                ax.xaxis.label.set_size(label_size)
                ax.yaxis.label.set_size(label_size)
                sns.despine(trim=True)
            axes[1, 2].axis('off')
            plt.tight_layout()
            plt.savefig(plot_dir + '/contact_neural_reg/RawData_' + '_'.join([feat, p_i]) + '.png', dpi=200)

    plt.show()


def get_variance(adapt_method='actual', split_gesture=False, per_repeat=False):
    if per_repeat:
        data_all = pd.read_pickle('data/semi_control_IFF/data_all_per_repeat.pkl')
    else:
        data_all = pd.read_pickle('data/semi_control_IFF/data_all.pkl')
    data_all = data_all[data_all['trial_id'].notna()]
    data_all['unique_id'] = data_all['unit'] + '_' + data_all['block_id'].astype(int).astype(str) + '_' + data_all['trial_id'].astype(int).astype(str)

    name = adapt_method + '_' + 'split_ges' if split_gesture else adapt_method + '_' + 'combine_ges'
    if adapt_method != 'actual':
        df_label = pd.read_csv(data_dir + 'labels_' + name + '.csv')
        for id in df_label['unique_id'].unique():
            print(id)
            print(len(data_all[data_all['unique_id'] == id].index))
            print(df_label.loc[df_label['unique_id'] == id, 'vel'])
            data_all.loc[data_all['unique_id'] == id, ['vel']] = df_label.loc[df_label['unique_id'] == id, ['vel']].values
            print(data_all.loc[data_all['unique_id'] == id, ['vel']])


    data_all.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                      'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}}, inplace=True)
    data_all.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                      'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}}, inplace=True)
    data_all['vel'] = data_all['vel'].astype(str)
    data_all['trial_unique'] = data_all['unit'] + '_' + data_all['block_id'].astype(str) + '_' + \
                           data_all['trial_id'].astype('Int32').astype(str)
    print(data_all.dtypes)
    data_all['label'] = data_all[['type', 'stimulus', 'vel', 'finger', 'force']].astype(str).agg('_'.join, axis=1)

    bin_size = 0.1
    cov_list = []
    for label in data_all['label'].unique():
        data_i = data_all[data_all['label'] == label]
        temp_list = []
        df_i = pd.DataFrame()
        for trial_i in data_i['trial_unique'].unique():
            data_trial_i = data_i[data_i['trial_unique'] == trial_i]
            if len(data_trial_i.index) < 2: continue
            if len(data_trial_i[data_trial_i['spike'] == 1].index) < 1: continue
            df_norm, _, _ = nerual_psth_smoothed_contact(data_trial_i, bin_size, norm_method='min_max')
            # print(df_norm)
            df_norm['trial_unique'] = str(trial_i)
            df_i = pd.concat([df_i, df_norm], ignore_index=True)
        if len(df_i.index) < 2: continue
        for col in features + ['spike_binned']:
            df_col = df_i[[col, 'trial_unique']]
            df_col = df_col.pivot(columns='trial_unique', values=col).fillna(0)
            cov_matrix = df_col.cov()
            _, s, _ = svd(cov_matrix)  # eigenvalue
            trace = np.sum(s**2)
            print(trace)
            temp_list.append(trace)
        temp_list.append(label)
        cov_list.append(temp_list)
    df_cov = pd.DataFrame(columns=[i+'_var' for i in features + ['spike_binned']] + ['label'], data=cov_list)

    print(df_cov)
    print(per_repeat)
    if per_repeat:
        print('----------------------------------')
        df_cov.to_csv('data/semi_control_IFF/df_cov_' + name + '_per_repeat.csv', index = False, header=True)
    else:
        df_cov.to_csv('data/semi_control_IFF/df_cov_' + name + '.csv', index=False, header=True)

def plot_variance(adapt_method='actual', split_gesture=False, per_repeat=False):
    name = adapt_method + '_' + 'split_ges' if split_gesture else adapt_method + '_' + 'combine_ges'
    if per_repeat:
        df_cov = pd.read_csv('data/semi_control_IFF/df_cov_' + name + '_per_repeat.csv')
    else:
        df_cov = pd.read_csv('data/semi_control_IFF/df_cov_' + name + '.csv')

    df_cov[['unit_type', 'stimulus', 'vel', 'finger', 'force']] = df_cov['label'].str.split('_', expand = True)
    df_cov['vel'] = df_cov['vel'].astype(float)
    df_cov.sort_values(by=['stimulus', 'vel', 'force', 'finger'], inplace=True)
    df_cov['label'] = df_cov['vel'].astype(str)+ '_' + df_cov['force'] + '_' + df_cov['finger']

    df_melt = pd.melt(df_cov, id_vars=['label','unit_type', 'stimulus', 'vel', 'finger', 'force'],
                      value_vars=[i+'_var' for i in features + ['spike_binned']],
                      var_name='feat', value_name='cov')
    df_melt['contact_f'] = np.where(df_melt['feat']=='spike_binned_var', False, True)
    print(df_melt)

    if per_repeat:
        dir_temp = plot_dir + 'per_repeat/stats_label_adapt/' + name
    else:
        dir_temp = plot_dir + 'stats_label_adapt/' + name
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    for stimulus in ['tap', 'stroke']:
        df_stimulus = df_melt[df_melt['stimulus'] == stimulus]
        fig, axes = plt.subplots(2, 3, sharey='all')
        fig.set_size_inches(16, 6, forward=True)
        for i in range(len(unit_order)):
            df_unit = df_stimulus[df_stimulus['unit_type'] == unit_order[i]]
            ax = axes[i//3, i%3]
            sns.boxplot(data=df_unit, x='contact_f', y='cov', hue='vel', showfliers=False,
                        order=[1, 0], #hue_order=[str(i) for i in np.sort(df_unit['vel'].astype(float).unique())],
                        palette='flare', ax=ax)#,**PROPS)
            # sns.pointplot(data=feature_df, x=plot_var, y=feat,markers='D', #zorder=500,
                #       color='orange', order=orders[plot_var])
            ax.set_title(unit_order[i], fontsize=label_size)
            ax.set_xlabel('', fontsize=label_size)
            ax.set_ylabel('Variance', fontsize=label_size)
            # ax.set_ylim(0, 0.14)
            ax.set_xticklabels(['contact','IFF'])
            ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
            ax.yaxis.set_tick_params(labelsize=tick_size)
            if i == 0:
                handles, label_default= ax.get_legend_handles_labels()
                ax.legend(handles, label_default,frameon=False, loc=0, fontsize=legend_size,handletextpad=0.3,
                    markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            else:
                ax.legend().remove()
            plt.tight_layout()

        fig.suptitle(stimulus, fontsize=label_size)
        plt.tight_layout()
        sns.despine(trim=True)
        # plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.93, hspace=0.274, wspace=0.274)
        plt.savefig(dir_temp+'/'+stimulus+'_var.png', dpi=600)

        ##### regression
        df_stimulus = df_cov[df_cov['stimulus'] == stimulus]
        fig, axes = plt.subplots(2, 3, sharey='all', sharex='all')
        fig.set_size_inches(16, 9, forward=True)
        for i in range(len(unit_order)):
            df_unit = df_stimulus[df_stimulus['unit_type'] == unit_order[i]]
            ax = axes[i // 3, i % 3]
            palette = sns.color_palette('Set2', n_colors=len(features))
            plot_feats = [i + '_var' for i in features]
            for i_p in range(len(features)):
                x, y = df_unit[plot_feats[i_p]].values.reshape(-1, 1), df_unit['spike_binned_var'].values.reshape(-1, 1)
                corr = stats.spearmanr(x, y)
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                p_val = corr.pvalue
                r2 = r2_score(y, y_pred)
                if p_val < 0.05:
                    label = features[i_p] + ' p:' + stars(p_val) + ' r2=' + str(round(r2, 2))
                    label = '-' + label if corr.statistic < 0 else label
                else:
                    label = features[i_p] + ' non-significant'

                sns.regplot(data=df_unit, x=plot_feats[i_p], y='spike_binned_var', color=palette[i_p],
                            lowess=True, scatter=None, ci=None, ax=ax, label=label)
                sns.scatterplot(data=df_unit, x=plot_feats[i_p], y='spike_binned_var', color=palette[i_p],
                                marker='.', ax=ax, size='vel', sizes=(40, 600), alpha=0.5)

            ax.set_title(unit_order[i], fontsize=label_size)
            ax.set_xlabel('Contact', fontsize=label_size)
            ax.set_ylabel('PSTH', fontsize=label_size)
            # ax.set_ylim(0, 0.14)
            # ax.set_xticklabels(['contact', 'IFF'])
            ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
            ax.yaxis.set_tick_params(labelsize=tick_size)
            handles, labels = ax.get_legend_handles_labels()
            new_h, new_l = [], []
            for i in range(len(labels)):
                if labels[i] not in ['1.0', '3.0', '9.0', '18.0', '24.0']:
                    new_h.append(handles[i])
                    new_l.append(labels[i])
            ax.legend(new_h, new_l, frameon=False, loc=0, fontsize=legend_size, handletextpad=0.3,
                      markerscale=1.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)

            plt.tight_layout()
        axes[1, 2].remove()

        fig.suptitle(stimulus, fontsize=label_size)
        plt.tight_layout()
        sns.despine(trim=True)
        # plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.93, hspace=0.274, wspace=0.274)
        plt.savefig(dir_temp + '/' + stimulus + '_var_reg.png', dpi=600)
    plt.show()



def fft_helper(signal, x_min=1, x_max=30):
    freqs = np.fft.fftfreq(signal.size, 0.001)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(signal))
    x_val, y_val = freqs[idx], ps[idx]
    x_mask = (x_val >= x_min) & (x_val <= x_max)
    x, y = x_val[x_mask], y_val[x_mask]

    # window = len(x) // 10
    # window = window if window % 2 != 0 else window + 1
    # order = 7 if window > 7 else window - 1
    # y = savgol_filter(y, window, order)
    # window = len(x) // 5
    # window = window if window % 2 != 0 else window + 1
    # y = savgol_filter(y, window, 3)

    return x, y


def plot_power_spectrum(signal_source):
    data_all = pd.read_pickle(data_dir + 'semi_control_IFF/data_all.pkl')
    data_all['trial_id'] = data_all['block_id'].astype(str) + '_' + data_all['trial_id'].astype('Int32').astype(str)
    hue_order = {'vel': [1, 3, 9, 18, 24], 'finger': [' one finger tip', ' whole hand'],
                 'force': [' light force', ' moderate force', ' strong force']}

    if signal_source == 'neural':
        for unit_type in unit_order:
            for stimulus in ['tap', 'stroke']:
                data_type = data_all[(data_all['type'] == unit_type) & (data_all['stimulus'] == stimulus)]
                fig = plt.figure()
                fig.set_size_inches(16, 6, forward=True)  # 16,6,
                feat_i = 0
                for vel in hue_order['vel']:
                    ax = fig.add_subplot(2, 3, feat_i + 1)  # 2,3,i+1
                    data_vel = data_type[data_type['vel'] == vel]
                    y_shift = 0
                    y_max = 0.
                    for trial_id in data_vel['trial_id'].unique():
                        data_trial = data_vel[data_vel['trial_id'] == trial_id]
                        iff = data_trial['IFF'].values
                        signal = iff
                        # signal = np.where(iff != 0, 1, 0)
                        x, y = fft_helper(signal)
                        ax.plot(x, y + y_shift * 100)
                        y_max = max(y.max(), y_max)
                        # f, Pxx_spec = signal.welch(signal, 1000, scaling='spectrum')#, detrend=False
                        # plt.semilogy(f, np.sqrt(Pxx_spec))
                        y_shift += 1
                    # ax.set_ylim([0, y_max])
                    ax.set_title(str(vel) + ' cm/s')
                    feat_i += 1
                fig.suptitle(stimulus + ' ' + unit_type, fontsize=label_size)
                plt.savefig(plot_dir + 'power_spectrum/' + signal_source + '_' + stimulus + '_' + unit_type + '.png',
                            dpi=200)

    if signal_source == 'contact':
        for stimulus in ['tap', 'stroke']:
            for cq in features:
                data_type = data_all[data_all['stimulus'] == stimulus]
                fig = plt.figure()
                fig.set_size_inches(16, 6, forward=True)  # 16,6,
                feat_i = 0
                for vel in hue_order['vel']:
                    ax = fig.add_subplot(2, 3, feat_i + 1)  # 2,3,i+1
                    data_vel = data_type[data_type['vel'] == vel]
                    y_shift = 0
                    y_max = 0
                    for trial_id in data_vel['trial_id'].unique():
                        data_trial = data_vel[data_vel['trial_id'] == trial_id]
                        if cq == 'velLong2':
                            signal = data_trial[cq[:-1]].values
                        else:
                            signal = data_trial[cq].values
                        x, y = fft_helper(signal)
                        if cq == 'velLong2':
                            idx = np.array(range(len(y)))
                            idx_2 = idx * 2
                            f_f = interp1d(idx_2[:len(idx_2) // 2], y[:len(idx_2) // 2], fill_value="extrapolate")
                            y = f_f(idx)
                        ax.plot(x, y + y_shift * 100)
                        y_max = max(y.max(), y_max)
                        ### WELCH METHODS
                        # f, Pxx_spec = signal.welch(signal, 1000, scaling='spectrum')#, detrend=False
                        # plt.semilogy(f, np.sqrt(Pxx_spec))
                        y_shift += 1
                    # ax.set_ylim([0, y_max])
                    ax.set_title(str(vel) + ' cm/s')
                    feat_i += 1
                fig.suptitle(stimulus + ' ' + cq, fontsize=label_size)
                plt.savefig(plot_dir + 'power_spectrum/' + signal_source + '_' + stimulus + '_' + cq + '.png', dpi=200)

    plt.show()


def extract_corr_contact_neural_power_sepctrum(method):
    data_all = pd.read_pickle(data_dir + 'semi_control_IFF/data_all.pkl')
    data_all['trial_id'] = data_all['unit'] + '_' + data_all['block_id'].astype(str) + '_' + data_all[
        'trial_id'].astype('Int32').astype(str)
    type_list, trial_list, vel_list, stimulus_list, cq_list, corr_list = [], [], [], [], [], []

    for trial_id in data_all['trial_id'].unique():
        print(trial_id)
        if trial_id.split('_')[-1] == '<NA>': continue
        data_trial = data_all[data_all['trial_id'] == trial_id]
        unit_type = data_trial['type'].values[0]
        vel = data_trial['vel'].values[0]
        stimulus = data_trial['stimulus'].values[0]
        iff_signal = data_trial['IFF'].values
        _, iff_fft = fft_helper(iff_signal)
        for cq in features:  # , 'velLong2'
            cq_signal = data_trial[cq[:-1]].values if cq == 'velLong2' else data_trial[cq].values
            _, cq_fft = fft_helper(cq_signal)
            if cq == 'velLong2' and len(cq_fft) != 0:
                idx = np.array(range(len(cq_fft)))
                idx_2 = idx * 2
                f_f = interp1d(idx_2[:len(idx_2) // 2], cq_fft[:len(idx_2) // 2], fill_value="extrapolate")
                cq_fft = f_f(idx)
            if method == 'pearson':
                corr = scipy.stats.pearsonr(iff_fft, cq_fft)[0]  # spearmanr
            if method == 'spearman':
                corr = scipy.stats.spearmanr(iff_fft, cq_fft)[0]  # spearmanr
            type_list.append(unit_type)
            trial_list.append(trial_id)
            vel_list.append(vel)
            stimulus_list.append(stimulus)
            cq_list.append(cq)
            corr_list.append(corr)
    df_corr = pd.DataFrame({'type': type_list, 'trial_id': trial_list, 'vel': vel_list,
                            'stimulus': stimulus_list, 'cq': cq_list, 'corr': corr_list})
    df_corr.dropna(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df_corr)
    df_corr.to_csv(data_dir + 'semi_control_IFF/corr_contact_neural_power_spectrum_' + method + '.csv', index=False,
                   header=True)


def smooth_IFF(IFF_raw, iff_format):
    t = np.arange(len(IFF_raw))
    if iff_format == 'interp':
        good = np.where(IFF_raw != 0, 1, 0)
        print(good.size)
        print(np.sum(good))
        good = np.where(good == 1)

        # replace zeros as the next IFF value
        f = interpolate.interp1d(t[good], IFF_raw[good], kind='next', bounds_error=False, fill_value=0)
        IFF_processed = np.where(IFF_raw != 0, IFF_raw, f(t))
        IFF_processed[np.isnan(IFF_processed)] = 0

    if iff_format == 'smooth':
        IFF_processed = savgol_filter(IFF_raw, 100, 3)

    # sns.set(style="ticks", font='Arial', font_scale=1.8)
    # fig, axs = plt.subplots(2, 1, sharex='all')
    # fig.set_size_inches(16, 9, forward=True)
    # ax = axs[0]
    # ax.plot(t, IFF_raw)
    # ax.set_ylabel('IFF_raw', fontsize=27)
    # ax.set_xlabel('')
    # ax.xaxis.set_tick_params(labelsize=24)
    # ax = axs[1]
    # ax.plot(t, IFF_raw, 'o', t, IFF_processed, '-')
    # ax.set_ylabel('Interpolation', fontsize=27)
    # ax.set_xlabel('')
    # ax.yaxis.set_tick_params(labelsize=24)
    # # fig.savefig(plot_dir+'IFF_processed.png')
    # # pickle.dump(fig, open('plots/'+folder_dir+'IFF_processed.pickle', 'wb'))
    # # plt.show()
    # # plt.close()
    # plt.show()

    return IFF_processed


def extract_corr_contact_neural(iff_format, method):
    data_all = pd.read_pickle(data_dir + 'semi_control_IFF/data_all.pkl')
    data_all['trial_id'] = data_all['unit'] + '_' + data_all['block_id'].astype(str) + '_' + data_all[
        'trial_id'].astype('Int32').astype(str)
    if iff_format != 'raw':
        data_all['IFF_processed'] = smooth_IFF(data_all['IFF'].values, iff_format)

    type_list, trial_list, vel_list, stimulus_list, cq_list, corr_list = [], [], [], [], [], []

    for trial_id in data_all['trial_id'].unique():
        print(trial_id)
        if trial_id.split('_')[-1] == '<NA>': continue
        data_trial = data_all[data_all['trial_id'] == trial_id]
        unit_type = data_trial['type'].values[0]
        vel = data_trial['vel'].values[0]
        stimulus = data_trial['stimulus'].values[0]
        iff_signal = data_trial['IFF'].values
        if iff_format == 'raw':
            iff_signal = data_trial['IFF'].values
        else:
            iff_signal = data_trial['IFF_processed'].values
        for cq in features:
            cq_signal = np.abs(data_trial[cq].values)
            if method == 'pearson':
                corr = scipy.stats.pearsonr(iff_signal, cq_signal)[0]  # spearmanr
            if method == 'spearman':
                corr = scipy.stats.spearmanr(iff_signal, cq_signal)[0]  # spearmanr
            type_list.append(unit_type)
            trial_list.append(trial_id)
            vel_list.append(vel)
            stimulus_list.append(stimulus)
            cq_list.append(cq)
            corr_list.append(corr)
    df_corr = pd.DataFrame({'type': type_list, 'trial_id': trial_list, 'vel': vel_list,
                            'stimulus': stimulus_list, 'cq': cq_list, 'corr': corr_list})
    df_corr.dropna(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df_corr)
    df_corr.to_csv(data_dir + 'corr_contact_neural_' + iff_format + '_' + method + '.csv', index=False, header=True)


def plot_corr_contact_neural(fft_flag, iff_format, method):
    if fft_flag:
        df_corr = pd.read_csv(data_dir + 'corr_contact_neural_power_spectrum_' + method + '.csv')
        fig_dir = plot_dir + 'power_spectrum/corr_' + method + '_'
    else:
        df_corr = pd.read_csv(data_dir + 'corr_contact_neural_' + iff_format + '_' + method + '.csv')
        fig_dir = plot_dir + 'corr_iff_' + iff_format + '_' + method + '_'

    for stimulus in ['tap', 'stroke']:
        df_i = df_corr[df_corr['stimulus'] == stimulus]
        fig = plt.figure()
        fig.set_size_inches(16, 6, forward=True)  # 16,6,
        i = 1
        for unit_type in unit_order:
            df_ii = df_i[df_i['type'] == unit_type]
            if len(df_ii.index):
                ax = fig.add_subplot(2, 3, i)
                sns.barplot(data=df_ii, x='vel', y='corr', hue='cq')
                # sns.stripplot(data=df_ii, x='vel', y='corr', hue='cq')
                ax.set_title(unit_type)
                ax.set_ylim([-0.2, 1])
                ax.set_xlabel('')
                ax.get_legend().remove()
                sns.despine()
                i += 1
            fig.suptitle(stimulus, fontsize=label_size)
            plt.tight_layout()
            plt.savefig(fig_dir + stimulus + '.png', dpi=200)
    plt.show()

if __name__ == '__main__':

    sns.set(style="ticks", font='Arial')

    # combine_neural_contact_data(True, True)

    # data_all = pd.read_pickle(data_dir+'data_all.pkl')
    # print('data loaded')

    # plot_example_TS()

    # extract_neural_features(interval=1000)
    # extract_nerual_features_per_repeat(interval=1000)
    # plot_neural_features(adapt_method='VarGaussianMix', split_gesture=False, per_repeat=False)

    # extract_contact_features()
    # extract_contact_features_per_repeat()
    # plot_contact_features(adapt_method='VarGaussianMix', split_gesture=False, per_repeat=False)

    # plot_contact_IFF_reg()

    # get_variance(adapt_method='actual', split_gesture=False, per_repeat=True)
    plot_variance(adapt_method='actual', split_gesture=False, per_repeat=True)


    ### --------- frequency domain ---------
    # plot_power_spectrum('neural')

    # method = 'pearson' #'spearman' #
    # extract_corr_contact_neural_power_sepctrum(method)
    # plot_corr_contact_neural(True, None, method)

    # extract_corr_contact_neural('interp', method)
    # plot_corr_contact_neural(False, 'interp', method)


