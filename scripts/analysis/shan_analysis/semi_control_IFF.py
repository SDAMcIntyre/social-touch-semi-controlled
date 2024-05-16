import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import scipy
from scipy import stats
from scipy.stats import f
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from utils import nerual_psth_smoothed_contact
from numpy.linalg import eig, svd
from sklearn.mixture import BayesianGaussianMixture

from params_semi_control_IFF import *


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

def plot_contact_hist(df_contact, c_feat):
    contact_conditions = {'velAbs_mean': 'vel', 'velLong_mean': 'vel', 'velVert_mean': 'vel', 'velLat_mean': 'vel',
                          'depth_mean': 'force', 'area_mean': 'finger'}

    labels = df_contact[contact_conditions[c_feat]]
    data = df_contact[c_feat].values.reshape(-1, 1)
    vel_ranges = []
    for l in np.unique(labels):
        if ~np.isnan(l) and l!=0:
            data_label = data[labels == l]
            print(l)

            try:
                vel_range = '[' + str(round(np.nanmin(data_label),2)) +', '+ str(round(np.nanmax(data_label),2)) + ']'
            except ValueError:
                pass

            print(vel_range)
            vel_ranges.append(vel_range)

    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
    fig.set_size_inches(8, 5, forward=True)
    for stim in ['stroke', 'tap']:
        ax = axes[0 if stim == 'stroke' else 1]
        df_plot = df_contact[df_contact['stimulus'] == stim]
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
    sns.despine(trim=True)
    return fig

def plot_all_combinations(contact_feat, df_combined, neural_feat, contact_conditions, name):

    dir_temp = plot_dir+'contact_neural_reg/kde_'+name
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    ### ---------- histgram of contact attributes -------------
    for c_feat in contact_feat:
        fig = plot_contact_hist(df_combined, c_feat)
        fig.savefig(plot_dir+'/contact_neural_reg/adapt_' + name + '_' + c_feat+'.png', dpi=200)

    # contact_feat = contact_feat + ['duration']
    for unit_type in unit_order:
        for p_i in ['stroke', 'tap', 'all']:
            df_unit = df_combined[df_combined['type'] == unit_type]
            if p_i != 'all':
                df_plot = df_unit[df_unit['stimulus'] == p_i]
            else:
                df_plot = df_unit

            ## ----------- plot contact - IFF feature correlation -------------------
            fig, axes = plt.subplots(len(neural_feat), len(contact_feat), sharex='col', sharey='row')
            fig.set_size_inches(20, 14)
            fig.suptitle('_'.join([unit_type, p_i]), fontsize=label_size)
            for i in range(len(neural_feat)):
                for j in range(len(contact_feat)):
                    ax = axes[i, j]
                    sns.regplot(data=df_plot, x=contact_feat[j], y=neural_feat[i], lowess=True, marker='.', ci=None, ax=ax)
                    sns.regplot(data=df_plot, x=contact_feat[j], y=neural_feat[i], order=1, scatter=False, ci=None,
                                color='tab:green', ax=ax)
                    corr = stats.spearmanr(df_plot[contact_feat[j]].values, df_plot[neural_feat[i]].values)
                    x, y = df_plot[contact_feat[j]].values.reshape(-1, 1), df_plot[neural_feat[i]].values.reshape(-1, 1)
                    model = LinearRegression().fit(x, y)
                    y_pred = model.predict(x)
                    r2 = r2_score(y, y_pred)
                    if corr.pvalue < 0.05:
                        ax.set_title('r=' + str(round(corr.statistic, 2)) + stars(corr.pvalue) \
                                     + ' r2=' + str(round(r2, 2)), color='orange')
                    # ax.set_ylim([-0.2, 1])
                    # ax.set_xlabel('')
                    # ax.get_legend().remove()
                    sns.despine()
            plt.tight_layout()
            plt.savefig(plot_dir + '/contact_neural_reg/' + '_'.join([unit_type, p_i]) + '.png', dpi=200)


            ## ---------- contact - IFF kde distribution
            for j in range(len(contact_feat)):
                sns.set(style="ticks", font='Arial', font_scale=1.9)
                palette = sns.cubehelix_palette(n_colors=len(df_plot[contact_conditions[contact_feat[j]]].unique()))
                g = sns.jointplot(data=df_plot,x=contact_feat[j],y='iff_mean',hue=contact_conditions[contact_feat[j]],
                                    kind='scatter', palette=palette, ratio=3,s=5)
                sns.kdeplot(data=df_plot,x=contact_feat[j],y='iff_mean',hue=contact_conditions[contact_feat[j]],
                            palette=palette, ax=g.ax_joint)
                # sns.regplot(df_plot, x=contact_feat[j], y='iff_mean', lowess=True, marker='.', ci=None, ax=g.ax_joint)
                g.set_axis_labels(contact_feat[j], 'iff_mean', fontsize=label_size)
                sns.despine()
                sns.move_legend(g.ax_joint, loc=0, frameon=False, title='', fontsize=legend_size)
                plt.gcf().set_size_inches(6, 6)
                g.savefig(dir_temp+'/'+'_'.join([unit_type, p_i, contact_feat[j]])+'_kde.png', dpi=200)


def fit_plot_sigmoid(x, y, p0):
    def sigmoid(x, L, k, b):
        y = L / (1 + np.exp(-k * (x))) + b
        return y

    popt, pcov = curve_fit(sigmoid, x, y, p0, method='lm', maxfev=5000)
    y_pred = sigmoid(x, *popt)

    return y_pred

def F_test_regression(y, y_pred, n_var):
    p = n_var  # Number of parameters in your model
    N = len(y)

    # Calculate TSS, RSS, and ESS
    TSS = np.sum((y - np.mean(y)) ** 2)
    RSS = np.sum((y - y_pred) ** 2)
    ESS = TSS - RSS

    F = (ESS / (p - 1)) / (RSS / (N - p))
    p_val = f.sf(F, p - 1, N - p)

    return F, p_val

def plot_reg_picked_feature(df_combined, dir_temp):
    ## ------------ regression for vel and depth ------------------

    plot_c = [['velAbs_mean', 'iff_mean'], ['depth_mean', 'n_spike']] #
    for c in plot_c:
        for p_i in ['stroke', 'tap', 'all']:
            if p_i != 'all':
                df_plot = df_combined[df_combined['stimulus'] == p_i]
            else:
                df_plot = df_combined

            fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
            fig.set_size_inches(16, 9)
            fig.suptitle('_'.join([p_i, c[0]]), fontsize=label_size)

            for i in range(len(unit_order)):
                df_unit = df_plot[df_plot['type'] == unit_order[i]]
                ax = axes[i//3, i%3]

                units = df_unit['unit'].unique()
                df_unit = df_unit.dropna(subset=[c[0], c[1]])
                palette = sns.color_palette('Set2', n_colors=len(units))
                for i_u in range(len(units)):
                    df_unit_ = df_unit[df_unit['unit'] == units[i_u]]
                    # df_unit_ = df_unit_.dropna(subset=[c[0], c[1]])
                    x, y = df_unit_[c[0]].values, df_unit_[c[1]].values
                    # x_nan, y_nan = np.argwhere(np.isnan(x)), np.argwhere(np.isnan(y))
                    # print(len(x), len(y))
                    # index_nan = list(set(x_nan.flatten().tolist()) | set(y_nan.flatten().tolist()))
                    # x, y = np.delete(x, index_nan), np.delete(y, index_nan)
                    # print(len(index_nan), len(x), len(y))
                    if unit_order[i] == 'CT' or len(df_unit_.index) >= 15:
                        if unit_order[i] == 'CT':
                            model = np.poly1d(np.polyfit(x, y, 2))
                            y_pred = model(x)
                            _, p_val = F_test_regression(y, y_pred, 3)
                        else:
                            if c[0] == 'velAbs_mean':
                                p0 = [50, 0, 50]
                                y_pred = fit_plot_sigmoid(x, y, p0)
                                _, p_val = F_test_regression(y, y_pred, len(p0))
                            if c[0] == 'depth_mean':
                                model = np.poly1d(np.polyfit(x, y, 1))
                                y_pred = model(x)
                                _, p_val = F_test_regression(y, y_pred, 2)
                        r2 = r2_score(y, y_pred)
                        if p_val < 0.05:
                            label = 'p:' + stars(p_val) + ' r2=' + str(round(r2, 2))
                        else:
                            label = 'non-significant'
                        print(df_unit_)
                        # sns.scatterplot(data=df_unit_, x=c[0], y=c[1], color=palette[i_u], marker='.', ax=ax,
                        #                 size='force', size_order=['lf', 'mf', 'sf'], sizes=(300, 70))#
                        sns.lineplot(x=x, y=y_pred, color=palette[i_u], ax=ax, linewidth=5, label=label)
                    else:
                        sns.regplot(data=df_unit_, x=c[0], y=c[1], color=palette[i_u], lowess=True, marker='.', ci=None, ax=ax)

                # sns.regplot(data=df_unit, x=c[0], y=c[1], lowess=True, scatter=False, ci=None, ax=ax,
                #             line_kws=dict(alpha=0.2, color='grey', linewidth=20))
                # corr = stats.spearmanr(df_unit[c[0]].values, df_unit[c[1]].values)
                # x, y = df_unit[c[0]].values.reshape(-1, 1), df_unit[c[1]].values.reshape(-1, 1)
                # model = LinearRegression().fit(x, y)
                # y_pred = model.predict(x)
                # p_val = corr.pvalue

                x, y = df_unit[c[0]].values, df_unit[c[1]].values
                if c[0] == 'velAbs_mean':
                    p0 = [50, 0.5, 50]
                    y_pred = fit_plot_sigmoid(x, y, p0)
                    _, p_val = F_test_regression(y, y_pred, len(p0))
                if c[0] == 'depth_mean':
                    model = np.poly1d(np.polyfit(x, y, 1))
                    y_pred = model(x)
                    _, p_val = F_test_regression(y, y_pred, 2)
                if unit_order[i] == 'CT':
                    model = np.poly1d(np.polyfit(x, y, 2))
                    y_pred = model(x)
                    _, p_val = F_test_regression(y, y_pred, 3)
                sns.lineplot(x=x, y=y_pred, color='grey', ax=ax, linewidth=15, alpha=0.4)

                r2 = r2_score(y, y_pred)
                if  p_val < 0.05:
                    ax.set_title(unit_order[i] + ' p:' + stars(p_val) + ' r2=' + str(round(r2, 2)),
                                 color='orange', fontsize=label_size)
                else:
                    ax.set_title(unit_order[i], fontsize=label_size)
                ax.xaxis.set_tick_params(labelsize=tick_size)
                ax.yaxis.set_tick_params(labelsize=tick_size)
                ax.xaxis.label.set_size(label_size)
                ax.yaxis.label.set_size(label_size)
                handles, labels = ax.get_legend_handles_labels()
                new_handles, new_labels = [], []
                for i in range(len(labels)):
                    if labels[i] not in ['lf', 'mf', 'sf']:
                        new_handles.append(handles[i])
                        new_labels.append(labels[i])
                ax.legend(new_handles, new_labels, frameon=False, loc=0, fontsize=legend_size, handletextpad=0.3,
                          markerscale=0.3, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            sns.despine(trim=True)
            axes[1, 2].axis('off')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4)
            plt.savefig(dir_temp + '_'.join([c[0], p_i]) + '.png', dpi=200)

def plot_reg_picked_feature_force_split(df_combined):
    ## ------------ regression for vel and depth with force split------------------

    plot_c = [['velAbs_mean', 'iff_mean'], ['depth_mean', 'n_spike']] #
    for c in plot_c:
        for p_i in ['stroke', 'tap', 'all']:
            if p_i != 'all':
                df_plot = df_combined[df_combined['stimulus'] == p_i]
            else:
                df_plot = df_combined

            fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
            fig.set_size_inches(16, 9)
            fig.suptitle('_'.join([p_i, c[0]]), fontsize=label_size)

            for i in range(len(unit_order)):
                df_unit = df_plot[df_plot['type'] == unit_order[i]]
                ax = axes[i//3, i%3]

                units = df_unit['unit'].unique()
                palette = sns.color_palette('Set2', n_colors=len(units))
                for i_u in range(len(units)):
                    df_unit_ = df_unit[df_unit['unit'] == units[i_u]]
                    for i_f in range(3):
                        forces = ['lf', 'mf', 'sf']
                        df_unit_force = df_unit_[df_unit_['force'] == forces[i_f]]
                        colors = sns.dark_palette(palette[i_u], n_colors=5, reverse=True)
                        sns.regplot(data=df_unit_force, x=c[0], y=c[1], color=colors[i_f], order=2,
                                    marker='.', ci=None, label=units[i_u]+'_'+forces[i_f], ax=ax)

                ax.set_title(unit_order[i], fontsize=label_size)
                ax.xaxis.set_tick_params(labelsize=tick_size)
                ax.yaxis.set_tick_params(labelsize=tick_size)
                ax.xaxis.label.set_size(label_size)
                ax.yaxis.label.set_size(label_size)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, frameon=False, loc=0, fontsize=legend_size, handletextpad=0.3,
                          markerscale=1, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            sns.despine(trim=True)
            axes[1, 2].axis('off')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4)
            plt.savefig(plot_dir + '/contact_neural_reg/' + '_'.join([c[0], p_i]) + '_force_split.png', dpi=200)

def plot_n_spike_duration(df_combined):
    for p_i in ['stroke', 'tap', 'all']:
        if p_i != 'all':
            df_plot = df_combined[df_combined['stimulus'] == p_i]
        else:
            df_plot = df_combined

        ###---------- n_spike & force level & contact duration
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(12, 5)
        fig.suptitle('_'.join([p_i]), fontsize=label_size)
        plot_feat = ['force', 'duration', 'n_spike']
        plot_var = 'length' #'duration'
        for i in range(3):
            ax = axes[i]
            if i == 0: x = 'force'; y = 'n_spike'
            if i == 1: x = 'force'; y = plot_var
            if i == 2: x = plot_var; y = 'n_spike'
            if i in [0, 1]:
                sns.boxplot(data=df_plot, x=x, y=y, showfliers=False, order=['lf', 'mf', 'sf'],  palette='flare', ax=ax)
            if i == 2:
                sns.regplot(data=df_plot, x=x, y=y, order=3, scatter=False, ci=None, color='tab:green', ax=ax)
                sns.scatterplot(data=df_plot, x=x, y=y, hue='force', hue_order=['lf', 'mf', 'sf'],  palette='flare', ax=ax)

            # ax.set_ylim([-0.2, 1])
            # ax.set_xlabel('')
            # ax.get_legend().remove()
            sns.despine()
        plt.tight_layout()
        plt.savefig(plot_dir + '/contact_neural_reg/n_spike_'+p_i+'_'+plot_var+'.png', dpi=200)

def plot_contact_neural_histogram(df_contact, df_neural, name):

    #### ---------- histgram per unit type ------------
    var_contact, var_neural, unit_list, stimuli_list = [], [], [], []
    for i_p in ['contact', 'neural']:
        if i_p == 'contact':
            df, plot_var = df_contact, 'velAbs_mean'
        if i_p == 'neural':
            df, plot_var = df_neural, 'iff_mean'

        fig, axes = plt.subplots(5, 2, sharex='all')#, sharey='all'
        fig.set_size_inches(16, 12, forward=True)
        palette = sns.color_palette('Set2')#sns.cubehelix_palette(n_colors=5)
        for i in range(len(unit_order)):
            df_unit = df[df['type'] == unit_order[i]]
            for j in range(2):
                stimuli = ['stroke', 'tap']
                df_plot = df_unit[df_unit['stimulus'] == stimuli[j]]
                df_plot_var = df_unit[df_unit['stimulus'] == stimuli[j]]
                if i_p == 'contact':
                    feats = [i + '_mean' for i in features]
                    for feat in feats:
                        max_value = df_plot_var[feat].max()
                        min_value = df_plot_var[feat].min()
                        df_plot_var[feat] = (df_plot_var[feat] - min_value) / (max_value - min_value)
                    vars = list(df_plot_var[feats].var().values)
                    var_contact.append(vars)
                if i_p == 'neural':
                    feats = ['iff_mean', 'n_spike', 'peak']
                    for feat in feats:
                        max_value = df_plot_var[feat].max()
                        min_value = df_plot_var[feat].min()
                        df_plot_var[feat] = (df_plot_var[feat] - min_value) / (max_value - min_value)
                    vars = list(df_plot_var[feats].var().values)
                    var_neural.append(vars)
                    unit_list.append(unit_order[i])
                    stimuli_list.append(stimuli[j])

                ax = axes[i, j]
                n_hue = len(df_plot['vel'].unique())
                sns.histplot(df_plot, x=plot_var, hue='vel', bins=50, multiple="stack",
                              palette=palette[:n_hue], ax=ax)
                sns.kdeplot(df_plot, x=plot_var, hue='vel', palette=palette[:n_hue], ax=ax)
                ax.set_title(unit_order[i] + '_' + stimuli[j], size=label_size)
                ax.set_ylabel(plot_var, fontsize=label_size)
                plt.tight_layout()
                sns.despine(trim=True)
        fig.savefig(plot_dir + '/contact_neural_reg/adapt_' + name + '_' + plot_var + '_per_unit.png', dpi=200)

    data = np.hstack((var_contact, var_neural, np.vstack(unit_list), np.vstack(stimuli_list)))
    df_var = pd.DataFrame(data=data, columns=[i + '_mean' for i in features] + ['iff_mean', 'n_spike', 'peak', 'type', 'stimulus'])
    df_var_melt = pd.melt(df_var, id_vars=['type', 'stimulus'], value_vars=[i + '_mean' for i in features] + ['iff_mean', 'n_spike', 'peak'])

    fig, axes = plt.subplots(5, 2, sharex='all')  # , sharey='all'
    fig.set_size_inches(16, 12, forward=True)
    palette = sns.color_palette('Set2')  # sns.cubehelix_palette(n_colors=5)
    for i in range(len(unit_order)):
        df_unit = df_var_melt[df_var_melt['type'] == unit_order[i]]
        for j in range(2):
            ax = axes[i, j]
            stimuli = ['stroke', 'tap']
            df_plot = df_unit[df_unit['stimulus'] == stimuli[j]]
            df_plot['value'] = df_plot['value'].astype('float')
            sns.barplot(df_plot, x='variable', y='value', palette=palette, ax=ax)
            ax.set_title(unit_order[i] + '_' + stimuli[j], size=label_size)
            ax.set_ylabel(plot_var, fontsize=label_size)
            plt.tight_layout()
            sns.despine(trim=True)
    fig.savefig(plot_dir + '/contact_neural_reg/adapt_' + name + '_var_mean_only.png', dpi=200)

    print(data)

def vis_contact_condition(df_contact, df_neural, name, per_repeat=False):
    # df_contact = pd.read_csv(data_dir + 'contact_features.csv')
    contact_feat = [i + '_mean' for i in features]
    contact_conditions = {'velAbs_mean': 'vel', 'velLong_mean': 'vel', 'velVert_mean': 'vel', 'velLat_mean': 'vel',
                          'depth_mean': 'force', 'area_mean': 'finger'}

    # df_neural = pd.read_csv(data_dir + 'neural_features.csv')
    neural_feat = ['iff_mean', 'iff_variation', 'n_spike', 'max_n_spike', 'peak', 'iff_entropy']
    df_neural_temp = df_neural.drop(columns=['type', 'stimulus', 'vel', 'finger', 'force'])

    df_combined = pd.merge(df_contact, df_neural_temp, on=['unit', 'trial_id'])
    df_combined.to_csv(data_dir + 'combined_features.csv', index=False, header=True)

    df_combined = df_combined[df_combined['finger'] != ' two finger pads']
    df_combined.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                         'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}},
                        inplace=True)
    df_combined.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                         'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}},
                        inplace=True)
    if per_repeat:
        dir_temp = plot_dir + '/per_repeat/contact_neural_reg/'
    else:
        dir_temp = plot_dir + '/contact_neural_reg/'
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    ### ---------- histogram per unit -----------------
    # plot_contact_neural_histogram(df_contact, df_neural, name)

    ### ---------- histgram of contact attributes -------------
    # plot_all_combinations(contact_feat, df_combined, neural_feat, contact_conditions, name)

    ## ------------ regression for vel and depth ------------------
    plot_reg_picked_feature(df_combined, dir_temp)

    # plot_reg_picked_feature_force_split(df_combined)

    # plot_n_spike_duration(df_combined)


    plt.show()

def adapt_contact_helper(method, df):
    has_nan = np.isnan(np.min(df['velAbs_mean'].values))
    if has_nan:
        data_raw = df['velAbs_mean'].values
        data = data_raw[~np.isnan(data_raw)]
        data = data.reshape(-1, 1)
        label = df['vel'].values
        print('data len:', len(data), 'label len:', len(label))
        # label = label[~np.isnan(data_raw)]
    else:
        data = df['velAbs_mean'].values.reshape(-1, 1)
        label = df['vel'].values

    # #------------ Kmeans clutstering
    if method == 'KMeans':
        kmeans = KMeans(n_clusters=5, init=[[1], [3], [9], [18], [24]], n_init="auto").fit(data)

        idx_ord = np.argsort(kmeans.cluster_centers_[:, 0])
        label_ord = np.zeros_like(kmeans.labels_) - 1
        for i in range(len(idx_ord)):
            label_ord[kmeans.labels_==idx_ord[i]] = i

        label_dict = {0:1, 1:3, 2:9, 3:18, 4:24}
        label = [label_dict[i] for i in label_ord]

    # ------------ find the closest label
    def closest(lst, num):
        lst = np.asarray(lst)
        idx = (np.abs(lst - num)).argmin()
        return lst[idx]

    if method == 'closest':
        for i in range(len(data)):
            if data[i] >= 6:
                label[i] = closest([1, 3, 9, 18, 24], data[i])

    # ------------ variational Gaussian Mixture (find the number of clusters)
    if method == 'VarGaussianMix':
        n = 8
        bgm = BayesianGaussianMixture(n_components=n, weight_concentration_prior=100, random_state=0).fit(data)
        pred = bgm.predict(data)
        n_new = len(np.unique(pred))
        print('number of cluster:', str(n_new))

        valid_mean = [bgm.means_[i,0] for i in np.unique(pred)]
        idx_ord = np.argsort(valid_mean)
        print(valid_mean, idx_ord)
        label_ord = np.zeros_like(pred) - 1
        for i in range(len(idx_ord)):
            label_ord[pred == np.sort(np.unique(pred))[idx_ord[i]]] = i
        label_ = label_ord + 1#['# ' + str(int(i)) for i in label_ord + 1]
        if has_nan:
            label = np.array([np.nan] * len(label))
            label[~np.isnan(data_raw)] = label_
        else:
            label = label_

    return label

def adapt_contact_condition(method='actual', split_gesture=False, per_repeat=False, vis=False):
    """
    :param method:
        'actual' for using raw data
        'VarGaussianMix' for changing velocity labels by clustering
    :param split_gesture:
        True for clustering for tapping and stroking separately
        False for clustering tapping and stroking data combined
    :param per_repeat:
        True for analysis per repeat
        False for analysis per trial
    :param vis:
        True for plotting additional figures in vis_contact_condition()
    :return:
        contact dataframe and neural dataframe with new velocity labels
    """
    if per_repeat:
        df_neural = pd.read_csv(os.path.join(data_dir, 'neural_features_per_repeat.csv'))
        df_contact = pd.read_csv(os.path.join(data_dir, 'contact_features_per_repeat.csv'))
    else:
        df_neural = pd.read_csv(os.path.join(data_dir, 'neural_features.csv'))
        df_contact = pd.read_csv(os.path.join(data_dir, 'contact_features.csv'))
    feat_list = [i + '_mean' for i in features]

    if method != 'actual':
        name = method + '_' + 'split_ges' if split_gesture else method + '_' + 'combine_ges'
        if split_gesture:
            for i_g in ['stroke', 'tap']:
                df_ges = df_contact[df_contact['stimulus'] == i_g]
                label = adapt_contact_helper(method, df_ges)
                df_contact.loc[df_contact['stimulus'] == i_g, 'vel'] = label
                df_neural.loc[df_neural['stimulus'] == i_g, 'vel'] = label
        else:
            label = adapt_contact_helper(method, df_contact)
            df_contact['vel'] = label
            df_neural['vel'] = label
    else:
        name = method

    fig = plot_contact_hist(df_contact, 'velAbs_mean')

    if per_repeat:
        output_path_abs = os.path.join(plot_dir,  'per_repeat/adapt_')
        if not os.path.exists(output_path_abs):
            os.makedirs(output_path_abs)
        fig.savefig(os.path.join(output_path_abs, name + '_velAbs_mean.png'), dpi=200)
    else:
        output_path_abs = os.path.join(plot_dir,  'contact_neural_reg/adapt_')
        if not os.path.exists(output_path_abs):
            os.makedirs(output_path_abs)
        fig.savefig(os.path.join(output_path_abs, name + '_velAbs_mean.png'), dpi=200)

    if vis:
        vis_contact_condition(df_contact, df_neural, name, per_repeat=per_repeat)

    return df_contact, df_neural


def plot_confusion_matrix(cm,ax,labels,normalize=True,title=None,label_combined=None):
    color = 'tab:blue'
    tick_size = 12

    accuracy = np.trace(cm) / np.sum(cm).astype('float')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax.set_title(title, fontsize=tick_size+4)
    ax.set_title(title + ': {:0.1f}%'.format(accuracy*100), fontsize=tick_size+2) # for gesture
    annot = True; lw = 0
    if label_combined == 'all': annot = False; lw = 0
    if label_combined == 'no': annot = True; lw = 1.5
    if label_combined == 'stimulus_vel': 
        annot = cm*100; annot = annot.astype('int').astype('str'); annot[annot=='0']=''; lw = 0
        sns.heatmap(cm*100,annot=annot,annot_kws={"size":tick_size-1},xticklabels=labels,yticklabels=labels,cbar=False,fmt='', 
                cmap=sns.light_palette(color, 20),vmin=0,vmax=100,linewidth=lw,square=True, ax=ax)
    else:
        sns.heatmap(100*cm,annot=annot,annot_kws={"size":tick_size-1},xticklabels=labels,yticklabels=labels,cbar=False,
                    cmap=sns.light_palette(color, 20),vmin=0,vmax=100,linewidth=lw,square=True, ax=ax)

    # cbar = ax.collections[0].colorbar
    # if cbar: cbar.ax.tick_params(labelsize=tick_size-2,size=0)
    ax.set_xlabel('Accuracy: {:0.1f}%'.format(accuracy*100), fontsize=tick_size+2)
    ax.set_xlabel('', fontsize=tick_size+2) # for gesture
    ax.set_ylabel('True', fontsize=tick_size)
    ax.tick_params(labelsize=tick_size,length=0)
    [ax.tick_params(axis=a,colors=(.4,.4,.4)) for a in ['x','y']]
    plt.yticks(rotation=0)

def classification_CV_n_folds(x, y, method, label_order, n_folds=5):
    cms, corrects, preds, importances = [], [], [], []
    for i in range(20):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
        if method == 'RF':
            clf = RandomForestClassifier(class_weight='balanced')
        if method == 'SVM':
            clf = SVC(class_weight='balanced')
        y_pred = cross_val_predict(clf,x,y,cv=cv)
        corrects.append([1 if y_pred[i] == y[i] else 0 for i in range(len(y))])
        preds.append(y_pred)
        # clf.fit(x,y)
        cm = confusion_matrix(y,y_pred,labels=label_order)
        cms.append(cm)
        clf.fit(x, y)
        importances.append(list(permutation_importance(clf, x, y).importances_mean))
    cm_aff = np.sum(cms, axis=0)
    correct_aff = np.sum(corrects, axis=0)
    pred_aff = ['_'.join([str(i[j]) for i in preds]) for j in range(len(preds[0]))]
    resutls_aff = [correct_aff, pred_aff, importances]

    return cm_aff, resutls_aff

def pca_helper(x, df_feature, feat_list, dir_temp, unit=''):
    pca = PCA()
    n_component = 6
    x_pca = pca.fit_transform(x)[:, :n_component]
    print(pca.explained_variance_ratio_.cumsum())
    loadings = pd.DataFrame(pca.components_[:n_component, :].T,
                            columns=['PC' + str(i + 1) for i in range(n_component)], index=feat_list)
    print(loadings)
    x = x_pca
    name_ = '_pca' + unit
    feat_imp = ['PC' + str(i + 1) for i in range(n_component)]

    fig = plt.figure()
    fig.set_size_inches(6, 4)
    ax = fig.add_subplot(1, 1, 1)
    sns.lineplot(pca.explained_variance_ratio_.cumsum(), ax=ax)
    ax.set_xlabel('# of PC', fontsize=label_size)
    ax.set_ylabel('explained variance', fontsize=label_size)
    ax.set_title(unit)
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(dir_temp + '/clf_PCA_variance' + name_ + '.png', dpi=200)

    fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
    fig.set_size_inches(12, 6, forward=True)
    for i in range(n_component):
        ax = axes[i // 3, i % 3]
        sns.barplot(x=list(range(len(feat_list))), y=loadings['PC' + str(i + 1)].values, ax=ax)
        ax.set_xticklabels(feat_list)
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_ylabel('correlation')
    fig.suptitle(unit)
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(dir_temp + '/clf_PCA_PCcorrelation' + name_ + '.png', dpi=200)

    df_feat_corr = df_feature.loc[:, feat_list]
    cov = df_feat_corr.corr()
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cov, annot=True, cmap='vlag', vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(unit)
    fig.tight_layout()
    fig.savefig(dir_temp + '/clf_iff_feautre_cov' + name_ + '.png', dpi=200)

    return x, feat_imp


def clf_contact_condition(data_source, clf_method, adapt_method='actual', split_gesture=False, apply_pca=False, per_repeat=False):
    """
    :param data_source:
        'contact': classify based on contact data
        'neural': classify based on neural data
    :param clf_method:
        'RF': random forest classifier
    :param adapt_method:
        'actual' for using raw data
        'VarGaussianMix' for changing velocity labels by clustering
    :param split_gesture:
        True for clustering for tapping and stroking separately
        False for clustering tapping and stroking data combined
    :param apply_pca:
        True: Apply PCA to features before classification
        False: classify using all features
    :param per_repeat:
        True for analysis per repeat
        False for analysis per trial
    :return: None plotting confusion matrix instead
    """
    df_contact, df_neural = adapt_contact_condition(adapt_method, split_gesture, per_repeat)
    if data_source == 'neural':
        df_feature = df_neural
        feat_list = ['iff_mean','iff_variation','n_spike','max_n_spike','peak','iff_entropy']
        feat_list = ['iff_mean', 'iff_variation', 'n_spike', 'max_n_spike', 'iff_high_qtr', 'peak', 'iff_entropy', \
                     'freq_max', 'freq_amp_max', 'freq_centroid', 'freq_entropy']
    if data_source == 'contact':
        df_feature = df_contact
        feat_list = [i+'_mean' for i in features]
        # feat_list = feat_list + [i+'_cv' for i in features]
    df_feature = df_feature.dropna(subset=feat_list)

    df_feature = df_feature[df_feature['finger'] != ' two finger pads']
    df_feature = df_feature[df_feature['finger'] != 'two finger pads']
    df_feature.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                        'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}}, 
                        inplace=True)
    df_feature.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                        'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}},
                       inplace=True)
    split_str = 'split_ges' if split_gesture else 'combine_ges'
    plot_name = '/label_adapt_' + '_'.join([adapt_method, split_str, clf_method])
    if per_repeat:
        dir_temp = plot_dir+'/per_repeat/CMs/'+data_source+plot_name+'_all_features'
    else:
        dir_temp = plot_dir+'/CMs/'+data_source+plot_name+'_all_features'
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    ### ---------------- label all
    label_combined = 'all'
    df_feature.replace({'stroke': 'S','tap': 'T'}, inplace=True)
    df_feature.sort_values(by=['stimulus', 'finger', 'vel', 'force'], inplace=True)
    df_feature['label'] = df_feature['stimulus'] + '_' + df_feature['finger'] + '_' + df_feature['vel'].astype('int').astype('str')
    label_order = df_feature['label'].unique()
    print(label_order)

    fig = plt.figure()
    fig.set_size_inches(12, 8,forward=True)
    plot_i = 1
    for unit_type in unit_order:
        df_type = df_feature[df_feature['type'] == unit_type]
        print(unit_type)
        x = df_type.loc[:, feat_list].values
        x = StandardScaler().fit_transform(x)
        y = df_type['label'].values
        print(df_type.groupby(['label']).size())

        cm_aff, _ = classification_CV_n_folds(x, y, clf_method, label_order, n_folds=5)

        ax = fig.add_subplot(2, 3, plot_i)
        plot_confusion_matrix(cm_aff,ax,labels=label_order,normalize=True,title=unit_type+' ('+str(len(y))+')',label_combined=label_combined)
        grid_stimuli, grid_finger = len(label_order) // 2, len(label_order) // 4
        ax.hlines([grid_finger, grid_stimuli+grid_finger], *ax.get_xlim(), color="lightgray")
        ax.vlines([grid_finger, grid_stimuli+grid_finger], *ax.get_xlim(), color="lightgray")
        ax.hlines([grid_stimuli], *ax.get_xlim(), color="gray")
        ax.vlines([grid_stimuli], *ax.get_xlim(), color="gray")
        plot_i += 1
    plt.subplots_adjust(left=0.086, right=0.99, top=0.943, bottom=0.107, hspace=0.39, wspace=0.28)

    fig.savefig(dir_temp+'/clf_accuracy_All.png', dpi=200)

    #-------------- lanel no
    label_combined = 'no'
    df_feature.sort_values(by=['type', 'stimulus', 'finger', 'vel', 'force'], inplace=True)

    for unit_type in unit_order:
        df_type = df_feature[df_feature['type'] == unit_type]
        print(unit_type)

        fig = plt.figure()
        fig.set_size_inches(9, 3,forward=True)
        plot_i = 1
        for label in ['stimulus', 'finger', 'vel', 'force']:
            x = df_type.loc[:, feat_list].values
            x = StandardScaler().fit_transform(x)
            y = df_type[label].values
            label_order = df_type[label].unique()
            print(label_order, df_type.groupby([label]).size())

            cm_aff, _ = classification_CV_n_folds(x, y, clf_method, label_order, n_folds=5)

            ax = fig.add_subplot(1, 4, plot_i)
            cbar = False
            plot_confusion_matrix(cm_aff,ax,labels=label_order,normalize=True,title=label+' ('+str(len(y))+')',label_combined=label_combined)
            plot_i += 1
        fig.suptitle(unit_type, fontsize=label_size)
        plt.subplots_adjust(left=0.063, right=0.98, top=0.943, bottom=0.107, hspace=0.512, wspace=0.28)
        fig.savefig(dir_temp+'/clf_accuracy'+'_'+unit_type+'.png', dpi=200)

    #------------ label stmi vel
    label_combined = 'stimulus_vel'
    df_feature.replace({'stroke': 'S','tap': 'T'}, inplace=True)
    df_feature.sort_values(by=['stimulus', 'finger', 'vel', 'force'], inplace=True)
    df_feature['label'] = df_feature['stimulus'] + '_' + df_feature['vel'].astype('int').astype('str')
    label_order = df_feature['label'].unique()
    print(label_order)

    fig = plt.figure()
    fig.set_size_inches(12, 8,forward=True)
    plot_i = 1
    df_clf = pd.DataFrame()
    for unit_type in unit_order:
        df_type = df_feature[df_feature['type'] == unit_type]
        print(unit_type)
        x = df_type.loc[:, feat_list].values
        x = StandardScaler().fit_transform(x)
        y = df_type['label'].values
        print(df_type.groupby(['label']).size())

        name_ = ''
        feat_imp = feat_list
        if apply_pca:
            x, feat_imp = pca_helper(x, df_type, feat_list, dir_temp, unit_type)
            name_ = '_pca'

        cm_aff, results_aff = classification_CV_n_folds(x, y, clf_method, label_order, n_folds=5)
        correct_aff, pred_aff = results_aff[0], results_aff[1]
        df_type['correct_pred'] = correct_aff
        df_type['pred'] = pred_aff
        df_clf = pd.concat([df_clf, df_type], ignore_index=True)

        if apply_pca:
            importances = results_aff[2]
            df_imp = pd.DataFrame(data=importances, columns=feat_imp)
            df_imp = pd.melt(df_imp, value_vars=feat_imp)

            fig1 = plt.figure()
            fig1.set_size_inches(6, 4)
            ax1 = fig1.add_subplot(1, 1, 1)
            sns.pointplot(df_imp, x='variable', y='value', ax=ax1)
            ax1.set_title(unit_type)
            ax1.xaxis.set_tick_params(rotation=90)
            fig1.tight_layout()
            sns.despine(trim=True)
            fig1.savefig(dir_temp + '/clf_importance' + name_ + unit_type + '.png', dpi=200)

        ax = fig.add_subplot(2, 3, plot_i)
        plot_confusion_matrix(cm_aff,ax,labels=label_order,normalize=True,title=unit_type+' ('+str(len(y))+')',label_combined=label_combined)
        grid_stimuli = len(label_order) // 2        
        ax.hlines([grid_stimuli], *ax.get_xlim(), color="gray")
        ax.vlines([grid_stimuli], *ax.get_xlim(), color="gray")
        plot_i += 1
    fig.subplots_adjust(left=0.086, right=0.99, top=0.943, bottom=0.107, hspace=0.39, wspace=0.28)
    fig.savefig(dir_temp+'/clf_accuracy_combined'+name_+'.png', dpi=200)
    # df_clf.to_csv(data_dir+'neural_features_clf.csv', index = False, header=True)

    plt.show()

def clf_neuron_type(clf_method, adapt_method='actual', split_gesture=False, apply_pca=False, per_repeat=False):
    """
    :param clf_method:
        'RF': random forest classifier
    :param adapt_method:
        'actual' for using raw data
        'VarGaussianMix' for changing velocity labels by clustering
    :param split_gesture:
        True for clustering for tapping and stroking separately
        False for clustering tapping and stroking data combined
    :param apply_pca:
        True: Apply PCA to features before classification
        False: classify using all features
    :param per_repeat:
        True for analysis per repeat
        False for analysis per trial
    :return: None plotting confusion matrix instead
    """
    _, df_feature = adapt_contact_condition(adapt_method, split_gesture, per_repeat)
    feat_list = ['iff_mean', 'iff_variation', 'n_spike', 'max_n_spike', 'peak', 'iff_entropy']
    feat_list = ['iff_mean', 'iff_variation', 'n_spike', 'max_n_spike', 'iff_high_qtr', 'peak', 'iff_entropy', \
                 'freq_max', 'freq_amp_max', 'freq_centroid', 'freq_entropy']
    df_feature = df_feature.dropna(subset=feat_list)

    df_feature = df_feature[df_feature['finger'] != ' two finger pads']
    df_feature = df_feature[df_feature['finger'] != 'two finger pads']
    df_feature.replace({'finger': {' one finger tip': '1f', ' whole hand': 'wh'},
                        'force': {' light force': 'lf', ' moderate force': 'mf', ' strong force': 'sf'}},
                        inplace=True)
    df_feature.replace({'finger': {'one finger tip': '1f', 'whole hand': 'wh'},
                        'force': {'light force': 'lf', 'moderate force': 'mf', 'strong force': 'sf'}},
                       inplace=True)

    df_feature.replace({'stroke': 'S', 'tap': 'T'}, inplace=True)
    df_feature.sort_values(by=['stimulus', 'vel', 'finger', 'force'], inplace=True)

    split_str = 'split_ges' if split_gesture else 'combine_ges'
    plot_name = '/unittype_' + '_'.join([adapt_method, split_str, clf_method])
    if per_repeat:
        dir_temp = plot_dir+'/per_repeat/CMs/neural/'+plot_name+'_all_features'
    else:
        dir_temp = plot_dir+'/CMs/neural/'+plot_name+'_all_features'
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    df_feature['condition'] = df_feature['stimulus'] + '_' + df_feature['vel'].astype('int').astype('str')
    c_order = df_feature['condition'].unique()
    print(c_order)

    fig1 = plt.figure()
    fig1.set_size_inches(16, 8, forward=True)
    fig2, axes = plt.subplots(2, 5, sharex='all', sharey='all')
    fig2.set_size_inches(16, 8, forward=True)
    plot_i = 1
    name_ = ''
    feat_imp = feat_list
    for c in c_order :
        df_c = df_feature[df_feature['condition'] == c]

        print(c)
        x = df_c.loc[:, feat_list].values
        x = StandardScaler().fit_transform(x)
        if apply_pca:
            n_components = 6
            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(x)
            print(pca.explained_variance_ratio_.cumsum())
            loadings = pd.DataFrame(pca.components_.T, columns=['PC'+str(i+1) for i in range(n_components)], index=feat_list)
            print(loadings)
            x = x_pca
            name_='_pca'
            feat_imp = ['PC'+str(i+1) for i in range(n_components)]

        y = df_c['type'].values
        print(df_c.groupby(['type']).size())

        cm_aff, results_aff = classification_CV_n_folds(x, y, clf_method, unit_order, n_folds=5)
        importances = results_aff[2]
        df_imp = pd.DataFrame(data=importances, columns=feat_imp)
        df_imp = pd.melt(df_imp, value_vars=feat_imp)

        ax1 = fig1.add_subplot(2, 5, plot_i)
        plot_confusion_matrix(cm_aff, ax1, labels=unit_order, normalize=True,
                              title=c + ' (' + str(len(y)) + ')', label_combined='no')

        ax2 = axes[(plot_i-1)//5, (plot_i-1)%5] #fig2.add_subplot(2, 5, plot_i)
        sns.pointplot(df_imp, x='variable', y='value', ax=ax2)
        ax2.xaxis.set_tick_params(rotation=90)
        plot_i += 1
    fig1.subplots_adjust(left=0.086, right=0.99, top=0.943, bottom=0.107, hspace=0.39, wspace=0.28)
    fig1.savefig(dir_temp + '/clf_accuracy'+name_+'.png', dpi=200)
    plt.tight_layout()
    sns.despine(trim=True)
    fig2.savefig(dir_temp + '/clf_importance'+name_+'.png', dpi=200)


    x = df_feature.loc[:, feat_list].values
    x = StandardScaler().fit_transform(x)
    y = df_feature['type'].values
    print(df_feature.groupby(['type']).size())

    if apply_pca:
        x, feat_imp = pca_helper(x, df_feature, feat_list, dir_temp)
        name_ = '_pca'

    cm_aff, results_aff = classification_CV_n_folds(x, y, clf_method, unit_order, n_folds=5)
    importances = results_aff[2]
    df_imp = pd.DataFrame(data=importances, columns=feat_imp)
    df_imp = pd.melt(df_imp, value_vars=feat_imp)

    fig = plt.figure()
    fig.set_size_inches(4, 4, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    plot_confusion_matrix(cm_aff, ax, labels=unit_order, normalize=True,
                          title='combined (' + str(len(y)) + ')', label_combined='no')
    plt.subplots_adjust(left=0.086, right=0.99, top=0.943, bottom=0.107, hspace=0.39, wspace=0.28)
    fig.savefig(dir_temp + '/clf_accuracy_combiend'+name_+'.png', dpi=200)

    fig = plt.figure()
    fig.set_size_inches(4, 4, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    sns.pointplot(df_imp, x='variable', y='value', ax=ax)
    ax.xaxis.set_tick_params(rotation=90)
    plt.tight_layout()
    sns.despine(trim=True)
    fig.savefig(dir_temp + '/clf_importance_combiend'+name_+'.png', dpi=200)
    plt.show()

if __name__ == '__main__':

    sns.set(style="ticks", font='Arial')


    ### --------- classification ----------
    adapt_contact_condition(method='actual', split_gesture=False, per_repeat=True, vis=True) #VarGaussianMix
    # clf_contact_condition(data_source='neural', clf_method='RF', adapt_method='actual', split_gesture=False, apply_pca=False, per_repeat=True)
    #clf_neuron_type(clf_method='RF', adapt_method='actual', split_gesture=False, apply_pca=False, per_repeat=True)

