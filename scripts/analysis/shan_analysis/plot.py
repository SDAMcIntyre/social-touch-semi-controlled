import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from params import *

class plot_results:
    def __init__(self, run_mode, method):
        self.run_mode = run_mode
        self.method = method
        if method == 'LR':
            self.plot_regression_results()
        if method == 'P_CORR':
            self.plot_p_corr_results()
        if method == 'CCA':
            self.plot_cca_results()
        if method in ['LF', 'LN_poisson', 'LN_relu', 'LN_poly']:
            self.plot_LN_results()
        if method == 'LR_LNP':
            if self.run_mode == 'per_trial':
                # self.plot_LR_LNP_results()
                self.plot_LR_LNP_results_per_trial()
            if self.run_mode == 'per_unit':
                self.plot_LR_LNP_results_per_type()
        if method == 'LR_ARIMA':
            self.plot_LR_ARIMA_results()

    def load_data(self):
        sns.set(style="ticks",font_scale=1)

        df = pd.read_csv(data_dir + self.method + '_Results.csv')
        df = df[(df['finger'] != 'two finger pads')]
        df.replace({'whole hand': 'hand', 'one finger tip': 'finger',
                        'light force': 'low', 'moderate force': 'mid', 'strong force': 'high'}, inplace=True)
        df.sort_values(['unit_type', 'stimulus', 'vel'], inplace=True)

        return df

    def plot_regression_results(self):

        df_LR = self.load_data()

        for y_i in ['imp_' + i for i in contact_feat]:
            df_LR[y_i] = df_LR[y_i].abs()
        df_LR['combined'] = df_LR['unit_type'] + '_' + df_LR['stimulus'] #+ df_LR['vel'].astype('str')

        # plot accuracy/importance vs. contact/neural type
        for y_i in ['r2', 'mse'] + ['imp_' + i for i in contact_feat]:
            fig, axes = plt.subplots(2, 3)
            fig.set_size_inches(12, 6, forward=True)
            x_list = ['unit_type', 'stimulus', 'vel', 'finger', 'force']
            for i in range(5):
                ax = axes[i//3, i%3]
                sns.boxplot(df_LR, x=x_list[i], y=y_i, palette='crest', showfliers=False, ax=ax)
                if x_list[i] == 'unit_type': ax.xaxis.set_tick_params(rotation=30)
            sns.despine(trim=True)
            plt.tight_layout()
            fig.savefig(plot_dir + 'figure_'+ self.method + '/compare_' + y_i +'.jpg')

        # plot accuracy/importance vs. combiend type
        fig, axes = plt.subplots(2, 4, sharex='all')
        fig.set_size_inches(12, 6, forward=True)
        y_list = ['r2', 'mse'] + ['imp_' + i for i in contact_feat]
        for i in range(len(y_list)):
            ax = axes[i//4, i%4]
            sns.boxplot(df_LR, x='combined', y=y_list[i], palette='crest', showfliers=False, ax=ax)
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(rotation=60)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.045, wspace=0.3)
        fig.savefig(plot_dir + 'figure_'+self.method+'/compare_combined.jpg')

        # plot importance
        y_list =  ['imp_' + i for i in contact_feat]
        df_LR_imp = pd.melt(df_LR, id_vars=df_LR.columns.difference(y_list))
        print(df_LR_imp)
        fig, axes = plt.subplots(5, 2, sharex='all', sharey='all')
        fig.set_size_inches(5, 9, forward=True)
        sub_list = [(x, y) for x in unit_list for y in ['stroke', 'tap']]
        print(sub_list)
        for i in range(10):
            df_LR_imp_i = df_LR_imp[(df_LR_imp['unit_type'] == sub_list[i][0]) & (df_LR_imp['stimulus'] == sub_list[i][1])]
            ax = axes[i//2, i%2]
            sns.pointplot(df_LR_imp_i, x='variable', y='value', palette='crest', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('_'.join(sub_list[i]))
            ax.xaxis.set_tick_params(rotation=60)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.07, wspace=0.11)
        fig.savefig(plot_dir + 'figure_'+self.method+'/compare_all_importances.jpg')

        y_list = ['r2', 'mse'] + ['imp_' + i for i in contact_feat]
        for i in range(len(y_list)):
            fig, axes = plt.subplots(2, 3, sharex='all', sharey='all')
            fig.set_size_inches(12, 6, forward=True)
            for ii in range(5):
                unit_i = unit_list[ii]
                ax = axes[ii//3, ii%3]
                df_LR_unit = df_LR[df_LR['unit_type'] == unit_i]
                df_LR_unit.loc[:, 'combined_'] = df_LR_unit['stimulus'] + '_' + df_LR_unit['vel'].astype(str)
                order = ['_'.join([x, y]) for x in ['stroke', 'tap'] for y in ['1', '3', '9', '18', '24']]
                sns.boxplot(df_LR_unit, x='combined_', y=y_list[i], palette='crest', order=order, showfliers=False, ax=ax)
                ax.set_xlabel('')
                ax.set_title(unit_i)
                ax.xaxis.set_tick_params(rotation=30)
            sns.despine(trim=True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.07, wspace=0.13)
            fig.savefig(plot_dir + 'figure_'+self.method+'/compare_combined_' + y_list[i] + '.jpg')

        plt.show()

    def plot_p_corr_results(self):
        df = self.load_data()

        for metric in ['coef_', 'p_corr_', 'corr_']:

            y_list = [metric + i for i in contact_feat]
            for i in y_list:
                df.loc[:, [i]] = df[i].abs()
            df_LR_imp = pd.melt(df, id_vars=['unit_type', 'unit_name', 'block_id', 'trial_id',
                                             'stimulus', 'vel', 'finger', 'force'], value_vars=y_list)
            print(df_LR_imp)
            fig, axes = plt.subplots(5, 2, sharex='all', sharey='all')
            fig.set_size_inches(5, 9, forward=True)
            sub_list = [(x, y) for x in unit_list for y in ['stroke', 'tap']]
            print(sub_list)
            for i in range(10):
                df_LR_imp_i = df_LR_imp[(df_LR_imp['unit_type'] == sub_list[i][0]) & (df_LR_imp['stimulus'] == sub_list[i][1])]
                ax = axes[i//2, i%2]
                sns.pointplot(df_LR_imp_i, x='variable', y='value', palette='crest', ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('_'.join(sub_list[i]))
                ax.xaxis.set_tick_params(rotation=60)
            sns.despine(trim=True)
            plt.tight_layout()
            # plt.subplots_adjust(left=0.07, wspace=0.11)
            fig.savefig(plot_dir + 'figure_'+self.method+'/' + metric + '/compare_all_importances.jpg')

            for ii in range(5):
                df_unit = df[df['unit_type'] == unit_list[ii]]
                fig, axes = plt.subplots(2, 6, sharex='all', sharey='all')
                fig.set_size_inches(18, 6, forward=True)
                for i in range(2):
                    stimuli = ['stroke', 'tap']
                    df_plot = df_unit[df_unit['stimulus'] == stimuli[i]]
                    for j in range(6):
                        ax = axes[i, j]
                        order = [1,3,9,18,24]
                        palette = 'crest'
                        sns.pointplot(df_plot, x='vel', y=y_list[j], palette=palette, order=order, ax=ax)
                        ax.set_xlabel('')
                        ax.set_xlabel(stimuli[i])
                        ax.set_title(unit_list[ii])
                        ax.xaxis.set_tick_params(rotation=0)
                sns.despine(trim=True)
                plt.tight_layout()
                plt.subplots_adjust(left=0.07, wspace=0.13)
                fig.savefig(plot_dir + 'figure_'+self.method+'/' + metric + '/compare_combined_' + unit_list[ii] + '_.jpg')

            for ii in range(5):
                df_unit = df_LR_imp[df_LR_imp['unit_type'] == unit_list[ii]]
                fig, axes = plt.subplots(2, 5, sharex='all', sharey='all')
                fig.set_size_inches(18, 6, forward=True)
                for i in range(2):
                    stimuli = ['stroke', 'tap']
                    df_plot = df_unit[df_unit['stimulus'] == stimuli[i]]
                    for j in range(5):
                        ax = axes[i, j]
                        vels = [1, 3, 9, 18, 24]
                        df_LR_unit = df_plot[df_plot['vel'] == vels[j]]
                        order = y_list
                        palette = 'crest'
                        sns.pointplot(df_LR_unit, x='variable', y='value', palette=palette, order=order, ax=ax)
                        ax.set_xlabel(unit_list[ii])
                        ax.set_ylabel(stimuli[i])
                        ax.set_title(str(vels[j]) + 'cm/s')
                        ax.xaxis.set_tick_params(rotation=30)
                sns.despine(trim=True)
                plt.tight_layout()
                plt.subplots_adjust(left=0.07, wspace=0.13)
                fig.savefig(plot_dir + 'figure_' + self.method + '/' + metric + '/compare_combined_' + unit_list[ii] + '.jpg')

        plt.show()

    def plot_cca_results(self):

        df = self.load_data()

        for y_i in ['corr_r2'] + ['loading_' + i for i in contact_feat]:
            df[y_i] = df[y_i].abs()

        df['combined'] = df['stimulus'] + '_' + df['vel'].astype(str)
        y_list = ['loading_' + i for i in contact_feat]
        for i in range(len(unit_list)):
            fig, axes = plt.subplots(2, 5, sharex='all', sharey='all')
            fig.set_size_inches(12, 6, forward=True)
            unit_i = unit_list[i]
            df_unit = df[df['unit_type'] == unit_i]
            for ii in range(10):
                df.sort_values(['stimulus', 'vel'], inplace=True)
                combined_list = df['combined'].unique()
                combiend_i = combined_list[ii]
                df_combined = df_unit[df_unit['combined'] == combiend_i]
                if len(df_combined.index) == 0: continue
                ax = axes[ii//5, ii%5]
                df_combined.drop(columns=['combined', 'corr_r2', 'loading_max'], inplace=True)
                df_loading = pd.melt(df_combined, id_vars=df_combined.columns.difference(y_list))
                print(df_loading)
                sns.pointplot(df_loading, x='variable', y='value', palette='crest', ax=ax)
                ax.set_xlabel('')
                ax.set_title(unit_i + '_' + combiend_i)
                ax.xaxis.set_tick_params(rotation=30)
            sns.despine(trim=True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.07, wspace=0.13)
            fig.savefig(plot_dir + 'figure_' + self.method + '/compare_combined_' + unit_list[i] + '.jpg')

        fig, axes = plt.subplots(5, 2, sharex='all', sharey='all')
        fig.set_size_inches(5, 9, forward=True)
        sub_list = [(x, y) for x in unit_list for y in ['stroke', 'tap']]
        for i in range(10):
            df_i = df[(df['unit_type'] == sub_list[i][0]) & (df['stimulus'] == sub_list[i][1])]
            ax = axes[i//2, i%2]
            sns.countplot(df_i, x='loading_max', palette='crest', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('Count')
            ax.set_title('_'.join(sub_list[i]))
            ax.xaxis.set_tick_params(rotation=60)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.07, wspace=0.11)
        fig.savefig(plot_dir + 'figure_' + self.method + '/compare_loading_max.jpg')

        plt.show()

    def plot_LN_results(self):
        df = self.load_data()
        df.sort_values(['stimulus', 'unit_type', 'vel'], inplace=True)

        df['combined'] = df['unit_type'] + '_' + df['stimulus']# + '_' + df['vel'].astype('str')
        for y_i in ['corr_r2'] + ['loading_' + i for i in contact_feat]:
            df[y_i] = df[y_i].abs()

        fig, axes = plt.subplots(3,1)
        fig.set_size_inches(7, 8)
        ax = axes[0]
        ax.plot(df['mse_fitted'].values, label='mse_fitted', alpha=0.8)
        ax.plot(df['mse_contact_cca'].values, label='mse_contact_cca', alpha=0.5)
        ax.plot(df['mse_contact_single'].values, label='mse_contact_single', alpha=0.5)
        ax.legend(frameon=False, loc=0)
        ax = axes[1]
        ax.plot(df['r2_fitted'].values, label='r2_fitted', alpha=0.8)
        ax.plot(df['r2_contact_cca'].values, label='r2_contact_cca', alpha=0.5)
        ax.plot(df['r2_contact_single'].values, label='r2_contact_single', alpha=0.5)
        ax.legend(frameon=False, loc=0)
        ax = axes[2]
        df_LR = pd.read_csv(data_dir + 'LR_Results.csv')
        df_LR = df_LR[(df_LR['finger'] != 'two finger pads')]
        ax.plot(df_LR['mse'].values - df['mse_fitted'].values, label='mse_LR-fitted', alpha=0.8)
        ax.plot(df['r2_fitted'].values - df_LR['r2'].values, label='r2_fitted-LR', alpha=0.8)
        ax.legend(frameon=False, loc=0)
        fig.savefig(plot_dir + 'figure_'+ self.method +'/compare_mse_r2.jpg')
        print('mean r2 diff:', np.mean(df['r2_fitted'].values - df_LR['r2'].values))
        print('mean mse diff:', np.mean(df_LR['mse'].values - df['mse_fitted'].values))

        # plot exp filter parameters
        fig, axes = plt.subplots(2, 10, sharex='all', sharey='row')
        fig.set_size_inches(18, 6)
        y_list = ['exp_conv_T', 'exp_conv_k']
        combined_list = df['combined'].unique()
        for i in range(2):
            for j in range(10):
                temp = combined_list[j].split('_')
                df_unit = df[(df['unit_type'] == temp[0]) & (df['stimulus'] == temp[1])]
                ax = axes[i, j]
                order = [1, 3, 9, 18, 24]
                # sns.stripplot(df_unit, x='vel', y=y_list[i], hue='vel', order=order, palette='crest', alpha=.3, ax=ax)
                sns.pointplot(df_unit, x='vel', y=y_list[i], order = order, estimator='mean',
                              palette='crest', markers="d", scale=1, ax=ax)
                ax.set_title('_'.join(temp))
                ax.set_xlabel('')
                # if temp[1] == 'stroke' and i == 0: ax.set_ylim(-0.2, 2)
                # if temp[1] == 'tap' and i == 0: ax.set_ylim(-0.01, 0.2)
                ax.xaxis.set_tick_params(rotation=0)
                ax.legend().remove()
        sns.despine(trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, wspace=0.3)
        fig.savefig(plot_dir + 'figure_' + self.method + '/compare_LR_param.jpg')

        plt.show()

    def plot_LR_ARIMA_results(self):
        df = self.load_data()
        df.sort_values(['stimulus', 'unit_type', 'vel'], inplace=True)

        df['combined'] = df['unit_type'] + '_' + df['stimulus']# + '_' + df['vel'].astype('str')

        df_LR = pd.read_csv(data_dir + 'LR_Results.csv')
        df_LR = df_LR[(df_LR['finger'] != 'two finger pads')]

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(7, 3)
        ax.plot(df_LR['mse'].values - df['mse'].values, label='mse_LR-fitted', alpha=0.8)
        ax.plot(df['r2'].values - df_LR['r2'].values, label='r2_fitted-LR', alpha=0.8)
        ax.legend(frameon=False, loc=0)
        fig.savefig(plot_dir + 'figure_' + self.method + '/compare_mse_r2.jpg')
        print('mean r2 diff:', np.mean(df['r2'].values - df_LR['r2'].values))
        print('mean mse diff:', np.mean(df_LR['mse'].values - df['mse'].values))

        # plot exp filter parameters
        fig, axes = plt.subplots(3, 10, sharex='all', sharey='row')
        fig.set_size_inches(18, 9)
        y_list = ['arima_p', 'arima_d', 'arima_q']
        combined_list = df['combined'].unique()
        for i in range(3):
            for j in range(10):
                temp = combined_list[j].split('_')
                df_unit = df[(df['unit_type'] == temp[0]) & (df['stimulus'] == temp[1])]
                ax = axes[i, j]
                order = [1, 3, 9, 18, 24]
                # sns.stripplot(df_unit, x='vel', y=y_list[i], hue='vel', order=order, palette='crest', alpha=.3, ax=ax)
                sns.pointplot(df_unit, x='vel', y=y_list[i], order = order, estimator='mean',
                              palette='crest', markers="d", scale=1, ax=ax)
                ax.set_title('_'.join(temp))
                ax.set_xlabel('')
                # if temp[1] == 'stroke' and i == 0: ax.set_ylim(-0.2, 2)
                # if temp[1] == 'tap' and i == 0: ax.set_ylim(-0.01, 0.2)
                ax.xaxis.set_tick_params(rotation=0)
                ax.legend().remove()
        sns.despine(trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, wspace=0.3)
        fig.savefig(plot_dir + 'figure_' + self.method + '/compare_LR_param.jpg')

        plt.show()

    def plot_LR_LNP_results(self):
        df = self.load_data()
        sns.set(style="ticks",font_scale=1.7)

        df.sort_values(['unit_type', 'stimulus', 'vel'], inplace=True)
        df['combined'] = df['unit_type'] + '_' + df['stimulus']# + '_' + df['vel'].astype('str')

        df_LR = pd.read_csv(data_dir + 'LR_Results.csv')
        df_LR = df_LR[(df_LR['finger'] != 'two finger pads')]

        with open(data_dir + self.method + '_LinearFilter.data', 'rb') as f:
            linear_filters = pickle.load(f)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(7, 3)
        ax.plot(df_LR['mse'].values - df['mse'].values, label='mse_LR-fitted', alpha=0.8)
        ax.plot(df['r2'].values - df_LR['r2'].values, label='r2_fitted-LR', alpha=0.8)
        ax.legend(frameon=False, loc=0)
        fig.savefig(plot_dir + 'figure_' + self.method + '/compare_mse_r2.jpg')
        print('mean r2 diff:', np.mean(df['r2'].values - df_LR['r2'].values))
        print('mean mse diff:', np.mean(df_LR['mse'].values - df['mse'].values))

        # plot linear filter
        for u_i in range(len(unit_list)):
            df_unit = df[df['unit_type'] == unit_list[u_i]]
            fig, axes = plt.subplots(2, 5)
            fig.set_size_inches(18, 9)
            for i in range(2):
                stims = ['stroke', 'tap']
                stim = stims[i]
                for j in range(5):
                    ax = axes[i, j]
                    vels = [1, 3, 9, 18, 24]
                    vel = vels[j]
                    df_plot = df_unit[(df_unit['stimulus'] == stim) & (df_unit['vel'] == vel)]
                    plot_idx = list(df_plot.index)
                    print(plot_idx)
                    plot_lf = np.array([linear_filters[x] for x in plot_idx])
                    if len(plot_lf) == 0: continue
                    d = len(plot_lf[0])
                    t = np.arange(-d + 1, 1)
                    ax.plot(t, plot_lf.T)
                    ax.plot(t, np.nanmedian(plot_lf, axis=0), linestyle='--', linewidth=5, color='black')
                    ax.text(-1, 1, 'n = ' + str(len(plot_idx)))
                    ax.set_title('_'.join([unit_list[u_i], stim, str(vel)]))
                    ax.set_xlabel('')
                    # if temp[1] == 'stroke' and i == 0: ax.set_ylim(-0.2, 2)
                    # if temp[1] == 'tap' and i == 0: ax.set_ylim(-0.01, 0.2)
                    ax.xaxis.set_tick_params(rotation=0)
                    # ax.legend().remove()
            sns.despine(trim=True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, wspace=0.3)
            fig.savefig(plot_dir + 'figure_' + self.method + '/linear_filters_' + unit_list[u_i] +'.jpg')
        plt.show()

    def plot_LR_LNP_results_per_type(self):
        # df_results = pd.DataFrame()
        #
        # os.chdir('data/psth_regression/results_per_unit')
        # for name in os.listdir():  # iterate through all file
        #     if Path(name).is_file() and len(name.split('_'))>3 and name.split('_')[3] == 'All':
        #         if name.split('.')[-1] == 'csv':
        #             print(name)
        #             df_i = pd.read_csv(name)
        #             df_i['bin_size'] = float(name.split('_')[2])
        #             df_results = pd.concat([df_results, df_i], ignore_index=True)
        # os.chdir(Path(__file__).parents[0])
        # print(os.getcwd())

        df_results = pd.read_csv('data/psth_regression/results_per_unit/LR_LNP_All_Results.csv')

        for i in range(2):
            ys = ['r2', 'mse']
            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(7, 7)
            sns.lineplot(data=df_results, x='bin_size', y=ys[i], hue='unit_type', style='his_len', ax=ax)
            # ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2])
            handles, label_default = ax.get_legend_handles_labels()
            ax.legend(handles, label_default, frameon=False, loc=0, fontsize=14, handletextpad=0.3,
                      markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            sns.despine(trim=True)
            plt.tight_layout()
            fig.savefig(plot_dir + 'figure_' + self.method + '/results_per_unit/all_'+ys[i]+'_bin_size.jpg')

        # fig, axes = plt.subplots(2, 3, sharey='all', sharex='all')
        # fig.set_size_inches(18, 9)
        # for i in range(len(unit_list)):
        #     ax = axes[i//3, i%3]
        #     df_unit = df_results[df_results['unit_type'] == unit_list[i]]
        #     df_unit = pd.melt(df_unit, id_vars=['unit_type', 'length', 'r2', 'mse', 'bin_size'],
        #                       value_vars=['coef_' + n for n in contact_feat],
        #                       var_name='coef', value_name='value')
        #     sns.pointplot(df_unit, x='coef', y='value', hue='bin_size', ax=ax)
        #     ax.set_ylabel(unit_list[i])
        #     ax.xaxis.set_tick_params(rotation=45)
        #     handles, label_default = ax.get_legend_handles_labels()
        #     ax.legend(handles, label_default, frameon=False, loc=0, fontsize=14, handletextpad=0.3,
        #               markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
        #     sns.despine(trim=True)
        #     plt.tight_layout()
        # fig.savefig(plot_dir + 'figure_' + self.method + '/results_per_unit/all_coef_bin_size.jpg')


        # # plot linear filter
        # fig, axes = plt.subplots(3, 5, sharex='all', sharey='all')
        # fig.set_size_inches(18, 9)
        # for i in range(3):
        #     bins = [0.1, 0.05, 0.01]
        #     bin = bins[i]
        #     with open(data_dir + self.method + '_' + str(bin) + '_All_LinearFilter.data', 'rb') as f:
        #         lf = pickle.load(f)
        #
        #     for j in range(5):
        #         ax = axes[i, j]
        #         unit_type = unit_list[j]
        #         d = len(lf[0])
        #         t = np.array(np.arange(-d + 1, 1)) * bin
        #         ax.plot(t, lf[j, 1:]) # first element is for intercept
        #         ax.set_title(unit_type)
        #         ax.set_xlabel('t (s)')
        #
        #     sns.despine()
        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0.05, wspace=0.3)
        #     fig.savefig(plot_dir + 'figure_' + self.method + '/linear_filters.jpg')
        # plt.show()

    def plot_LR_LNP_results_per_trial(self):
        df_results = pd.read_csv('data/psth_regression/results_per_trial/LR_LNP_Results.csv')
        print(len(df_results.index))
        df_results = df_results[df_results['r2'] >= -1000]
        print(len(df_results.index))

        for i in range(2):
            ys = ['r2', 'mse']
            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(7, 7)
            sns.lineplot(data=df_results, x='bin_size', y=ys[i], hue='unit_type', style='his_len', ax=ax)
            # ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2])
            handles, label_default = ax.get_legend_handles_labels()
            ax.legend(handles, label_default, frameon=False, loc=0, fontsize=14, handletextpad=0.3,
                      markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            sns.despine(trim=True)
            plt.tight_layout()
            fig.savefig(plot_dir + 'figure_' + self.method + '/results_per_trial/all_'+ys[i]+'_bin_size.jpg')

