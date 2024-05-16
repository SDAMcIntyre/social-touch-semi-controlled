import matplotlib
import os
import pickle

import numpy as np
import pandas as pd

from utils import *
from plot import *
from psth_regression import psth_regression


def define_column(method):
    if method in ['LR', 'RF']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force',
                   'length', 'r2', 'mse']
        for c_i in contact_feat:
            columns.append('imp_' + c_i)

    if method in ['P_CORR']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force', 'length']
        for c_i in contact_feat:
            columns.append('vif_' + c_i)
        for c_i in contact_feat:
            columns.append('coef_' + c_i)
        for c_i in contact_feat:
            columns.append('p_corr_' + c_i)
        for c_i in contact_feat:
            columns.append('corr_' + c_i)

    if method in ['CCA']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force',
                   'length', 'corr_r2', 'r2', 'loading_max']
        for c_i in contact_feat:
            columns.append('loading_' + c_i)

    if method in ['LF', 'LN_poisson', 'LN_relu', 'LN_poly']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force',
                   'length', 'corr_r2', 'r2_contact_cca', 'loading_max']
        for c_i in contact_feat:
            columns.append('loading_' + c_i)
        if method == 'LN_relu':
            columns = columns + ['exp_conv_T', 'exp_conv_k', 'relu_x', 'relu_k', 'optm_min',
                                 'mse_fitted', 'mse_contact_cca', 'mse_contact_single',
                                 'r2_fitted', 'r2_contact_single']
        else:
            columns = columns + ['exp_conv_T', 'exp_conv_k', 'optm_min',
                                 'mse_fitted', 'mse_contact_cca', 'mse_contact_single',
                                 'r2_fitted', 'r2_contact_single']
    if method in ['LR_LNP']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force',
                   'length', 'r2', 'mse']
        for c_i in contact_feat:
            columns.append('coef_' + c_i)

    if method in ['ARIMA', 'LR_ARIMA']:
        columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
                   'stimulus', 'vel', 'finger', 'force',
                   'length', 'r2', 'mse', 'arima_p', 'arima_d', 'arima_q']

    return columns

def run_method(model, method, counts=None):
    if method in ['LR', 'RF']:
        results, output = model.func_regression()

    if method in ['P_CORR']:
        results = model.partial_corr()
        output = []

    if method in ['LR_ARIMA']:
        results, output = model.func_LR_ARIMA(counts)

    if method in ['CCA']:
        results, output = model.func_cca()

    if method in ['LF', 'LN_poisson', 'LN_relu', 'LN_poly']:
        results, output = model.func_LN_model(LN=method)

    if method in ['LR_LNP']:
        results, linear_filter, _ = model.func_LR_LNP()
        return results, linear_filter

    if method in ['ARIMA']:
        model.func_ARIMA('LN_poly', counts)
        return

    return results, output

def run_all_units(method, bin_size=0.1):
    matplotlib.use("Agg")

    dir_data = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/new_axes/'
    dict_unit = {'SA-II': ['ST13-unit1', 'ST14-unit3', 'ST16-unit2', 'ST18-unit4'],
                 'CT': ['ST13-unit2', 'ST16-unit3'],
                 'HFA': ['ST13-unit3', 'ST14-unit2'],
                 'SA-I': ['ST14-unit1', 'ST16-unit5', 'ST18-unit2'],
                 'Field': ['ST14-unit4', 'ST15-unit1', 'ST15-unit2', 'ST16-unit4', 'ST18-unit1']}

    norm_method = 'min_max' if method in ['LN_poisson', 'LR_LNP'] else 'z_score'

    counts = [0, 0, 0]  # all, short, non_stationary
    outputs = []
    spike_psths = []
    labels = []

    results_list = []
    for name in os.listdir(os.fsencode(dir_data)):
        name = os.fsdecode(name)
        unit_name = '-'.join(name.split('-')[3:5])
        print(unit_name)
        for key, value in dict_unit.items():
            if unit_name in value: unit_type = key; print(key)
        df = pd.read_csv(dir_data + name)
        df.dropna(subset=['block_id', 'trial_id'], inplace=True)

        for block_i in df['block_id'].unique():
            df_b = df[df['block_id'] == block_i]
            for trial_i in df_b['trial_id'].unique():
                df_b_t = df_b[df_b['trial_id'] == trial_i]
                data_info_list = [unit_type, unit_name, int(block_i), int(trial_i),
                             df_b_t['stimulus'].values[0], int(df_b_t['vel'].values[0]),
                             df_b_t['finger'].values[0][1:], df_b_t['force'].values[0][1:]]
                data_info = '_'.join(str(i) for i in data_info_list)
                df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_b_t, bin_size=bin_size, norm_method=norm_method)
                spike_psths.append(list(df_norm['spike_binned'].values))
                labels.append(data_info_list)

                if len(df_norm.index) < 3: continue
                model = psth_regression(df_norm, y, y_scaler, data_info, bin_size)
                model.set_method(method)
                results, output = run_method(model, method, counts)

                print(data_info_list + results)
                results_list.append(data_info_list + results)
                outputs.append(output)

    columns = define_column(method)
    df_results = pd.DataFrame(data=results_list, columns=columns)
    df_labels = pd.DataFrame(data=labels, columns=columns[:len(data_info_list)])
    print(df_results)
    df_results.to_csv(data_dir + '/results_per_trial/' + method + '_Results.csv', index=False, header=True)
    if method == 'LR_LNP':
        df_labels.to_csv(data_dir + '/results_per_trial/' + 'labels.csv', index=False, header=True)
        with open(data_dir + '/results_per_trial/' + 'spike_psth.data', 'wb') as f:
            pickle.dump(spike_psths, f)
        with open(data_dir + '/results_per_trial/' + method + '_LinearFilter.data', 'wb') as f:
            pickle.dump(outputs, f)

    # plt.show()

def run_all_units_LR_LNP(method, bin_size=0.1):
    matplotlib.use("Agg")

    dir_data = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/new_axes/'
    dict_unit = {'SA-II': ['ST13-unit1', 'ST14-unit3', 'ST16-unit2', 'ST18-unit4'],
                 'CT': ['ST13-unit2', 'ST16-unit3'],
                 'HFA': ['ST13-unit3', 'ST14-unit2'],
                 'SA-I': ['ST14-unit1', 'ST16-unit5', 'ST18-unit2'],
                 'Field': ['ST14-unit4', 'ST15-unit1', 'ST15-unit2', 'ST16-unit4', 'ST18-unit1']}

    norm_method = 'min_max'

    # outputs = []
    # spike_psths = []
    labels = []

    results_list = []

    for name in os.listdir(os.fsencode(dir_data)):
        name = os.fsdecode(name)
        unit_name = '-'.join(name.split('-')[3:5])
        print(unit_name)
        for key, value in dict_unit.items():
            if unit_name in value: unit_type = key; print(key)
        df = pd.read_csv(dir_data + name)
        df.dropna(subset=['block_id', 'trial_id'], inplace=True)

        for block_i in df['block_id'].unique():
            df_b = df[df['block_id'] == block_i]
            for trial_i in df_b['trial_id'].unique():
                df_b_t = df_b[df_b['trial_id'] == trial_i]

                for bin_size in [0.01, 0.05] + list(np.arange(0.1, 1.6, 0.05)) + [2]:  #
                    for his_len in [0.1, 0.2, 0.3, 0.4, 0.5]:

                        data_info_list = [unit_type, unit_name, int(block_i), int(trial_i),
                                          df_b_t['stimulus'].values[0], int(df_b_t['vel'].values[0]),
                                          df_b_t['finger'].values[0][1:], df_b_t['force'].values[0][1:],
                                          bin_size, his_len]

                        data_info = '_'.join(str(i) for i in data_info_list)
                        df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_b_t, bin_size=bin_size, norm_method=norm_method)

                        if len(df_norm.index) < 3: continue
                        model = psth_regression(df_norm, y, y_scaler, data_info, bin_size, his_len)
                        model.set_method(method)
                        results, output = run_method(model, method)

                        print(data_info_list + results)
                        results_list.append(data_info_list + results)

    columns = ['unit_type', 'unit_name', 'block_id', 'trial_id',
               'stimulus', 'vel', 'finger', 'force', 'bin_size', 'his_len',
               'length', 'r2', 'mse']
    for c_i in contact_feat:
        columns.append('coef_' + c_i)
    columns.append('interncept')

    df_results = pd.DataFrame(data=results_list, columns=columns)
    print(df_results)
    df_results.to_csv(data_dir + '/results_per_trial/' + method + '_Results.csv', index=False, header=True)


    # plt.show()


def run_per_unit_type(method, bin_size=0.1):
    matplotlib.use("Agg")

    dir_data = 'D:/OneDrive - University of Virginia/Projects/TSSingleUnitModel/python/data/semi_control_IFF/data_all.pkl'
    df_data = pd.read_pickle(dir_data)

    norm_method = 'min_max' if method in ['LN_poisson', 'LR_LNP'] else 'z_score'
    df_data.replace({'type': {'SAI': 'SA-I'}}, inplace=True)
    df_data.replace({'type': {'SAII': 'SA-II'}}, inplace=True)
    df_results_all = pd.DataFrame()

    for bin_size in [0.01, 0.05] + list(np.arange(0.1, 1.6, 0.05)) + [2]:#
        for his_len in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results_list, outputs = [], []
            for unit_type in unit_list:
                df_unit = df_data[df_data['type'] == unit_type]

                df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_unit, bin_size=bin_size, norm_method=norm_method)

                model = psth_regression(df_norm, y, y_scaler, unit_type, bin_size, run_mode='per_unit', his_len=his_len)
                model.set_method(method)
                results, output = run_method(model, method)

                print([unit_type] + results)
                results_list.append([unit_type, bin_size, his_len] + results)
                outputs.append(output)

            columns = ['unit_type', 'bin_size', 'his_len', 'length', 'r2', 'mse']
            for c_i in contact_feat:#'PC1', 'PC2'
                columns.append('coef_' + c_i)
            columns.append('intercept')
            df_results = pd.DataFrame(data=results_list, columns=columns)
            print(df_results)
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)

            if method == 'LR_LNP':
                with open(data_dir + '/results_per_unit/LR_LNP_linear_filter/LF_' + str(bin_size) +
                          '_' + str(his_len) + '.data', 'wb') as f:
                    pickle.dump(outputs, f)
    df_results_all.to_csv(data_dir + '/results_per_unit/' + method + '_All_Results.csv', index=False, header=True)


if __name__ == '__main__':
    sns.set(style="ticks", font='Arial', font_scale=1.8)
    bin_size = 1

    run_mode = 'per_trial' #'per_unit' #
    method = 'LR_LNP' #'LR' # 'P_CORR' # 'LN_poisson' # 'LR_ARIMA' #'LF' # 'LN_poly' # ARIMA' # 'CCA' # 'RF' # 'LN_relu' #

    # if run_mode == 'per_trial':
    # #    run_all_units(method, bin_size)
    #     run_all_units_LR_LNP(method)
    # if run_mode == 'per_unit':
    #     run_per_unit_type(method, bin_size)

    plot_class = plot_results(run_mode, method)
    plt.show()

