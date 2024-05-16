import matplotlib
import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from utils import *
# from plot import *
from psth_regression import lnp_model
from psth_regression import psth_regression
from main import run_method


def run_per_unit_type(dir_data, method, bin_size=0.1):
    df_data = pd.read_csv(dir_data)

    norm_method = 'min_max' if method in ['LN_poisson', 'LR_LNP'] else 'z_score'
    results_list, outputs = [], []
    unit_type = 'SA-II'

    df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_data, bin_size=bin_size, norm_method=norm_method)

    model = psth_regression(df_norm, y, y_scaler, unit_type, bin_size)
    model.set_method(method)
    results, output = run_method(model, method, counts=[])

    print([unit_type] + results)
    results_list.append([unit_type] + results)
    outputs.append(output)

    columns = ['unit_type', 'length', 'r2', 'mse']
    for c_i in contact_feat:
        columns.append('coef_' + c_i)
    columns.append('intercept')
    df_results = pd.DataFrame(data=results_list, columns=columns)
    print(df_results)
    df_results.to_csv(data_dir + method + '_' + str(bin_size) + '_emotion_All_Results.csv', index=False, header=True)

    if method == 'LR_LNP':
        with open(data_dir + method + '_' + str(bin_size) + '_emotion_All_LinearFilter.data', 'wb') as f:
            pickle.dump(outputs, f)

def pred_per_unit_type(dir_data, method, bin_size=0.1):
    # matplotlib.use("Agg")

    df_data = pd.read_csv(dir_data)
    df_param = pd.read_csv(data_dir + method + '_' + str(bin_size) + '_All_Results.csv')

    norm_method = 'min_max' if method in ['LN_poisson', 'LR_LNP'] else 'z_score'
    results_list, outputs = [], []
    unit_type = 'SA-II'

    df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_data, bin_size=bin_size, norm_method=norm_method)
    X_norm, y_norm = df_norm[contact_feat].values, df_norm['spike_binned']

    # use pretraiend PCA instead
    pca_reloaded = pickle.load(open(data_dir + 'pca_' + unit_type + '.pkl', 'rb'))
    X_norm = pca_reloaded.transform(X_norm)

    ### linear regression from semi-controled contact model
    coefs = df_param.loc[df_param['unit_type'] == unit_type, ['coef_' + i for i in ['PC1', 'PC2']]].values #contact_feat
    intercept = df_param.loc[df_param['unit_type'] == unit_type, ['intercept']].values
    print(coefs, intercept)
    LR = LinearRegression()
    LR.intercept_ = intercept
    LR.coef_ = coefs
    y_LR_norm = LR.predict(X_norm)
    y_LR_norm = MinMaxScaler().fit_transform(y_LR_norm.reshape(-1, 1))[:, 0]


    lnp = lnp_model()

    his_len = int(0.2 / bin_size) + 1
    with open(data_dir + method + '_' + str(bin_size) + '_All_LinearFilter.data', 'rb') as f:
        lf = pickle.load(f)
    y_pred_norm = lnp.predict(y_LR_norm, his_len, linear_filter=lf[unit_list.index(unit_type)])

    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.set_size_inches(8, 3)
    y_norm = np.array(y_norm)
    ax.plot(df_norm['t'].values, y_norm, label='true', alpha=0.5)
    ax.plot(df_norm['t'].values, y_LR_norm, label='LR', alpha=0.5)
    ax.plot(df_norm['t'].values, y_pred_norm, label='LR_LNP', alpha=0.5)
    ax.set_title(unit_type)
    plt.legend(frameon=False, loc=0, fontsize=13)
    sns.despine(trim=True)
    dir_temp = plot_dir + 'figure_' + method + '/predict/'
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    fig.savefig(dir_temp + method + '_' + unit_type + '.png')
    plt.show()

    columns = ['unit_type', 'length', 'r2', 'mse']
    r2 = r2_score(y_norm, y_pred_norm)
    mse = mean_squared_error(y_norm, y_pred_norm)
    length = len(df_norm.index)
    results_list = [[unit_type, length, r2, mse]]
    df_results = pd.DataFrame(data=results_list, columns=columns)
    print(df_results)
    df_results.to_csv(data_dir + method + '_' + str(bin_size) + '_pred_All_Results.csv', index=False, header=True)



if __name__ == '__main__':
    sns.set(style="ticks", font='Arial', font_scale=1.8)
    bin_size = 1

    dir_data = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/old/new/2021-12-08-ST10-unit1-1.csv'
    method = 'LR_LNP'
    pred_per_unit_type(dir_data, method, bin_size=bin_size)
    # run_per_unit_type(dir_data, method, bin_size=bin_size)