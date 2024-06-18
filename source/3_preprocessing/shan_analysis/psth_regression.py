import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import pickle

import scipy
from scipy.ndimage import convolve1d
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from pingouin import partial_corr

from params import *

# suppressing warining (zero in exp filter optimization)
import warnings
warnings.filterwarnings("ignore")


class psth_regression:

    def __init__(self, df, y, y_scaler, data_info, bin_size, run_mode='per_trial', his_len=0.2):
        self.df = df
        self.y = y
        self.y_scaler = y_scaler
        self.data_info = data_info
        self.bin_size = bin_size
        self.run_mode = run_mode
        self.his_len = his_len

    def set_method(self, method):
        self.method = method

    def func_regression(self):
        X_norm, y_norm = self.df[contact_feat].values, self.df['spike_binned'].values.reshape(-1, 1)

        if self.method == 'LR':
            model = LinearRegression()
            model.fit(X_norm, y_norm)
            importances = list(model.coef_[0])
            print(importances)
            importances = list(permutation_importance(model, X_norm, y_norm).importances_mean)
            print(importances)
        if self.method == 'RF':
            model = RandomForestRegressor()
            model.fit(X_norm, y_norm)
            importances = list(permutation_importance(model, X_norm, y_norm).importances_mean)

        y_pred_norm = model.predict(X_norm)
        y_pred = self.y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))

        fig, ax = plt.subplots(1,1, sharex=True)
        fig.set_size_inches(8, 3, forward=True)
        ax.plot(self.df['t'].values, self.y, alpha=0.5)
        ax.plot(self.df['t'].values, y_pred, alpha=0.5)
        ax.set_title(self.data_info)
        fig.savefig(plot_dir + 'figure_'+ self.method + '/regression/' + self.method + '_' + self.data_info + '.png')

        r2 = r2_score(y_norm, y_pred_norm)
        mse = mean_squared_error(y_norm, y_pred_norm)
        length = len(self.df.index)
        print('length: ', length)
        print(importances)
        print('R2: ', r2)
        print('MSE: ', mse, '\n')
        return [length, r2, mse] + importances, y_pred

    def partial_corr(self):
        # derive variance inflation factor for all contact features
        df_contact = self.df[contact_feat]
        cc = scipy.corrcoef(df_contact.values, rowvar=False)
        VIF = np.linalg.pinv(cc)
        vifs = list(VIF.diagonal())

        # linear regression with L1 regularization, CV for best alpha
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(self.df[contact_feat], self.df['spike_binned'])
        coefs = list(lasso.coef_)

        # correlation
        corrs = []
        for feat in contact_feat:
            pr = scipy.stats.pearsonr(self.df['spike_binned'], self.df[feat])
            if pr[1] > 0.05:
                corrs.append(None)
            else:
                corrs.append(pr[0])

        # partial correlation
        p_corrs = []
        for feat in contact_feat:
            if len(self.df.index) < 3:
                p_corrs.append(np.nan)
                continue
            p_corr = partial_corr(data=self.df, x=feat, y='spike_binned', covar=[i for i in contact_feat if i != feat], method='pearson')
            # print(feat, p_corr['r'].values[0])
            if p_corr['p-val'].values[0] > 0.05:
                p_corrs.append(np.nan)
            else:
                # print('----------yes---------')
                p_corrs.append(p_corr['r'].values[0])

        return [len(self.df.index)] + vifs + coefs + p_corrs + corrs

    def func_LR_ARIMA(self, counts):
        X_norm, y_norm = self.df[contact_feat].values, self.df['spike_binned'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_norm, y_norm)
        y_pred_norm = model.predict(X_norm)
        y_LR = self.y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))

        y_residual = y_norm - y_pred_norm
        counts[0] = counts[0] + 1
        his_duration = contact_duration[self.data_info.split('_')[5]]
        max_his = int(his_duration / self.bin_size)

        if len(y_residual) > 10:
            stationary_test = adfuller(y_residual)
            p_val = stationary_test[1]
            print(p_val)
            if p_val > 0.05:
                print('-------------- not stationary --------------')
                counts[2] = counts[2] + 1
                arima = auto_arima(y_residual, start_p=1, start_q=1, max_p=max_his, max_q=max_his, stationary=False)
            else:
                arima = auto_arima(y_residual, start_p=1, start_q=1, max_p=max_his, max_q=max_his, stationary=True)
            print(list(arima.order))
            arima_order = arima.order
            y_res_pred = arima.predict_in_sample()
            print(len(y_residual), len(y_res_pred))
            # print(y_pred_norm, y_res_pred)
            y_pred_combined = y_pred_norm[:,0] + y_res_pred
            y_pred = self.y_scaler.inverse_transform(y_pred_combined.reshape(-1, 1))

            fig, ax = plt.subplots(1,1, sharex=True)
            fig.set_size_inches(8, 3, forward=True)
            ax.plot(self.df['t'].values, self.y, label='true', alpha=0.5)
            ax.plot(self.df['t'].values, y_LR, label='LR', alpha=0.5)
            ax.plot(self.df['t'].values, y_pred, label='LR_ARIMA', alpha=0.5)
            ax.set_title(self.data_info)
            plt.legend(frameon=False, loc=0, fontsize=13)
            sns.despine(trim=True)
            dir_temp = plot_dir + 'figure_'+ self.method + '/regression/'
            if not os.path.exists(dir_temp):
                os.makedirs(dir_temp)
            fig.savefig(dir_temp + self.method + '_' + self.data_info + '.png')

            r2 = r2_score(y_norm, y_pred_norm)
            mse = mean_squared_error(y_norm, y_pred_norm)
            length = len(self.df.index)

        else:
            r2 = np.nan
            mse = np.nan
            length = len(self.df.index)
            arima_order = (np.nan, np.nan, np.nan)
            y_pred = np.nan
            counts[1] = counts[1] + 1

        return [length, r2, mse] + list(arima_order), y_pred

    def func_cca(self):
        X_norm, y_norm = self.df[contact_feat].values, self.df['spike_binned'].values.reshape(-1, 1)

        model = CCA(n_components=1)
        X_c, _ = model.fit_transform(X_norm, y_norm)
        # print(model.x_weights_)
        # print(model.y_weights_)
        # print(model.y_loadings_)
        x_loadings = model.x_loadings_
        score = model.score(X_norm, y_norm)
        r2 = r2_score(y_norm, X_c)
        length = len(self.df.index)
        loading_max = contact_feat[np.argmax(x_loadings)]
        print(len(X_norm), len(X_c))
        if np.mean(x_loadings) < 0: X_c = - X_c
        return [length, score, r2, loading_max] + list(x_loadings[:,0]), X_c

    def func_LN_model(self, LN):
        cca_results, X_c = self.func_cca()
        X_c = X_c[:,0]
        y_norm = self.df['spike_binned'].values

        def linear_conv_filter(params, input_data, output_target, bin_size, his_duration):
            a, b = params[0], params[1]
            weights = b * np.exp(- 1 / a * np.arange(int(his_duration / bin_size + 1)) * bin_size)
            output_data = convolve1d(input_data, weights=weights,
                                     mode='constant', cval=0, origin=-(len(weights)//2))
            mse = np.mean((output_data - output_target)**2)
            return mse

        def neg_log_lik_lnp(params, input_data, output_target, bin_size, his_duration):
            """
            negative log likelihood of the linear-nonlinear Poisson model
            linear: linear_conv_filter
            noninear: inverse link function for Poisson distribution
            """
            a, b = params[0], params[1]
            weights = b * np.exp(- 1 / a * np.arange(int(his_duration / bin_size + 1)) * bin_size)
            input_data = input_data + 10 ** -6
            linear_data = convolve1d(input_data, weights=weights,
                                     mode='constant', cval=0, origin=-(len(weights)//2))
            log_lik_nonlinear = output_target @ np.log(linear_data) - linear_data.sum()
            return - log_lik_nonlinear

        def nonlinear_ReLU(params, input_data, output_target, bin_size, his_duration):
            T, K, a, b = params[0], params[1], params[2], params[3]
            weights = K * np.exp(- 1 / T * np.arange(int(his_duration / bin_size + 1)) * bin_size)
            linear_data = convolve1d(input_data, weights=weights,
                                     mode='constant', cval=0, origin=-(len(weights)//2))
            # ReLU shape nonlinear function
            nonlinear_data = np.maximum(a, b * linear_data)
            mse = np.mean((nonlinear_data - output_target)**2)
            return mse

        his_duration = contact_duration[self.data_info.split('_')[5]]

        if LN in ['LF', 'LN_poly']:
            bounds = [(0, his_duration), (-10, 10)]
            result_diff_evo = differential_evolution(linear_conv_filter, bounds,
                                                     args=(X_c, y_norm, self.bin_size, his_duration))

        if LN == 'LN_poisson':
            bounds = [(0, his_duration), (0, 10)]
            X_c = np.array([0 if i < 0 else i for i in X_c])
            X_c = MinMaxScaler().fit_transform(X_c.reshape(-1,1))[:,0]
            result_diff_evo = differential_evolution(neg_log_lik_lnp, bounds,
                                                     args=(X_c, y_norm, self.bin_size, his_duration))

        if LN == 'LN_relu':
            bounds = [(0, his_duration), (0, 10), (-1, 1), (-5, 5)]
            result_diff_evo = differential_evolution(nonlinear_ReLU, bounds,
                                                     args=(X_c, y_norm, self.bin_size, his_duration))

        x = result_diff_evo.x
        optm_min = result_diff_evo.fun

        weights = x[1] * np.exp(- 1 / x[0] * np.arange(int(his_duration / self.bin_size + 1)) * self.bin_size)
        y_fitted = convolve1d(X_c, weights=weights, mode='constant',
                                  cval=0, origin=-(len(weights)//2))
        if LN == 'LN_relu':
            y_fitted = np.maximum(x[2], x[3] * y_fitted)

        if LN == 'LN_poly':
            poly = np.poly1d(np.polyfit(y_fitted, y_norm, 3))
            y_fitted = poly(y_fitted)

        mse = np.mean((y_fitted - y_norm)**2)
        r2_fitted = r2_score(y_norm, y_fitted)
        mse_contact_cca = np.mean((X_c - y_norm)**2)
        mse_contact_single = np.mean((self.df[cca_results[3]].values - y_norm)**2)
        r2_contact_single = r2_score(y_norm, self.df[cca_results[3]].values)

        fig, ax = plt.subplots(1,1, sharex=True)
        fig.set_size_inches(8, 3, forward=True)
        ax.plot(self.df['t'].values, y_norm, label='true', alpha=0.5)
        ax.plot(self.df['t'].values, y_fitted, label='fitted', alpha=0.5)
        ax.plot(self.df['t'].values, X_c, label='contact_cca', alpha=0.5)
        ax.plot(self.df['t'].values, self.df[cca_results[3]].values, label='contact_single',
                color='grey', linewidth = 3, alpha=0.3)
        ax.set_title(self.data_info)
        plt.legend(frameon=False, loc=0, fontsize=13)
        sns.despine(trim=True)
        fig.savefig(plot_dir + 'figure_' + LN + '/regression_' + LN + '/fitted_' + self.data_info + '.png')

        results = cca_results + list(x) + \
                  [optm_min, mse, mse_contact_cca, mse_contact_single, r2_fitted, r2_contact_single]

        return results, y_fitted

    def func_ARIMA(self, LN, counts):
        y_fitted, _ = self.func_LN_model(LN=LN)
        y_norm = self.df['spike_binned'].values

        y_residual = y_norm - y_fitted
        counts[0] = counts[0] + 1

        if len(y_residual) > 10:
            stationary_test = adfuller(y_residual)
            p_val = stationary_test[1]
            print(p_val)
            if p_val > 0.05:
                print('-------------- not stationary --------------')
                counts[2] = counts[2] + 1
        else:
            counts[1] = counts[1] + 1

    def func_LR_LNP(self):
        X_norm, y_norm = self.df[contact_feat].values, self.df['spike_binned']

        # if self.run_mode == 'per_unit':
        #     pca = PCA(n_components=2)
        #     X_norm = pca.fit_transform(X_norm)
        #     explained = pca.explained_variance_ratio_
        #     print(explained)
        #     pickle.dump(pca, open(data_dir + 'PCA/pca_' + self.data_info + '.pkl', "wb"))

        model = LinearRegression()
        model.fit(X_norm, y_norm)
        coefs = list(model.coef_)
        intercept = model.intercept_
        y_LR_norm = model.predict(X_norm)
        y_LR = self.y_scaler.inverse_transform(y_LR_norm.reshape(-1, 1))

        if self.data_info in unit_list:
            his_duration = self.his_len
            his_len = int(his_duration / self.bin_size) + 1
        else:
            his_len = int(contact_duration[self.data_info.split('_')[5]] / self.bin_size) + 1
            his_len = min(his_len, 20)

        his_len = int(self.his_len / self.bin_size) + 1

        # y_LR_norm = MinMaxScaler().fit_transform(y_LR_norm.reshape(-1, 1))[:, 0]
        self.lnp_model = lnp_model()
        print(his_len)
        linear_filter, y_pred_norm = self.lnp_model.run_model(y_LR_norm, y_norm, his_len)

        if self.run_mode == 'per_unit':
            fig, ax = plt.subplots(1, 1, sharex='all')
            fig.set_size_inches(8, 3)
            y_norm = np.array(y_norm)
            ax.plot(self.df['t'].values, y_norm, label='true', alpha=0.5)
            ax.plot(self.df['t'].values, y_LR_norm, label='LR', alpha=0.5)
            ax.plot(self.df['t'].values, y_pred_norm, label='LR_LNP', alpha=0.5)
            ax.set_title(self.data_info)
            plt.legend(frameon=False, loc=0, fontsize=13)
            sns.despine(trim=True)
            if self.run_mode == 'per_unit':
                dir_temp = plot_dir + 'figure_' + self.method + '/results_per_unit/' + \
                           str(int(1000*self.bin_size)) + 'ms_' + \
                           str(int(1000*self.his_len)) + 'ms/regression/'
            else:
                dir_temp = plot_dir + 'figure_' + self.method + '/regression/'
            if not os.path.exists(dir_temp):
                os.makedirs(dir_temp)
            fig.savefig(dir_temp + self.method + '_' + self.data_info + '.png')

            fig, ax = plt.subplots(1, 1, sharex='all')
            fig.set_size_inches(8, 3)
            t = np.arange(-len(linear_filter) + 1, 1) * self.bin_size
            ax.plot(t, linear_filter, label='true', alpha=0.5)
            ax.set_title(self.data_info)
            plt.legend(frameon=False, loc=0, fontsize=13)
            sns.despine(trim=True)
            if self.run_mode == 'per_unit':
                dir_temp = plot_dir + 'figure_' + self.method + '/results_per_unit/' + \
                           str(int(1000*self.bin_size)) + 'ms_' + \
                           str(int(1000*self.his_len)) + 'ms/linear_filter/'
            else:
                dir_temp = plot_dir + 'figure_' + self.method + '/linear_filter/'
            if not os.path.exists(dir_temp):
                os.makedirs(dir_temp)
            fig.savefig(dir_temp + self.method + '_' + self.data_info + '.png')

        r2 = r2_score(y_norm, y_pred_norm)
        mse = mean_squared_error(y_norm, y_pred_norm)
        length = len(self.df.index)
        # if abs(mse) > 5: r2, mse = np.nan, np.nan
        # linear_filter = [i if -30<i<30 else np.nan for i in linear_filter]

        return [length, r2, mse] + coefs + [intercept], linear_filter, y_pred_norm




class lnp_model:
    def __init__(self):
        return

    def __call__(self):
        return

    def make_stim_matrix(self, stim, d):
        padded_stim = np.concatenate([np.zeros(d - 1), stim])
        T = len(stim)
        X = np.zeros((T, d))
        for t in range(T):
            X[t] = padded_stim[t:t + d]
        return X

    def neg_log_lik_lnp(self, theta, X, y):
        # Compute the negative Poisson log likelihood
        rate = np.exp(X @ theta)
        log_lik = y @ np.log(rate) - rate.sum()
        return -log_lik

    def fit_lnp(self, X, y, d):
        # Use a random vector of weights to start (mean 0, sd .2)
        x0 = np.random.normal(0, .2, d + 1)
        # Find parameters that minmize the negative log likelihood function
        res = minimize(self.neg_log_lik_lnp, x0, args=(X, y))
        return res["x"]

    def run_model(self, stim, spikes, his_len):

        # Build the design matrix
        y = spikes
        constant = np.ones_like(y)
        X = np.column_stack([constant, self.make_stim_matrix(stim, his_len)])

        theta_lnp = self.fit_lnp(X, y, his_len)
        linear_filter = theta_lnp

        y_pred = np.exp(X @ theta_lnp)
        y_pred[(y_pred == -np.inf) | (y_pred == np.inf)] = 0

        return linear_filter, y_pred

    def predict(self, stim, his_len, linear_filter):
        constant = np.ones_like(stim)
        X = np.column_stack([constant, self.make_stim_matrix(stim, his_len)])

        y_pred = np.exp(X @ linear_filter)
        y_pred[(y_pred == -np.inf) | (y_pred == np.inf)] = 0

        return y_pred



