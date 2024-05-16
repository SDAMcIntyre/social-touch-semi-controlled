import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import os
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import CCA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

# suppressing warining (zero in exp filter optimization)
import warnings
warnings.filterwarnings("ignore")

def find_consecutive_zeros(data):
    iszero = np.concatenate(([0], np.equal(data, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    diff = ranges[:, 1] - ranges[:, 0]
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

def nerual_psth_smoothed_contact(df, bin_size, norm_method = 'z_score'):

	# plot_raw_data(df)

	df_t = df['t'].values
	df_spike = df['spike'].values
	spike_times = df_t[df_spike == 1]

	n_bins = int(np.ceil(spike_times[-1] / bin_size))
	bins = np.arange(0, round((n_bins + 1) * bin_size, 2), bin_size)
	t = np.arange(bin_size / 2, n_bins * bin_size, bin_size)

	### neural psth
	spike_binned, _ = np.histogram(spike_times, bins=bins)	
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


def func_regression(df, y, y_scaler, data_info_name, method):
	X_norm, y_norm = df[contact_feat].values, df['spike_binned'].values.reshape(-1, 1)

	if method == 'LR':
		model = LinearRegression()	
		model.fit(X_norm, y_norm)
		importances = list(model.coef_[0])
		print(importances)
		importances = list(permutation_importance(model, X_norm, y_norm).importances_mean)
		print(importances)
	if method == 'RF':
		model = RandomForestRegressor()
		model.fit(X_norm, y_norm)
		importances = list(permutation_importance(model, X_norm, y_norm).importances_mean)

	y_pred_norm = model.predict(X_norm)
	y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))

	fig, ax = plt.subplots(1,1, sharex=True)    
	fig.set_size_inches(8, 3, forward=True)
	ax.plot(df['t'].values, y, alpha=0.5)
	ax.plot(df['t'].values, y_pred, alpha=0.5)
	ax.set_title(data_info_name)
	fig.savefig(plot_dir + 'figure_'+ method + '/regression/' + method + '_' + data_info_name + '.png')

	r2 = r2_score(y_norm, y_pred_norm)
	mse = mean_squared_error(y_norm, y_pred_norm)
	length = len(df.index)
	print('length: ', length)
	print(importances)
	print('R2: ', r2)
	print('MSE: ', mse, '\n')
	return [length, r2, mse] + importances

def func_LR_ARIMA(df, y, y_scaler, data_info_name, method, counts):
	X_norm, y_norm = df[contact_feat].values, df['spike_binned'].values.reshape(-1, 1)

	model = LinearRegression()	
	model.fit(X_norm, y_norm)
	y_pred_norm = model.predict(X_norm)
	y_LR = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))

	y_residual = y_norm - y_pred_norm
	counts[0] = counts[0] + 1

	if len(y_residual) > 10:
		stationary_test = adfuller(y_residual)	
		p_val = stationary_test[1]
		print(p_val)
		if p_val > 0.05:
			print('-------------- not stationary --------------')
			counts[2] = counts[2] + 1
			arima = auto_arima(y_residual, start_p=1, start_q=1, max_p=50, max_q=50, stationary=False)
		else:
			arima = auto_arima(y_residual, start_p=1, start_q=1, max_p=50, max_q=50, stationary=True)
		print(list(arima.order))
		y_res_pred = arima.predict_in_sample()
		print(len(y_residual), len(y_res_pred))
		# print(y_pred_norm, y_res_pred)
		y_pred_combined = y_pred_norm[:,0] + y_res_pred 
		y_pred = y_scaler.inverse_transform(y_pred_combined.reshape(-1, 1))

		fig, ax = plt.subplots(1,1, sharex=True)    
		fig.set_size_inches(8, 3, forward=True)
		ax.plot(df['t'].values, y, label='true', alpha=0.5)
		ax.plot(df['t'].values, y_LR, label='LR', alpha=0.5)
		ax.plot(df['t'].values, y_pred, label='LR_ARIMA', alpha=0.5)
		ax.set_title(data_info_name)
		plt.legend(frameon=False, loc=0, fontsize=13)
		sns.despine(trim=True)
		dir_temp = plot_dir + 'figure_'+ method + '/regression/'
		if not os.path.exists(dir_temp):
   			os.makedirs(dir_temp)
		fig.savefig(dir_temp + method + '_' + data_info_name + '.png')

		r2 = r2_score(y_norm, y_pred_norm)
		mse = mean_squared_error(y_norm, y_pred_norm)
		length = len(df.index)

	else:
		r2 = np.nan
		mse = np.nan
		length = len(df.index)
		counts[1] = counts[1] + 1

	return [length, r2, mse] + list(arima.order)

	
	

def func_cca(df):
	X_norm, y_norm = df[contact_feat].values, df['spike_binned'].values.reshape(-1, 1)

	model = CCA(n_components=1)	
	X_c, _ = model.fit_transform(X_norm, y_norm)
	# print(model.x_weights_)
	# print(model.y_weights_)
	# print(model.y_loadings_)
	x_loadings = model.x_loadings_
	score = model.score(X_norm, y_norm)
	r2 = r2_score(y_norm, X_c)
	length = len(df.index)
	loading_max = contact_feat[np.argmax(x_loadings)]
	print(len(X_norm), len(X_c))
	if np.mean(x_loadings) < 0: X_c = - X_c
	return [length, score, r2, loading_max] + list(x_loadings[:,0]), X_c

def func_LN_model(df, y, y_scaler, data_info_name, LN='LF'):
	cca_results, X_c = func_cca(df)
	X_c = X_c[:,0]
	y_norm = df['spike_binned'].values

	def linear_conv_filter(params, input_data, output_target):
		a, b = params[0], params[1]
		weights = b * np.exp(- 1 / a * np.arange(1001) * 0.001)
		output_data = convolve1d(input_data, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))		
		mse = np.mean((output_data - output_target)**2)
		return mse

	def neg_log_lik_lnp(params, input_data, output_target):
		""" 
		negative log likelihood of the linear-nonlinear Poisson model
		linear: linear_conv_filter
		noninear: inverse link function for Poisson distribution
		"""
		a, b = params[0], params[1]
		weights = b * np.exp(- 1 / a * np.arange(1001) * 0.001)
		input_data = input_data + 10 ** -6
		linear_data = convolve1d(input_data, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))
		log_lik_nonlinear = output_target @ np.log(linear_data) - linear_data.sum()		
		return - log_lik_nonlinear

	def nonlinear_ReLU(params, input_data, output_target):
		T, K, a, b = params[0], params[1], params[2], params[3]
		weights = K * np.exp(- 1 / T * np.arange(1001) * 0.001)
		linear_data = convolve1d(input_data, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))
		# ReLU shape nonlinear function
		nonlinear_data = np.maximum(a, b * linear_data)
		mse = np.mean((nonlinear_data - output_target)**2)
		return mse

	def nonlinear_polynomial(params, input_data, output_target):
		T, K, a, b = params[0], params[1], params[2], params[3]
		weights = K * np.exp(- 1 / T * np.arange(1001) * 0.001)
		linear_data = convolve1d(input_data, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))
		poly = np.poly1d(np.polyfit(linear_data, output_target, 3)) 
		nonlinear_data = poly(linear_data)
		mse = np.mean((nonlinear_data - output_target)**2)
		return mse

	# initial_params = [0.0001, 1]
	# result_minimize = minimize(linear_conv_filter, initial_params, args=(X_c, y_norm))
	# r_x_min = result_minimize.x
	# r_mse_min = result_minimize.fun

	T_bounds_list = {'1': 2, '3': 1, '9': 0.33, '18': 0.166, '24': 0.125}
	if LN in ['LF', 'LN_poly']:
		bounds = [(0, T_bounds_list[data_info_name.split('_')[5]]), (-10, 10)]
		result_diff_evo = differential_evolution(linear_conv_filter, bounds, args=(X_c, y_norm))
	if LN == 'LN_poisson':
		bounds = [(0, T_bounds_list[data_info_name.split('_')[5]]), (0, 10)]
		X_c = np.array([0 if i < 0 else i for i in X_c])
		X_c = MinMaxScaler().fit_transform(X_c.reshape(-1,1))[:,0]
		result_diff_evo = differential_evolution(neg_log_lik_lnp, bounds, args=(X_c, y_norm))
	if LN == 'LN_relu':
		bounds = [(0, T_bounds_list[data_info_name.split('_')[5]]), (0, 10), (-1, 1), (-5, 5)]
		result_diff_evo = differential_evolution(nonlinear_ReLU, bounds, args=(X_c, y_norm))
		
	x = result_diff_evo.x
	optm_min = result_diff_evo.fun

	weights = x[1] * np.exp(- 1 / x[0] * np.arange(1001) * 0.001)
	y_fitted = convolve1d(X_c, weights=weights, mode='constant', 
							  cval=0, origin=-(len(weights)//2))
	if LN == 'LN_relu':
		y_fitted = np.maximum(x[2], x[3] * y_fitted)
	mse = np.mean((y_fitted - y_norm)**2)
	r2_fitted = r2_score(y_norm, y_fitted)
	mse_contact_cca = np.mean((X_c - y_norm)**2)
	mse_contact_single = np.mean((df[cca_results[3]].values - y_norm)**2)
	r2_contact_single = r2_score(y_norm, df[cca_results[3]].values)

	fig, ax = plt.subplots(1,1, sharex=True)    
	fig.set_size_inches(8, 3, forward=True)
	ax.plot(df['t'].values, y_norm, label='true', alpha=0.5)
	ax.plot(df['t'].values, y_fitted, label='fitted', alpha=0.5)
	ax.plot(df['t'].values, X_c, label='contact_cca', alpha=0.5)
	ax.plot(df['t'].values, df[cca_results[3]].values, label='contact_single', 
			color='grey', linewidth = 3, alpha=0.3)
	ax.set_title(data_info_name)
	plt.legend(frameon=False, loc=0, fontsize=13)
	sns.despine(trim=True)
	fig.savefig(plot_dir + 'figure_' + LN + '/regression_' + LN + '/fitted_' + data_info_name + '.png')

	results = cca_results + list(x) + \
			  [optm_min, mse, mse_contact_cca, mse_contact_single, r2_fitted, r2_contact_single]

	return y_fitted, results

def plot_linear_conv_filter_param(df):
	_, X_c = func_cca(df)
	X_c = X_c[:,0]
	y_norm = df['spike_binned'].values

	a_list = list(np.arange(200) * 0.0001)[1:]
	mse_list = []
	for a in a_list:
		weights = np.exp(- 1 / a * np.arange(1001) * 0.001)
		output_data = convolve1d(X_c, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))		
		mse = np.mean((output_data - y_norm)**2)
		mse_list.append(mse)
	plt.plot(a_list, mse_list)
	plt.show()

def compare_optimizer(df):
	cca_results, X_c = func_cca(df)
	X_c = X_c[:,0]
	y_norm = df['spike_binned'].values

	def linear_conv_filter(params, input_data, output_target):
		a = params[0]
		weights = np.exp(- 1 / a * np.arange(1001) * 0.001)
		output_data = convolve1d(input_data, weights=weights, 
								 mode='constant', cval=0, origin=-(len(weights)//2))		
		mse = np.mean((output_data - output_target)**2)
		return mse

	initial_params = [0.0001]
	result_minimize = minimize(linear_conv_filter, initial_params, args=(X_c, y_norm))
	r_x_min = result_minimize.x
	r_fun_min = result_minimize.fun

	bounds = [(0, 1)]
	result_diff_evo = differential_evolution(linear_conv_filter, bounds, args=(X_c, y_norm))
	r_x_diff_evo = result_diff_evo.x
	r_fun_diff_evo = result_diff_evo.fun

	r_fun_compare = r_fun_min - r_fun_diff_evo
	return r_fun_compare, r_x_diff_evo

def func_ARIMA(df, y, y_scaler, data_info_name, method, counts):
	y_fitted, _ = func_LN_model(df, y, y_scaler, data_info_name, method)
	y_norm = df['spike_binned'].values

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


def define_column(method):
	if method in ['LR', 'RF']:
		columns = ['unit_type', 'unit_name', 'block_id', 'trial_id', 
					'stimulus', 'vel', 'finger', 'force', 
					'length', 'r2', 'mse']
		for c_i in contact_feat:
			columns.append('imp_' + c_i)
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
						 'mse_fitted','mse_contact_cca', 'mse_contact_single', 
						 'r2_fitted', 'r2_contact_single']
		else:
			columns = columns + ['exp_conv_T', 'exp_conv_k', 'optm_min', 
							 'mse_fitted','mse_contact_cca', 'mse_contact_single', 
							 'r2_fitted', 'r2_contact_single']
	if method in ['ARIMA', 'LR_ARIMA']:
		columns = ['unit_type', 'unit_name', 'block_id', 'trial_id', 
					'stimulus', 'vel', 'finger', 'force', 
					'length', 'r2', 'mse', 'arima_p', 'arima_d', 'arima_q']
	return columns

def run_all_units(method):
	matplotlib.use("Agg")

	dir_data = 'D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/combined_data/new_axes/'
	dict_unit = {'SA-II': ['ST13-unit1', 'ST14-unit3', 'ST16-unit2', 'ST18-unit4'], 
				 'CT': ['ST13-unit2', 'ST16-unit3'], 
				 'HFA': ['ST13-unit3', 'ST14-unit2'],
				 'SA-I': ['ST14-unit1', 'ST16-unit5', 'ST18-unit2'], 
				 'Field': ['ST14-unit4', 'ST15-unit1', 'ST15-unit2', 'ST16-unit4', 'ST18-unit1']}

	norm_method = 'min_max' if method == 'LN_poisson' else 'z_score'

	bin_size = 0.1
	columns = []
	counts = [0, 0, 0] #all, short, non_stationary

	results_list = []
	for name in os.listdir(os.fsencode(dir_data)):
		name = os.fsdecode(name)
		unit_name = '-'.join(name.split('-')[3:5])
		print(unit_name)
		for key, value in dict_unit.items():
			if unit_name in value: unit_type = key; print(key)
		df = pd.read_csv(dir_data + name)
		df.dropna(subset = ['block_id', 'trial_id'], inplace=True)

		for block_i in df['block_id'].unique():
			df_b = df[df['block_id'] == block_i]
			for trial_i in df_b['trial_id'].unique():
				df_b_t = df_b[df_b['trial_id'] == trial_i]
				data_info = [unit_type, unit_name, int(block_i), int(trial_i), 
							df_b_t['stimulus'].values[0], int(df_b_t['vel'].values[0]),
							df_b_t['finger'].values[0][1:], df_b_t['force'].values[0][1:]]
				data_info_name = '_'.join(str(i) for i in data_info)
				df_norm, y, y_scaler = nerual_psth_smoothed_contact(df_b_t, bin_size=bin_size, norm_method=norm_method)

				if method in ['LR', 'RF']:
					results = func_regression(df_norm, y, y_scaler, data_info_name, method)
				if method in ['LR_ARIMA']:
					results = func_LR_ARIMA(df_norm, y, y_scaler, data_info_name, method, counts)
				if method in ['CCA']:
					results, _ = func_cca(df_norm, y, y_scaler, data_info_name)
				if method in ['LF', 'LN_poisson', 'LN_relu', 'LN_poly']:
					# plot_linear_conv_filter_param(df_norm)
					# r_compare, r_fun = compare_optimizer(df_norm)
					# results_list.append([r_compare, r_fun])
					_, results = func_LN_model(df_norm, y, y_scaler, data_info_name, method)
				if method in ['ARIMA']:
					func_ARIMA(df_norm, y, y_scaler, data_info_name, 'LN_poly', counts)
					continue	
				
				print(data_info + results)
				results_list.append(data_info + results)

	columns = define_column(method)
	df_results = pd.DataFrame(data = results_list, columns = columns)
	print(df_results)
	df_results.to_csv(data_dir + method + '_Results.csv', index=False, header=True)

	# plt.show()



def plot_regression_results(method):
	df_LR = pd.read_csv(data_dir + method + '_Results.csv')
	df_LR = df_LR[(df_LR['finger'] != 'two finger pads') & (df_LR['length'] >= 35)]
	df_LR.replace({'whole hand': 'hand', 'one finger tip': 'finger',
					'light force': 'low', 'moderate force': 'mid', 'strong force': 'high'}, inplace=True)
	df_LR.sort_values(['unit_type', 'stimulus', 'vel'], inplace=True)
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
		fig.savefig(plot_dir + 'figure_'+ method + '/compare_' + y_i +'.jpg')

	# plot accuracy/importance vs. combiend type
	sns.set(font_scale=1)
	fig, axes = plt.subplots(2, 4, sharex=True)    
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
	fig.savefig(plot_dir + 'figure_'+method+'/compare_combined.jpg')

	# plot importance 
	y_list =  ['imp_' + i for i in contact_feat]
	df_LR_imp = pd.melt(df_LR, id_vars=df_LR.columns.difference(y_list))
	print(df_LR_imp)
	fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)    
	fig.set_size_inches(5, 9, forward=True)
	sub_list = [(x, y) for x in unit_list for y in ['stroke', 'tap']]
	print(sub_list)
	for i in range(10):
		df_LR_imp_i = df_LR_imp[(df_LR_imp['unit_type'] == sub_list[i][0]) & (df_LR_imp['stimulus'] == sub_list[i][1])]
		ax = axes[i//2, i%2]
		sns.boxplot(df_LR_imp_i, x='variable', y='value', palette='crest', showfliers=False, ax=ax)
		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_title('_'.join(sub_list[i]))
		ax.xaxis.set_tick_params(rotation=60)
	sns.despine(trim=True)
	plt.tight_layout()
	plt.subplots_adjust(left=0.07, wspace=0.11)
	fig.savefig(plot_dir + 'figure_'+method+'/compare_all_importances.jpg')

	y_list = ['r2', 'mse'] + ['imp_' + i for i in contact_feat]
	for i in range(len(y_list)):
		fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)    
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
		fig.savefig(plot_dir + 'figure_'+method+'/compare_combined_' + y_list[i] + '.jpg')

	plt.show()

def plot_cca_results():
	sns.set(style="ticks",font_scale=1)
	df = pd.read_csv(data_dir + 'CCA_Results.csv')
	df = df[(df['finger'] != 'two finger pads') & (df['length'] >= 35)]
	df.replace({'whole hand': 'hand', 'one finger tip': 'finger',
				'light force': 'low', 'moderate force': 'mid', 'strong force': 'high'}, inplace=True)
	df.sort_values(['unit_type', 'stimulus', 'vel'], inplace=True)
	for y_i in ['corr_r2'] + ['loading_' + i for i in contact_feat]:
		df[y_i] = df[y_i].abs()

	df['combined'] = df['stimulus'] + '_' + df['vel'].astype(str)
	y_list = ['loading_' + i for i in contact_feat]
	for i in range(len(unit_list)):
		fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)    
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
		fig.savefig(plot_dir + 'figure_CCA/compare_combined_' + unit_list[i] + '.jpg')

	fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)    
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
	fig.savefig(plot_dir + 'figure_CCA/compare_loading_max.jpg')

	plt.show()

def plot_LN_results(LN):
	sns.set(style="ticks", font_scale=1)
	df = pd.read_csv(data_dir + LN + '_Results.csv')
	df = df[(df['finger'] != 'two finger pads') & (df['length'] >= 35)]
	df.replace({'whole hand': 'hand', 'one finger tip': 'finger',
				'light force': 'low', 'moderate force': 'mid', 'strong force': 'high'}, inplace=True)
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
	df_LR = df_LR[(df_LR['finger'] != 'two finger pads') & (df_LR['length'] >= 35)]
	ax.plot(df_LR['mse'].values - df['mse_fitted'].values, label='mse_LR-fitted', alpha=0.8)
	ax.plot(df['r2_fitted'].values - df_LR['r2'].values, label='r2_fitted-LR', alpha=0.8)
	ax.legend(frameon=False, loc=0)
	fig.savefig(plot_dir + 'figure_'+ LN +'/compare_mse_r2.jpg')
	print('mean r2 diff:', np.mean(df['r2_fitted'].values - df_LR['r2'].values))
	print('mean mse diff:', np.mean(df_LR['mse'].values - df['mse_fitted'].values))

	# plot exp filter parameters
	fig, axes = plt.subplots(2, 10, sharex=True, sharey='row')    
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
	fig.savefig(plot_dir + 'figure_' + LN + '/compare_LR_param.jpg')

	plt.show()

def plot_LR_ARIMA_results():
	sns.set(style="ticks", font_scale=1)
	df = pd.read_csv(data_dir + LN + '_Results.csv')
	df = df[(df['finger'] != 'two finger pads') & (df['length'] >= 35)]
	df.replace({'whole hand': 'hand', 'one finger tip': 'finger',
				'light force': 'low', 'moderate force': 'mid', 'strong force': 'high'}, inplace=True)
	df.sort_values(['stimulus', 'unit_type', 'vel'], inplace=True)
	df['combined'] = df['unit_type'] + '_' + df['stimulus']# + '_' + df['vel'].astype('str')

	df_LR = pd.read_csv(data_dir + 'LR_Results.csv')
	df_LR = df_LR[(df_LR['finger'] != 'two finger pads') & (df_LR['length'] >= 35)]

	fig, ax = plt.subplots(1,1)    
	fig.set_size_inches(7, 3)
	ax.plot(df_LR['mse'].values - df['ms'].values, label='mse_LR-fitted', alpha=0.8)
	ax.plot(df['r2'].values - df_LR['r2'].values, label='r2_fitted-LR', alpha=0.8)
	ax.legend(frameon=False, loc=0)
	fig.savefig(plot_dir + 'figure_LR_ARIMA/compare_mse_r2.jpg')
	print('mean r2 diff:', np.mean(df['r2'].values - df_LR['r2'].values))
	print('mean mse diff:', np.mean(df_LR['mse'].values - df['mse'].values))

	# plot exp filter parameters
	fig, axes = plt.subplots(3, 10, sharex=True, sharey='row')    
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
	fig.savefig(plot_dir + 'figure_LR_ARIMA/compare_LR_param.jpg')

	plt.show()

if __name__ == '__main__':
	sns.set(style="ticks", font='Arial', font_scale=1.8)
	contact_feat = ['area', 'depth', 'velAbs', 'velLat', 'velVert', 'velLong']
	unit_list = ['SA-I', 'SA-II', 'HFA', 'Field', 'CT']
	plot_dir = 'plots/psth_regression/'
	data_dir = 'data/psth_regression/'


	method = 'LR_ARIMA' #'ARIMA' # 'LF' # 'LN_poisson' # 'LN_poly' # 'CCA' # 'LR' # 'RF' # 'LN_relu' # 
	run_all_units(method)

	# plot_regression_results(method)
	# plot_cca_results()
	# plot_LN_results(method)


