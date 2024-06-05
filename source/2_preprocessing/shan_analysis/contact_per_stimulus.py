import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler


def get_contact_df(ContactQuantities_dir='',a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100):
    Contact_Quan_df = pd.read_csv(ContactQuantities_dir) 
    
    # change mm to cm
    Contact_Quan_df['Contact_Area'] = Contact_Quan_df['Contact_Area']/100.*3.
    Contact_Quan_df['Depth'] = Contact_Quan_df['Depth']/20.
    Contact_Quan_df['Vel_Abs'] = Contact_Quan_df['Vel_Abs']/10.
    Contact_Quan_df['Vel_Vert'] = Contact_Quan_df['Vel_Vert']/10.
    Contact_Quan_df['Vel_Late'] = Contact_Quan_df['Vel_Late']/10.
    Contact_Quan_df['Vel_Longi'] = Contact_Quan_df['Vel_Longi']/10.
    
    # remove hand mismatching datapoints  
    Contact_Quan_df['Contact_Area'] = Contact_Quan_df.where(Contact_Quan_df['Contact_Area']<a_thre,other=np.nan)['Contact_Area']
    Contact_Quan_df['Depth'] = Contact_Quan_df.where(Contact_Quan_df['Depth']<d_thre,other=np.nan)['Depth']
    Contact_Quan_df['Vel_Abs'] = Contact_Quan_df.where(Contact_Quan_df['Vel_Abs']<vabs_thre,other=np.nan)['Vel_Abs']
    Contact_Quan_df['Vel_Vert'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Vert']<vv_thre)&(Contact_Quan_df['Vel_Vert']>-vv_thre),other=np.nan)['Vel_Vert']
    Contact_Quan_df['Vel_Late'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Late']<vlt_thre)&(Contact_Quan_df['Vel_Late']>-vlt_thre),other=np.nan)['Vel_Late']
    Contact_Quan_df['Vel_Longi'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Longi']<vlg_thre)&(Contact_Quan_df['Vel_Longi']>-vlg_thre),other=np.nan)['Vel_Longi']
    
    return Contact_Quan_df

def fillNaN(data):
    t = np.arange(len(data))
    good = ~np.isnan(data)
    f = interp1d(t[good], data[good], bounds_error=False)
    filledData = np.where(np.isnan(data), f(t), data)
    filledData[np.isnan(filledData)] = 0
    return filledData

def combine_all_trails_emotion(dir_0,dir_1):
    cwd = os.getcwd()
    output_name = dir_1.replace('/','_')[:-1]
    output_dir = 'contact_per_emotion/'
    input_dir = dir_0 + dir_1
    os.chdir(input_dir)  # Change the directory

    combine_df = pd.DataFrame()
    t_start = 0.0  
    smoothWinSize = {'a':5, 'g':9, 'l':33, 's':33, 'h':9, 'c':33}
    smoothWinSize2 = {'a':5, 'g':5, 'l':11, 's':11, 'h':5, 'c':11}

    for i in range(len(os.listdir())):  # iterate through all file
        name = os.listdir()[i]
        if name[-5] == 'l': continue
        print(name.rpartition('-')[0])
        emotion = name.split('_')[-1].split('-')[0]
        ContactQuantities_dir = dir_0+dir_1+name

        Contact_Quan_df = get_contact_df(ContactQuantities_dir)
        Contact_Quan_df.rename(columns={'Contact_Area':'area', 'Depth':'depth', 'Vel_Abs':'velAbs',
          'Vel_Late':'velLat', 'Vel_Longi':'velLong', 'Vel_Vert':'velVert'}, inplace=True)
        if np.nansum(Contact_Quan_df['area'].values) == 0:
            continue

        trial_df = pd.DataFrame()    
        for feature in list(Contact_Quan_df.columns):
            if feature!='Contact_Flag' and feature!='t':
                if emotion in ['calming','gratitude','sadness'] and feature == 'depth':
                    f_f = interp1d(Contact_Quan_df['t'],Contact_Quan_df[feature].values/2,fill_value="extrapolate")
                else:
                    f_f = interp1d(Contact_Quan_df['t'],Contact_Quan_df[feature],fill_value="extrapolate")
                trial_df[feature] = f_f(Contact_Quan_df['t'])
                # smooth
                smoothed = savgol_filter(trial_df[feature].values,smoothWinSize[name[39]],3)
                if feature in {'area', 'depth', 'velAbs'}:
                    smoothed[smoothed < 0] = 0
                trial_df[feature+'Smooth'] = fillNaN(smoothed)
                # 1st derivative
                smoothed1D = np.diff(smoothed)
                smoothed1D = np.insert(smoothed1D, 0, 0)
                trial_df[feature+'1D'] = fillNaN(smoothed1D)
                # 2nd derivative
                smoothed1D = savgol_filter(smoothed1D,smoothWinSize2[name[39]],4)
                smoothed2D = np.diff(smoothed1D)
                smoothed2D = np.insert(smoothed2D, 0, 0)
                trial_df[feature+'2D'] = fillNaN(smoothed2D)
        trial_df['emotion'] = [emotion] * len(trial_df.index) 
        trial_df['trail'] = [int(name.split('_')[-2])] * len(trial_df.index)
        combine_df = combine_df.append(trial_df)

    os.chdir(cwd)
    combine_df.to_csv(output_dir+output_name+'.csv', index = False, header=True)

def contact_per_trail_emotion(exps):
    input_dir = 'contact_per_emotion/'
    combine_df = pd.DataFrame()
    columns = ['areaSmooth', 'depthSmooth','velAbsSmooth','velLatSmooth','velLongSmooth','velVertSmooth',
                'area1D', 'depth1D','velAbs1D','velLat1D','velLong1D','velVert1D']
    x_ticks = ['areaS', 'depthS','vAbsS','vLatS','vLongS','vVertS',
                'area1D', 'depth1D','vAbs1D','vLat1D','vLong1D','vVert1D']

    for exp in exps:
        contact_df = pd.read_csv(input_dir+exp+'.csv') 
        contact_df.replace(0, np.nan, inplace=True)
        contact_df = contact_df.dropna()
        combine_df = combine_df.append(contact_df)
    for feature in columns:
        vals = np.abs(combine_df[feature].values)
        upper = np.percentile(vals, 98)
        print(vals, upper)
        vals[vals > upper] = np.nan
        combine_df[feature] = vals

    sns.set(style="ticks", font='Arial', font_scale=1.8)
    fig = plt.figure()
    fig.set_size_inches(20,9,forward=True)
    i = 1
    for feature in columns:
        ax = fig.add_subplot(2,6,i)
        sns.boxplot(x='emotion',y=feature,data=combine_df,fliersize=0)
        ax.set_title(feature, pad=-50)
        ax.set_xlabel('')
        ax.set_ylabel('')
        sns.despine(trim=True)
        ax.set_xticklabels(['A','L','C','G','H','S'])
        i += 1
    plt.subplots_adjust(left=0.057, right=0.96, top=0.96, bottom=0.15,hspace=0.38)
    fig.savefig('plots/emotion_contact.png',dpi=800)


    for feature in columns:
        vals = combine_df[feature].values
        vals_norm = MinMaxScaler().fit_transform(vals.reshape(-1, 1))[:,0]
        combine_df[feature] = vals_norm      
    combine_df = pd.melt(combine_df, id_vars=['emotion'], value_vars=columns)

    sns.set(style="ticks", font='Arial', font_scale=1.8)
    fig = plt.figure()
    fig.set_size_inches(14,9,forward=True)
    i = 1
    for emo in combine_df['emotion'].unique():
        df_emo = combine_df[combine_df['emotion'] == emo]
        ax = fig.add_subplot(3,2,i)
        sns.boxplot(x='variable',y='value',data=df_emo,fliersize=0)
        ax.set_title(emo, pad=-50)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(None,1)
        ax.set_yticks([0,0.5,1])
        ax.set_xticklabels(x_ticks)        
        ax.xaxis.set_tick_params(rotation=60)
        sns.despine(trim=True)
        if i != 5 and i != 6:
            ax.set_xticklabels(['']*12)
        i += 1
    plt.subplots_adjust(left=0.057, right=0.96, top=0.96, bottom=0.15,hspace=0.38)
    fig.savefig('plots/emotion_contact_norm.png',dpi=800)
    plt.show()

def set_parameters(dir_1):
  if dir_1 == '2022-06-17/unit2/':
    contactThre = [500,1,500,500,500,500]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    gesture0 = {(0,1,12,13):'tap',(2,3,14,15):'stroke',(4,5,8,9):'press',(6,7,10,11):'slide'}
    smoothWinSize01 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 9, (6,7,10,11): 33}
    smoothWinSize02 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 5, (6,7,10,11): 11}
  if dir_1 == '2022-06-17/unit5/':
    contactThre = [500,1,500,500,500,500]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    gesture0 = {(0,1,8,9):'tap',(2,3,10,11):'stroke',(4,5,12,13):'press',(6,7,14,15):'slide'}
    smoothWinSize01 = {(0,1,2,3,8,9,10,11): 5, (4,5,12,13): 9, (6,7,14,15): 33}
    smoothWinSize02 = {(0,1,2,3,8,9,10,11): 5, (4,5,12,13): 5, (6,7,14,15): 11}

  gesture = {}
  for k, v in gesture0.items():
      for key in k:
        gesture[key] = v  
  smoothWinSize1 = {}
  for k, v in smoothWinSize01.items():
    for key in k:
      smoothWinSize1[key] = v
  smoothWinSize2 = {}
  for k, v in smoothWinSize02.items():
    for key in k:
      smoothWinSize2[key] = v

  return gesture, smoothWinSize1, smoothWinSize2, contactThre

def combine_all_trails_semiControl(dir_0,dir_1):
    cwd = os.getcwd()
    output_name = dir_1.replace('/','_')[:-1]
    output_dir = 'contact_per_semiControl/'
    input_dir = dir_0 + dir_1
    os.chdir(input_dir)  # Change the directory

    combine_df = pd.DataFrame() 
    gesture, smoothWinSize1, smoothWinSize2, cThre = set_parameters(dir_1)

    ii = 0
    for i in range(len(os.listdir())):  # iterate through all file
        name = os.listdir()[i]
        if name[-5] == 'l': continue
        print(name.rpartition('-')[0])
        ContactQuantities_dir = dir_0+dir_1+name

        Contact_Quan_df = get_contact_df(ContactQuantities_dir,cThre[0],cThre[1],cThre[2],cThre[3],cThre[4],cThre[5])
        Contact_Quan_df.rename(columns={'Contact_Area':'area', 'Depth':'depth', 'Vel_Abs':'velAbs',
          'Vel_Late':'velLat', 'Vel_Longi':'velLong', 'Vel_Vert':'velVert'}, inplace=True)
        if np.nansum(Contact_Quan_df['area'].values) == 0:
            continue

        trial_df = pd.DataFrame()    
        for feature in list(Contact_Quan_df.columns):
            if feature!='Contact_Flag' and feature!='t':
                f_f = interp1d(Contact_Quan_df['t'],Contact_Quan_df[feature],fill_value="extrapolate")
                trial_df[feature] = f_f(Contact_Quan_df['t'])
                # smooth
                smoothed = savgol_filter(Contact_Quan_df[feature].values,smoothWinSize1[ii],3)
                if feature in {'area', 'depth', 'velAbs'}:
                    smoothed[smoothed < 0] = 0
                trial_df[feature+'Smooth'] = fillNaN(smoothed)
                # 1st derivative
                smoothed1D = np.diff(smoothed)
                smoothed1D = np.insert(smoothed1D, 0, 0)
                trial_df[feature+'1D'] = fillNaN(smoothed1D)
                # 2nd derivative
                smoothed1D = savgol_filter(smoothed1D,smoothWinSize2[ii],4)
                smoothed2D = np.diff(smoothed1D)
                smoothed2D = np.insert(smoothed2D, 0, 0)
                trial_df[feature+'2D'] = fillNaN(smoothed2D)
        trial_df['gesture'] = [gesture[ii]] * len(trial_df.index) 
        trial_df['trail'] = [ii+1] * len(trial_df.index)
        trial_df.replace(0, np.nan, inplace=True)
        trial_df = trial_df.dropna()
        temp_columns = list(trial_df.columns)
        c_columns = [i for i in temp_columns if i not in ['gesture','trial']]
        for feature in c_columns:
            vals = np.abs(trial_df[feature].values)
            trial_df[feature] = vals
        combine_df = combine_df.append(trial_df)
        ii += 1

    os.chdir(cwd)
    combine_df.to_csv(output_dir+output_name+'.csv', index = False, header=True)

def contact_per_trail_semiControl(exps):
    input_dir = 'contact_per_semiControl/'
    combine_df = pd.DataFrame()
    columns = ['area', 'depth','velAbs','velLat','velLong','velVert',
                'area1D', 'depth1D','velAbs1D','velLat1D','velLong1D','velVert1D']
    x_ticks = ['area', 'depth','vAbs','vLat','vLong','vVert',
                'area1D', 'depth1D','vAbs1D','vLat1D','vLong1D','vVert1D']

    for exp in exps:
        contact_df = pd.read_csv(input_dir+exp+'.csv') 
        # contact_df.replace(0, np.nan, inplace=True)
        # contact_df = contact_df.dropna()
        combine_df = combine_df.append(contact_df)
    # for feature in columns:
    #     vals = combine_df[feature].values
        # upper = np.percentile(vals, 98)
        # print(vals, upper)
        # vals[vals > upper] = np.nan
        # combine_df[feature] = vals

    sns.set(style="ticks", font='Arial', font_scale=1.8)
    fig = plt.figure()
    fig.set_size_inches(20,9,forward=True)
    plot_df = combine_df[(combine_df['gesture']=='tap') | (combine_df['gesture']=='stroke')]
    i = 1
    for feature in columns:
        ax = fig.add_subplot(2,6,i)
        sns.boxplot(x='gesture',y=feature,data=plot_df,showfliers=False)
        ax.set_title(feature, pad=-50)
        ax.set_xlabel('')
        ax.set_ylabel('')
        sns.despine(trim=True)
        i += 1
    plt.subplots_adjust(left=0.057, right=0.96, top=0.96, bottom=0.15,hspace=0.38)
    fig.savefig('plots/semiControl_contact_gesture.png',dpi=800)

    for feature in columns:
        vals = combine_df[feature].values
        vals_norm = MinMaxScaler().fit_transform(vals.reshape(-1, 1))[:,0]
        combine_df[feature] = vals_norm      
    combine_df = pd.melt(combine_df, id_vars=['gesture'], value_vars=columns)

    sns.set(style="ticks", font='Arial', font_scale=1.8)
    fig = plt.figure()
    fig.set_size_inches(8,4,forward=True)
    i = 1
    for ges in ['tap','stroke']:
        df_ges = combine_df[combine_df['gesture'] == ges]
        ax = fig.add_subplot(1,2,i)
        sns.barplot(x='variable',y='value',data=df_ges)
        ax.set_title(ges, pad=-50)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(None,1)
        ax.set_yticks([0,0.5,1])
        ax.set_xticklabels(x_ticks)        
        ax.xaxis.set_tick_params(rotation=60)
        sns.despine(trim=True)
        i += 1
    plt.subplots_adjust(left=0.098, right=0.95, top=0.88, bottom=0.32)
    fig.savefig('plots/semiControl_gesture.png',dpi=800)
    plt.show()


if __name__ == '__main__':
    dir_0 = "D:/Projects/Hand_tracking/OtherProjects/SwedenData/contact_quantities/"
    dir_1 = '2021-12-08/unit1-1/'
    # dir_1 = '2021-12-08/unit1-2/'

    # combine_all_trails_emotion(dir_0,dir_1)

    exps = ['2021-12-08_unit1-1','2021-12-08_unit1-2']
    contact_per_trail_emotion(exps)


    # dir_1 = '2022-06-17/unit2/'
    # dir_1 = '2022-06-17/unit5/'
    # combine_all_trails_semiControl(dir_0,dir_1)

    # exps = ['2022-06-17_unit5']
    # contact_per_trail_semiControl(exps)