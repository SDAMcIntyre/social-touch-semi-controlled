

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings

def fillNaN(data):
  t = np.arange(len(data))
  good = ~np.isnan(data)
  f = interp1d(t[good], data[good], bounds_error=False)
  filledData = np.where(np.isnan(data), f(t), data)
  filledData[np.isnan(filledData)] = 0
  return filledData


def find_dir_names(date, unit):
  database_path = "D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/"

  if date == '06-17' and unit == 'unit2':
    unit_foldername_standard2023 = '2022-06-17/unit2/' 
    unit_foldername = '2022-06-17_ST16-02/'
    ply_name = 'NoIR_2022-06-17_15-44-19_controlled-touch-MNG_ST16_2_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/'+unit_foldername_standard2023

  if date == '06-14' and unit == 'unit3':
    unit_foldername_standard2023 = '2022-06-14/unit3/'  
    unit_foldername = '2022-06-14_ST13-03/'
    ply_name = 'NoIR_2022-06-14_17-39-15_controlled-touch-MNG_ST13_3_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-14' and unit == 'unit2':
    unit_foldername_standard2023 = '2022-06-14/unit2/'
    unit_foldername = '2022-06-14_ST13-02/'
    ply_name = 'NoIR_2022-06-14_17-19-59_controlled-touch-MNG_ST13_2_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit5':
    unit_foldername_standard2023 = '2022-06-17/unit5/' 
    unit_foldername = '2022-06-17_ST16-05/'
    ply_name = 'NoIR_2022-06-17_17-36-59_ST16_unit5_01_love'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-16' and unit == 'unit1':
    unit_foldername_standard2023 = '2022-06-16/unit1/' 
    unit_foldername = '2022-06-16_ST15-01/'
    ply_name = 'NoIR_2022-06-16_14-56-13_controlled-touch-MNG_ST15_1_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/'+unit_foldername_standard2023

  if date == '06-16' and unit == 'unit2':
    unit_foldername_standard2023 = '2022-06-16/unit2/'
    unit_foldername = '2022-06-16_ST15-02/'
    ply_name = 'NoIR_2022-06-16_17-11-56_ST15_unit2_01_happiness'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit3':
    unit_foldername_standard2023 = '2022-06-17/unit3/'
    unit_foldername = '2022-06-17_ST16-03/'
    ply_name = 'NoIR_2022-06-17_16-52-18_controlled-touch-MNG_ST16_3_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit4':
    unit_foldername_standard2023 = '2022-06-17/unit4/'
    unit_foldername = '2022-06-17_ST16-04/'
    ply_name = 'NoIR_2022-06-17_17-20-51_controlled-touch-MNG_ST16_4_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit1':
    unit_foldername_standard2023 = '2022-06-15/unit1/' 
    unit_foldername = '2022-06-15_ST14-01/'
    ply_name = 'NoIR_2022-06-15_13-54-40_controlled-touch-MNG_ST14_01_block2'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit2':
    unit_foldername_standard2023 = '2022-06-15/unit2/'
    unit_foldername = '2022-06-15_ST14-02/'
    ply_name = 'NoIR_2022-06-15_15-09-09_controlled-touch-MNG_ST14_2_block3'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' # changed to this name
    Video_database_path = database_path+'videos_mp4/'+unit_foldername_standard2023
    handMesh_database_path = database_path+'ArmPLY/'+unit_foldername_standard2023

  if date == '06-15' and unit == 'unit3':
    unit_foldername_standard2023 = '2022-06-15/unit3/'
    unit_foldername = '2022-06-15_ST14-03/'
    ply_name = 'NoIR_2022-06-15_16-35-35_controlled-touch-MNG_ST14_3_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit4':
    unit_foldername_standard2023 = '2022-06-15/unit4/'
    unit_foldername = '2022-06-15_ST14-04/'
    ply_name = 'NoIR_2022-06-15_16-51-43_controlled-touch-MNG_ST14_4_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt' 
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit1':
    unit_foldername_standard2023 = '2022-06-22/unit1/'  #unit1-emotion
    unit_foldername = '2022-06-22_ST18-01/'
    ply_name = 'NoIR_2022-06-22_14-52-47_controlled-touch-MNG_ST18_1_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit2':
    unit_foldername_standard2023 = '2022-06-22/unit2/'
    unit_foldername = '2022-06-22_ST18-02/'
    ply_name = 'NoIR_2022-06-22_16-23-10_controlled-touch-MNG_ST18_2_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+ply_name+'-Normal.txt'
    # N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit4':
    unit_foldername_standard2023 = '2022-06-22/unit4/' 
    unit_foldername = '2022-06-22_ST18-04/'
    ply_name = 'NoIR_2022-06-22_17-26-19_controlled-touch-MNG_ST18_4_block2'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+'gravity_v.txt'
    handMesh_database_path = database_path+'ArmPLY/handmesh/'

  if date == '06-14' and unit == 'unit1':
    unit_foldername_standard2023 = '2022-06-14/unit1/'
    unit_foldername = '2022-06-14_ST13-01/'
    ply_name = 'NoIR_2022-06-14_15-42-43_controlled-touch-MNG_ST13_1_block1'
    N_Dir = database_path+'ArmPLY/'+unit_foldername_standard2023+ply_name+'-Normal.txt' 
    handMesh_database_path = database_path+'ArmPLY/'+unit_foldername_standard2023

  return database_path, unit_foldername_standard2023, unit_foldername, ply_name, N_Dir, handMesh_database_path


def set_parameters(unit_foldername_standard2023):
  contactThre = [100,1,100,100,50,100]

  if unit_foldername_standard2023 == '2022-06-17/unit2/':
    contactThre = [100,1,100,100,50,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
    # smoothWinSize01 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 9, (6,7,10,11): 33}
    # smoothWinSize02 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 5, (6,7,10,11): 11}    
    smoothWinSize01 = {(0,1,2,3,12,13,14,15): 151, (4,5,8,9): 301, (6,7,10,11): 1001}
    smoothWinSize02 = {(0,1,2,3,12,13,14,15): 151, (4,5,8,9): 151, (6,7,10,11): 301}

  if unit_foldername_standard2023 == '2022-06-17/unit5/':
    contactThre = [100,0.5,125,100,55,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
    smoothWinSize01 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 301, (6,7,14,15): 1001}
    smoothWinSize02 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 151, (6,7,14,15): 301}

  if unit_foldername_standard2023 == '2022-06-22/unit1/':
    contactThre = [100,1,100,100,55,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.6, (8,9,10,11):0.9, (12,13,14,15):1.2}
    smoothWinSize01 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 301, (6,7,14,15): 1001}
    smoothWinSize02 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 151, (6,7,14,15): 301}

  if unit_foldername_standard2023 == '2022-06-14/unit1/':
    contactThre = [100,1,100,40,50,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0}
    smoothWinSize01 = {(0,1): 151, (2,3): 301}
    smoothWinSize02 = {(0,1): 151, (2,3): 151}

  if unit_foldername_standard2023 == '2022-06-14/unit3/':
    pulse_duration0 = {(0,1):0, (2,3):0.5, (4,5):0.3, (6,7,8,9):0.9}
    smoothWinSize01 = {(2,3,4,5): 151, (0,1,6,7): 301, (8,9): 1001}
    smoothWinSize02 = {(2,3,4,5): 151, (0,1,6,7): 151, (8,9): 301}

  if unit_foldername_standard2023 == '2022-06-14/unit2/':
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3): 301, (4,5,6,7): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3): 151, (4,5,6,7): 301}

  if unit_foldername_standard2023 == '2022-06-15/unit1/':
    pulse_duration0 = {(0,1):0, (2,2):-0.3, (3,4,5,6):0.3, (7,8):0.8}
    smoothWinSize01 = {(0,1,7): 151, (2,3,4,8): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1,7): 151, (2,3,4,8): 151, (5,6): 301}

  if unit_foldername_standard2023 == '2022-06-15/unit2/':
    pulse_duration0 = {(0,1,2,3):0, (4,5):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3,4): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3,4): 151, (5,6): 301}

  if unit_foldername_standard2023 == '2022-06-15/unit3/' or unit_foldername_standard2023 == '2022-06-16/unit2/':
    pulse_duration0 = {(0,1):0}
    smoothWinSize01 = {(0,1): 151}
    smoothWinSize02 = {(0,1): 151}

  if unit_foldername_standard2023 == '2022-06-16/unit1/' or unit_foldername_standard2023 == '2022-06-17/unit4/':
    pulse_duration0 = {(0,1,2,3):0}
    smoothWinSize01 = {(0,1): 301, (2,3): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3): 301}

  if unit_foldername_standard2023 == '2022-06-17/unit3/':
    pulse_duration0 = {(0,1):0, (2,3,4):0.3}
    smoothWinSize01 = {(0,0): 151, (1,2,3): 301, (4,4): 1001}
    smoothWinSize02 = {(0,0): 151, (1,2,3): 151, (4,4): 301}

  if unit_foldername_standard2023 == '2022-06-15/unit4/':
    pulse_duration0 = {(0,0):-2.1, (1,2):0}
    smoothWinSize01 = {(0,1): 151, (2,2): 301}
    smoothWinSize02 = {(0,1): 151, (2,2): 151}

  if unit_foldername_standard2023 == '2022-06-22/unit4/':
    pulse_duration0 = {(0,1,2):0, (3,4,5,6):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3,4): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3,4): 151, (5,6): 301}

  if unit_foldername_standard2023 == '2022-06-22/unit2/':  
    pulse_duration0 = {(0,0):0, (1,2):-0.2}
    smoothWinSize01 = {(0,1): 151, (2,2): 301}
    smoothWinSize02 = {(0,1): 151, (2,2): 151}

  pulse_duration = {}
  for k, v in pulse_duration0.items():
      for key in k:
        pulse_duration[key] = v  
  smoothWinSize1 = {}
  for k, v in smoothWinSize01.items():
    for key in k:
      smoothWinSize1[key] = v
  smoothWinSize2 = {}
  for k, v in smoothWinSize02.items():
    for key in k:
      smoothWinSize2[key] = v

  return pulse_duration, smoothWinSize1, smoothWinSize2, contactThre




def get_contact_df(ContactQuantities_dir='',intplt=False,a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100):
  Contact_Quan_df = pd.read_csv(ContactQuantities_dir) 
  # Contact_Quan_df = Contact_Quan_df.iloc[int(contact_time[0]*30):int(contact_time[1]*30)]
  # Gesture_df = pd.read_csv(gesture_dir) 

  # change mm to cm
  Contact_Quan_df['Contact_Area'] = Contact_Quan_df['Contact_Area']/100.*3.
  Contact_Quan_df['Depth'] = Contact_Quan_df['Depth']/20.
  Contact_Quan_df['Vel_Abs'] = Contact_Quan_df['Vel_Abs']/10.
  Contact_Quan_df['Vel_Vert'] = Contact_Quan_df['Vel_Vert']/10.
  Contact_Quan_df['Vel_Late'] = Contact_Quan_df['Vel_Late']/10.
  Contact_Quan_df['Vel_Longi'] = Contact_Quan_df['Vel_Longi']/10.
  for feat in ['velAbsRaw', 'velVertRaw', 'velLatRaw', 'velLongRaw']:
    if feat in Contact_Quan_df.columns:
      Contact_Quan_df[feat] = Contact_Quan_df[feat] / 10.

  # remove hand mismatching datapoints  
  Contact_Quan_df['Contact_Area'] = Contact_Quan_df.where(Contact_Quan_df['Contact_Area']<a_thre,other=np.nan)['Contact_Area']
  Contact_Quan_df['Depth'] = Contact_Quan_df.where(Contact_Quan_df['Depth']<d_thre,other=np.nan)['Depth']
  Contact_Quan_df['Vel_Abs'] = Contact_Quan_df.where(Contact_Quan_df['Vel_Abs']<vabs_thre,other=np.nan)['Vel_Abs']
  Contact_Quan_df['Vel_Vert'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Vert']<vv_thre)&(Contact_Quan_df['Vel_Vert']>-vv_thre),other=np.nan)['Vel_Vert']
  Contact_Quan_df['Vel_Late'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Late']<vlt_thre)&(Contact_Quan_df['Vel_Late']>-vlt_thre),other=np.nan)['Vel_Late']
  Contact_Quan_df['Vel_Longi'] = Contact_Quan_df.where((Contact_Quan_df['Vel_Longi']<vlg_thre)&(Contact_Quan_df['Vel_Longi']>-vlg_thre),other=np.nan)['Vel_Longi']
  # Contact_Quan_df['Vel_Longi'][0:50*30] = Contact_Quan_df.where((Contact_Quan_df['Vel_Longi']<50)&(Contact_Quan_df['Vel_Longi']>-50),other=np.nan)['Vel_Longi'][0:50*30]
  if intplt==True:
    Contact_Quan_df = Contact_Quan_df.astype(float).interpolate()

  return Contact_Quan_df



def combine_all_trials_semi_control_new(date, unit,split_trial=False,new_axes=False):

  database_path, unit_foldername_standard2023, unit_foldername, _, _, _ = find_dir_names(date, unit)

  Output_Dir = database_path+'combined_data/manual_axes_3Dposition_RF/'
  video_database_path = database_path+'videos_mp4/'+unit_foldername_standard2023  
  neural_database_path = database_path+'synched_neural_video_data/'+unit_foldername
  stimuli_database_path = database_path+'stimuli_log/'+unit_foldername_standard2023   
  cq_database_path = database_path+'contact_quantities_RF/'+unit_foldername_standard2023
  position_3d_database_path = database_path+'contact_quantities_RF/'+unit_foldername_standard2023

  combine_df = pd.DataFrame()
  t_start = 0.0

  pulse_duration, smoothWinSize1, smoothWinSize2, cThre = set_parameters(unit_foldername_standard2023)

  for i in range(len(os.listdir(video_database_path))):  # iterate through all file
    name = os.listdir(video_database_path)[i]
    print(name[:-4])
    ContactQuantities_dir = cq_database_path+name[:-4]+'-ContQuant.csv'
    position_3d_dir = position_3d_database_path+name[:-4]+'-ContQuantAll.csv'
    i_neural = os.listdir(neural_database_path)[i]
    NeuralData_dir = neural_database_path+i_neural
    if len(NeuralData_dir) == 0:
      print('----no matched neural data-----', video_name)
      exit()

    Contact_Quan_df = get_contact_df(ContactQuantities_dir,False,cThre[0],cThre[1],cThre[2],cThre[3],cThre[4],cThre[5])
    Contact_Quan_df.rename(columns={'Contact_Area':'area', 'Depth':'depth', 'Vel_Abs':'velAbs',
                                    'Vel_Late':'velLat', 'Vel_Longi':'velLong', 'Vel_Vert':'velVert'}, inplace=True)
    position_3d_df = pd.read_csv(position_3d_dir) 
    position_columns = ['Position_x', 'Position_y', 'Position_z', 'Position_index_x', 'Position_index_y', 'Position_index_z', 'arm_points']
    Contact_Quan_df[position_columns] = position_3d_df[position_columns]

    # delay_neural = delay_neural0 + pulse_duration[i] - i*0.1
    delay_neural = 0
    Contact_Quan_df['t'] = Contact_Quan_df['t'].values + delay_neural
    Neural_Data_df = pd.read_csv(NeuralData_dir) 
    Neural_Data_df.rename(columns={'Nerve_freq':'IFF', 'Nerve_spike':'spike'}, inplace=True)
    trial_df = pd.DataFrame()

    ### crop contact/neural data according to the shorter one
    step = 0.001
    # t_spike = Neural_Data_df['t_Spike'].values.round(decimals=3)
    # t_spike_end = t_spike[-1] 
    # t_spike_raw = Neural_Data_df['t_Spike'].values  ## keep the highest IFF resolution with 1kHZ sampling rate
    spike_col = Neural_Data_df['IFF']
    t_spike_end_idx = spike_col.last_valid_index()
    t_spike = np.array(range(t_spike_end_idx + 1)) * step
    Neural_Data_df = Neural_Data_df[Neural_Data_df.index <= t_spike_end_idx]
    print(len(t_spike), len(Neural_Data_df.index))
    Neural_Data_df['t'] = t_spike
    Neural_Data_df = Neural_Data_df.round({'t': 3})
    t_spike_end = t_spike[-1]
    t_contact_end = Contact_Quan_df['t'].values[-1].round(decimals=3)
    if t_contact_end <= t_spike_end:
      t_spike = t_spike[t_spike <= t_contact_end]
    elif not split_trial:
      Contact_Quan_df = Contact_Quan_df.drop(Contact_Quan_df[Contact_Quan_df['t'] > t_spike_end].index)

    if split_trial:
      t_end = max(t_spike_end, t_contact_end)
      t_plot = np.arange(0.0,t_end+step,step).round(decimals=3)
    else:
      t_plot = np.arange(0.0,t_spike[-1]+step,step).round(decimals=3)

    # ### get IFF and Spike
    # f_plot = [.0]*len(t_plot)
    # spike_plot = [0]*len(t_plot)
    # ij = 0
    # for ii in range(len(t_plot)):
    #   if t_plot[ii] in t_spike:
    #     spike_plot[ii] = 1
    #   if ij < len(t_spike) and abs(t_plot[ii] - t_spike[ij]) < 0.0004:
    #     if ij == 0: 
    #       f_plot[ii] = 1./t_spike_raw[ij]
    #       if t_spike_raw[ij] < 1:
    #         f_plot[ii] = 0.0001
    #     else:
    #       f_plot[ii] = 1./(t_spike_raw[ij]-t_spike_raw[ij-1])
    #     ij += 1
    trial_df['t'] = t_plot + t_start
    t_start = t_plot[-1] + step + t_start
    # if f_plot[-1] != 0: f_plot[-1] = 0.


    # trial_df['IFF'] = Neural_Data_df['Nerve_freq']
    # trial_df['spike'] = Neural_Data_df['Nerve_spike']
    trial_df['block_id'] = [i+1] * len(trial_df.index)
    # print(letn(t_spike), sum(np.array(f_plot) != 0))

    ### get features and direvatives
    for feature in ['area', 'depth', 'velAbs', 'velLat', 'velLong', 'velVert']:
      q_feat = Contact_Quan_df[feature].values

      if feature in ['area', 'depth', 'velAbs']:
        Q_upper = np.nanpercentile(q_feat, 99.9)
        q_feat = np.where(q_feat<=Q_upper,q_feat,np.nan)
      else:
        Q_upper = np.nanpercentile(q_feat, 99.9)
        Q_lower = np.nanpercentile(q_feat, 0.1)
        q_feat = np.where((q_feat<=Q_upper) & (q_feat>=Q_lower),q_feat,np.nan)

      f_f = interp1d(Contact_Quan_df['t'],q_feat,fill_value="extrapolate")
      interpolated = f_f(t_plot)
      if feature in {'area', 'depth', 'velAbs'}:
        interpolated[interpolated < 0] = 0
      if feature+'Raw' not in list(Contact_Quan_df.columns):
        trial_df[feature+'Raw'] = interpolated
      interpolated = fillNaN(interpolated)
      trial_df[feature] = interpolated
      # smooth
      smoothed = savgol_filter(interpolated,smoothWinSize1[i],3)
      if feature in {'area', 'depth', 'velAbs'}:
        smoothed[smoothed < 0] = 0
      trial_df[feature+'Smooth'] = smoothed
      # 1st derivative
      smoothed1D = np.diff(smoothed) / step
      smoothed1D = np.insert(smoothed1D, 0, 0)
      trial_df[feature+'1D'] = smoothed1D
      # 2nd derivative
      smoothed1D = savgol_filter(smoothed1D,smoothWinSize2[i],4)
      smoothed2D = np.diff(smoothed1D) / step
      smoothed2D = np.insert(smoothed2D, 0, 0)
      trial_df[feature+'2D'] = smoothed2D
    for feature in ['velAbsRaw', 'velLatRaw', 'velLongRaw', 'velVertRaw']:
      if feature in list(Contact_Quan_df.columns):
        q_feat = Contact_Quan_df[feature].values
        if feature == 'velAbsRaw':
          Q_upper = np.nanpercentile(q_feat, 99)
          q_feat = np.where(q_feat<=Q_upper,q_feat,np.nan)
        else:
          Q_upper = np.nanpercentile(q_feat, 99)
          Q_lower = np.nanpercentile(q_feat, 1)
          q_feat = np.where((q_feat<=Q_upper) & (q_feat>=Q_lower),q_feat,np.nan)
        f_f = interp1d(Contact_Quan_df['t'],q_feat,fill_value="extrapolate")
        interpolated = f_f(t_plot)
        if feature == 'velAbsRaw':
          interpolated[interpolated < 0] = 0
        trial_df[feature] = interpolated
    trial_df['Contact_Flag'] = np.where(trial_df['areaRaw'] > 0, 1, 0)

    ### add 3d positions
    for feature in position_columns[:-1]:
      point_ = Contact_Quan_df[feature].values
      f_f = interp1d(Contact_Quan_df['t'],point_,fill_value="extrapolate")
      trial_df[feature] = f_f(t_plot)

    ### add neural data
    for feature in ['IFF', 'spike']:
      point_ = Neural_Data_df[feature].values
      f_f = interp1d(Neural_Data_df['t'],point_,fill_value="extrapolate")
      trial_df[feature] = f_f(t_plot)

    trial_df['arm_points'] = None
    for i, row in Contact_Quan_df.iterrows():
      trial_df.loc[trial_df['t']==row['t'], 'arm_points'] = row['arm_points']
    trial_df['arm_points'] = trial_df['arm_points'].ffill()


    ### get trial info
    if split_trial:
      trial_df['t'] = trial_df['t'] - trial_df['t'].values[0]
    trial_df['trial_id'] = [np.nan] * len(trial_df.index)
    os.chdir(stimuli_database_path)
    for i_stimuli in os.listdir():
      if i_stimuli.split('_')[-1] != 'log.csv': continue
      stimuli_name = i_stimuli[:16]
      video_name = name[5:][:16]
      print(stimuli_name, video_name)
      if stimuli_name == video_name:
        print('yes')
        stimuli_dir = stimuli_database_path+i_stimuli
        stimuli_log_df = pd.read_csv(stimuli_dir, names=['time', 'event', 'vel', 'finger', 'force'], header=None) #, usecols=[0,1]
        stimuli_log_df = stimuli_log_df.iloc[1:].reset_index()
        stimuli_log_df['time'] = stimuli_log_df['time'].apply(pd.to_numeric)
        print(stimuli_log_df)
    os.chdir(video_database_path)

    block_id = name[-5]
    for idx, row in stimuli_log_df.iterrows():
      marker1 = 'told triggerbox to send '
      if marker1 in row['event'] and block_id == row['event'][len(marker1)]:
        start_idx = idx - 2
      marker2 = 'complete'
      if marker2 in row['event'] and block_id == row['event'][6]:
        end_idx = idx
    stimuli_i_df = stimuli_log_df.iloc[start_idx:end_idx - 2]
    stimuli_i_df.loc[:,'time'] = stimuli_i_df['time'].values - stimuli_i_df['time'].values[0]
    # trial_df['block_id'] = int(block_id)

    trial_id = 0
    # trial_start_adj = {('2022-06-17/unit3/', 2, 3): 1, ('2022-06-17/unit3/', 3, 2): 2.5 , ('2022-06-17/unit5/', 3, 3): 1,
    #                    ('2022-06-15/unit4/', 1, 4): -2.5, ('2022-06-15/unit4/', 1, 5): -2.5, ('2022-06-15/unit4/', 1, 6): -2.5, 
    #                    ('2022-06-15/unit4/', 1, 7): -2.5, ('2022-06-15/unit4/', 1, 8): -3, ('2022-06-14/unit2/', 3, 4): 0.8,
    #                    ('2022-06-14/unit2/', 4, 2): 1, ('2022-06-14/unit2/', 4, 3): 1, ('2022-06-14/unit2/', 4, 4): 1, 
    #                    ('2022-06-14/unit2/', 8, 2): 1, ('2022-06-14/unit2/', 8, 3): 1}
    trial_df['stimulus'] = None
    trial_df['vel'] = None
    trial_df['finger'] = None
    trial_df['force'] = None
    for idx, row in stimuli_i_df.iterrows():
      if row['event'] == 'TTL/LED on':
        trial_id += 1
        start_t = row['time']
        trial_temp_idx = (unit_foldername_standard2023, trial_df['block_id'].values[0], trial_id)
        # if trial_temp_idx in trial_start_adj: start_t += trial_start_adj[trial_temp_idx]
        stimuli_type = stimuli_i_df.loc[idx+1, 'event'].split(': ')[-1]
        vel = float(stimuli_i_df.loc[idx+1, 'vel'].split('c')[0])
        finger = stimuli_i_df.loc[idx+1, 'finger']
        force = stimuli_i_df.loc[idx+1, 'force']
      if row['event'] == 'TTL/LED off':
        end_t = row['time']
        # trial_id = int(stimuli_i_df.loc[idx+1, 'event'].split(' ')[1])
        if split_trial:
          trial_df.loc[(trial_df['t']>start_t) & (trial_df['t']<end_t), ['trial_id', 'stimulus', 'vel', 'finger', 'force']] = [trial_id, stimuli_type, vel, finger, force]
        else:
          t_0 = trial_df['t'].values[0]
          trial_df.loc[(trial_df['t']-t_0>start_t) & (trial_df['t']-t_0<end_t), ['trial_id', 'stimulus', 'vel', 'finger', 'force']] = [trial_id, stimuli_type, vel, finger, force]

    for c_i in ['trial_id', 'stimulus', 'vel', 'finger', 'force']:
      trial_df[c_i].interpolate(method='pad', inplace=True)

    combine_df = combine_df.append(trial_df)
       
    if split_trial:
      path = database_path+'contact_IFF_trial/'+unit_foldername_standard2023
      isExist = os.path.exists(path)
      if not isExist:
         os.makedirs(path)
      trial_df.to_csv(path+name[5:-4]+'.csv', index = False, header=True)

  if not split_trial:
    combine_df.to_csv(Output_Dir+unit_foldername_standard2023[:10]+'-'+name[46:50]+'-'+unit_foldername_standard2023[11:-1]+'-semicontrol.csv', index = False, header=True)

  ### plot
  # columns = combine_df.columns.difference(['spike','IFF','t', 'stimulus', 'vel', 'finger', 'force'])
  # for ai in range(6):
  #   n = 5
  #   t = combine_df['t'].values
  #   sns.set(style = "ticks", font = 'Arial')#, font_scale = 2.7)
  #   fig, axs = plt.subplots(n, 1, sharex='all')
  #   fig.set_size_inches(8, 3 * n, forward = True)

  #   for i in range(n):
  #     ax = axs[i]
  #     ax.plot(t, combine_df[columns[ai*n + i]], color = 'darkseagreen', linewidth = 1)
  #     ax.set_ylabel(columns[ai*n + i])
  #   dir_plot = 'combined/feature_new_axes' if new_axes else 'combined/feature'
  #   fig.savefig(database_path+'plots/'+unit_foldername_standard2023+dir_plot+str(ai)+'.png')

  plot_var = ['block_id', 'trial_id', 'area', 'velAbs', 'velLong', 'spike', 'IFF']
  fig, axs = plt.subplots(len(plot_var), sharex=True)
  fig.set_size_inches(16, 10)
  for i in range(len(plot_var)):
    axs[i].plot(combine_df['t'],combine_df[plot_var[i]],color='darkseagreen',linewidth=1.5)
    axs[i].yaxis.set_label_coords(-0.05,0.5)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].set(ylabel=plot_var[i])
    # if i>3 and i<7:
    #   axs[i].set(yticks=[-25,25])
  dir_plot = 'combined/All_with_RF'
  fig.savefig(database_path+'plots/'+unit_foldername_standard2023+dir_plot+'.png')
  # plt.show()



if __name__ == '__main__':

    # Define the date and unit pairs
    data = [
        ('06-17', 'unit2'),
        ('06-14', 'unit3'),
        ('06-14', 'unit2'),
        ('06-17', 'unit5'),
        ('06-16', 'unit1'),
        ('06-16', 'unit2'),
        ('06-17', 'unit3'),
        ('06-17', 'unit4'),
        ('06-15', 'unit1'),
        ('06-15', 'unit2'),
        ('06-15', 'unit3'),
        ('06-15', 'unit4'),
        ('06-22', 'unit1'),
        ('06-22', 'unit2'),
        ('06-22', 'unit4'),
        ('06-14', 'unit1')
    ]

    for date, unit in data:
        combine_all_trials_semi_control_new(date, unit)


    # video_idx = list(range(10))
    # # video_idx = [2]
    # vis_all_RF(date, unit, video_idx, iff_vis_thre=5)

    # plt.show()
