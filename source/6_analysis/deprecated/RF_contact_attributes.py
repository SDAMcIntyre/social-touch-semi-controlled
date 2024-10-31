import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
#from wrappers import ModelPipeline
from utils import *
from sympy import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
from scipy.signal import butter,filtfilt,savgol_filter
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns
import copy
import os
import warnings
from collections import Counter
from ast import literal_eval


def Receptive_field_track(video_dir,ArmPLY_dir,CorrectedJointsColor_dir,
  root_shift0,Other_Shifts,hand_rotate0,
  color_idx,ArmNParas,N_dir,left,video_start,handMesh_Dir,ContactQuantities_dir, df_neural):
  """
  calculate contact attribtues and receptive field
  """
  capture = cv2.VideoCapture(video_dir)
  capture.set(cv2.CAP_PROP_POS_FRAMES, video_start)
  if capture.isOpened():
    hasFrame, frame = capture.read()
  else:
    hasFrame = False
  connections = [(0, 1), (1, 2), (2, 3), (3, 4),
      (5, 6), (6, 7), (7, 8),
      (9, 10), (10, 11), (11, 12),
      (13, 14), (14, 15), (15, 16),
      (17, 18), (18, 19), (19, 20),
      (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

  ############ output visualization ############
  view_mat0 = axangle2mat([hand_rotate0[0],hand_rotate0[1],hand_rotate0[2]],hand_rotate0[3]) # align different coordinate systems
  window_width = 1000

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat0, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()
  hand_pcd = o3d.geometry.PointCloud() # hand PC from mesh vertices
  marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
  marker_sphere.paint_uniform_color([0, 0, 1])
  thumb_idx = np.where(hand_mesh.verts[:,2] > 0.05)    # remove thumb for contact quantities
  index_idx = np.where((hand_mesh.verts[:,0] > 0.065) & (hand_mesh.verts[:,2] > 0.02))  # index finger tip for recalibration
  palm_idx = np.where((hand_mesh.verts[:,0]>-0.045)&(hand_mesh.verts[:,0]<-0.035)&
             (hand_mesh.verts[:,2]>-0.01)&(hand_mesh.verts[:,2]<0.01)&(hand_mesh.verts[:,1]>0)) # palm green point

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_width + 1, height=window_width + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)
  viewer.add_geometry(marker_sphere)
  mesh_smoother = OneEuroFilter(4.0, 0.0)

  ######## arm point cloud ###############
  arm_pcd = o3d.io.read_point_cloud(ArmPLY_dir)
  arm_points = np.asarray(arm_pcd.points)
  if left == True:
    arm_points_left = []
    for point in arm_points:
      point[0] = -point[0]
      arm_points_left.append(point)
    arm_pcd.points = o3d.utility.Vector3dVector(np.array(arm_points_left))
    arm_points = np.asarray(arm_pcd.points)

  n_arm_points = arm_points.shape[0]
  arm_pcd.estimate_normals()
  arm_pcd.orient_normals_to_align_with_direction(([0., 0., -1.]))
  arm_normals = np.asarray(arm_pcd.normals)
  arm_neibor_dis = arm_pcd.compute_nearest_neighbor_distance()
  arm_point_dis = np.mean(arm_neibor_dis)
  # arm_Normal = copy.deepcopy(arm_normals[int(n_arm_points/ArmNParas[0])+ArmNParas[1]])
  arm_Normal = np.genfromtxt(N_dir, delimiter=',')
  arm_Normal = arm_Normal/np.linalg.norm(arm_Normal)
  # np.savetxt(N_dir,arm_Normal,delimiter=',')  
  viewer.add_geometry(arm_pcd)
  arm_intesect_pcd = o3d.geometry.PointCloud()
  arm_intesect_pcd.paint_uniform_color([1, 0, 0])
  viewer.add_geometry(arm_intesect_pcd)
  points = [arm_points[int(n_arm_points/ArmNParas[0])+ArmNParas[1]],
            arm_points[int(n_arm_points/ArmNParas[0])+ArmNParas[1]]+150*arm_Normal]
  lines=[[0,1]]
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(points)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  viewer.add_geometry(line_set)

  view_control = viewer.get_view_control()
  view_control.rotate(0,1000)
  view_control.scale(.001)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()

  ############ input visualization ############
  pygame.init()
  display = pygame.display.set_mode((500, 500))
  pygame.display.set_caption('Minimal Hand - input')
  clock = pygame.time.Clock()

  ############ model and correction ############
  corrected_joints_color = np.genfromtxt(CorrectedJointsColor_dir, delimiter=',')
  for colorxyz in corrected_joints_color:  #interpolation for the missing color marker position
    if colorxyz[0]==0 and colorxyz[1]==0 and colorxyz[2]==0: 
      colorxyz[0]=colorxyz[1]=colorxyz[2]=nan   
    if colorxyz[3]==0 and colorxyz[4]==0 and colorxyz[5]==0: 
      colorxyz[3]=colorxyz[4]=colorxyz[5]=nan
    if colorxyz[6]==0 and colorxyz[7]==0 and colorxyz[8]==0: 
      colorxyz[6]=colorxyz[7]=colorxyz[8]=nan
  corrected_joints_color[0] = [0]*9
  # for i in range(9):
  #   corrected_joints_color[840:1040,i] = pd.Series(corrected_joints_color[840:1040,i]).interpolate().values
  frame_n = int(corrected_joints_color.shape[0])
  corrected_joints_color = np.vstack((np.zeros((2,9)),corrected_joints_color))  #3,9
  if left ==True:
    corrected_joints_color[:,0] = -corrected_joints_color[:,0]
    corrected_joints_color[:,3] = -corrected_joints_color[:,3]
    corrected_joints_color[:,6] = -corrected_joints_color[:,6]
  # corrected_joints = np.genfromtxt(CorrectedJoints_dir, delimiter=',')  
  # corrected_joints = np.vstack((np.zeros((3,3)),corrected_joints))
  ## identify best hand marker based on the least NaN
  n_zeros = np.count_nonzero(np.isnan(corrected_joints_color), axis=0)
  color_idx_ = int(np.argmin(n_zeros)/3)
  if color_idx == -1: color_idx = color_idx_

  # model = ModelPipeline()
  v_handMesh = np.loadtxt(handMesh_Dir)
  if left == False:
    v_handMesh[:,1] = -v_handMesh[:,1]
    v_handMesh[:,0] = -v_handMesh[:,0]

  ########## Contact quantities ############
  frame_idx = video_start  #int(0*30) #
  t = []
  contact_flag = []
  contact_area = []
  depth = []
  vel_abs = []
  vel_vert = []
  vel_late = []
  vel_longi = []  
  position_x = []
  position_y = []
  position_z = []
  position_index_x = []
  position_index_y = []
  position_index_z = []
  position_red_x = []
  position_red_y = []
  position_red_z = []
  position_green_x = []
  position_green_y = []
  position_green_z = []
  position_yellow_x = []
  position_yellow_y = []
  position_yellow_z = []
  arm_p_idx_list = []
  iff_list = []
  depth_mean = 0
  vel_p = [0,0,0]
  vel_p_index = [0,0,0]
  vel_p_red = [0,0,0]
  vel_p_green = [0,0,0]
  vel_p_yellow = [0,0,0]
  fig, ax = plt.subplots()
  fig.set_size_inches(8,5)
  plt.ion()


  neural_data = df_neural['Nerve_freq'].values
  # t_temp = df_neural['t'].values
  t_temp = np.array(range(len(neural_data))) / 1000.
  # t_temp = t_temp + neural_shift

  # replace zeros as the next IFF value
  f = interp1d(t_temp, neural_data, kind='next', bounds_error=False, fill_value=0)
  # neural_t = df_neural['t'].values
  # neural_t = neural_t[~np.isnan(neural_t)]
  # t_new = np.arange(neural_t[0], neural_t[-1], 1/30.)
  # IFFinterp = f(t_new)
  # IFFinterp[np.isnan(IFFinterp)] = 0
  # neural_i = 0
  arm_p_idx = []

  while hasFrame:
    # seperate vedio for different position shift
    root_shift = root_shift0
    for Shift_param in Other_Shifts:
      if Shift_param[0].size > 2:
        periods = np.split(Shift_param[0],Shift_param[0].size/2)
      else:
        periods =  np.array([Shift_param[0]])
      for period in periods:
        if frame_idx>=period[0] and frame_idx<=period[1]:
          root_shift = Shift_param[1] 
    print(frame_idx)#, root_shift)

    v = np.matmul(view_mat0, v_handMesh.T).T    
    v[:,0] = -v[:,0]
    vel_p_prev = vel_p
    vel_p_index_prev = vel_p_index
    vel_p_red_prev = vel_p_red
    vel_p_green_prev = vel_p_green
    vel_p_yellow_prev = vel_p_yellow
    red_p = corrected_joints_color[frame_idx,0:3]
    green_p = corrected_joints_color[frame_idx,3:6]
    yellow_p = corrected_joints_color[frame_idx,6:9]
    # root_p = corrected_joints[frame_idx]  
    fingernail_p = np.mean(v[index_idx[0],:],axis=0)
    palm_p = np.mean(v[palm_idx[0],:],axis=0)
    #### calibration using root & color 
    v0 = v
    # v = v0 * 1000 + root_p + root_shift.T ;p_temp = root_p
    # print('color_idx=',color_idx)   
    if color_idx==0: p_temp=red_p
    if color_idx==1: p_temp=green_p
    if color_idx==2: p_temp=yellow_p
    if color_idx==3: 
      p_temp=red_p
      v = v0 * 1000 + p_temp - fingernail_p*1000 + root_shift.T  
    else:  
      v = v0 * 1000 + p_temp - palm_p*1000 + root_shift.T
    # v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    v_no_thumb = np.delete(v, thumb_idx[0], axis=0)
    marker_sphere.translate(p_temp, relative=False)
    vel_p = np.mean(v[palm_idx[0],:],axis=0) #np.mean(v[index_idx[0],:],axis=0)
    vel_p_index = np.mean(v[index_idx[0],:],axis=0)
    vel_p_red = red_p
    vel_p_green = green_p
    vel_p_yellow = yellow_p
    ######## calculate contact quantites ################
    ### contact detection
    hand_inside = []
    arm_intersect = []
    arm_p_idx_i = []
    hand_arm_dis = []
    arm_norm = []
    depth_prev = depth_mean
    for i_hand in range(v_no_thumb.shape[0]):
      hand_arm_vector = v_no_thumb[i_hand]-arm_pcd.points
      hand_arm_min_idx = np.argmin(np.sum(np.abs(hand_arm_vector)**2,axis=-1))
      if np.dot(hand_arm_vector[hand_arm_min_idx],arm_pcd.normals[hand_arm_min_idx]) <= 0:
        arm_point_color = arm_pcd.colors[hand_arm_min_idx]
        if arm_point_color[0] > 0.9 and arm_point_color[1] > 0.9 and arm_point_color[2] > 0.9: # for hnad mesh with missing parts
          continue 
        hand_inside.append(v_no_thumb[i_hand])
        arm_intersect.append(arm_pcd.points[hand_arm_min_idx])
        arm_p_idx_i.append(hand_arm_min_idx)
        hand_arm_dis.append(np.linalg.norm(hand_arm_vector[hand_arm_min_idx]))
        arm_norm.append(arm_pcd.normals[hand_arm_min_idx])
    # print(np.array(arm_intersect).shape)
    arm_Longitudinal = [0,1,0]-np.dot([0,1,0],arm_Normal)*arm_Normal #same direction as camera y axis
    arm_Longitudinal = arm_Longitudinal/np.linalg.norm(arm_Longitudinal)
    arm_Lateral = np.cross(arm_Longitudinal,arm_Normal) # opposite direction of camera x axis
    arm_Lateral = arm_Lateral/np.linalg.norm(arm_Lateral)   
    
    iff_frame = f(frame_idx / 30.0)
    if np.array(arm_intersect).size > 0:
      arm_intesect_pcd.points = o3d.utility.Vector3dVector(np.array(arm_intersect))
      arm_intesect_pcd.paint_uniform_color([1, 0, 0])
      arm_intersect_uniq = np.unique(np.array(arm_intersect), axis=0)
      arm_p_idx_i = list(np.unique(np.array(arm_p_idx_i), axis=0))
      arm_p_idx += arm_p_idx_i
      cont_flag_i = 1        
      depth_mean = np.mean(hand_arm_dis)
      cont_area_i = arm_intersect_uniq.shape[0]*arm_point_dis**2
      vel_i = (vel_p-vel_p_prev)/(1./30)
      vel_abs_i = np.linalg.norm(vel_i)
      vel_vert_i = np.dot(vel_i,arm_Normal) #in vertical direction of the arm
      vel_late_i = np.dot(vel_i,arm_Lateral) # in lateral direction of the arm
      vel_longi_i = np.dot(vel_i,arm_Longitudinal) #in longitudinal direction of the arm   
    # print(vel_i,vel_abs_i,vel_vert_i,vel_late_i,vel_longi_i)
      print(np.array(arm_p_idx_i).shape)
    else:
      arm_intesect_pcd.clear()
      arm_p_idx_i = []
      cont_flag_i = 0
      depth_mean = 0
      cont_area_i = 0
      vel_i = [0,0,0]
      vel_abs_i = 0
      vel_vert_i = 0
      vel_late_i = 0
      vel_longi_i = 0


    t.append(frame_idx*(1./30))
    contact_area.append(cont_area_i)
    contact_flag.append(cont_flag_i)
    depth.append(depth_mean)
    vel_abs.append(vel_abs_i)
    vel_vert.append(vel_vert_i)
    vel_late.append(vel_late_i)
    vel_longi.append(vel_longi_i)
    position_x.append(vel_p[0])
    position_y.append(vel_p[1])
    position_z.append(vel_p[2])
    position_index_x.append(vel_p_index[0])
    position_index_y.append(vel_p_index[1])
    position_index_z.append(vel_p_index[2])
    position_red_x.append(vel_p_red[0])
    position_red_y.append(vel_p_red[1])
    position_red_z.append(vel_p_red[2])
    position_green_x.append(vel_p_green[0])
    position_green_y.append(vel_p_green[1])
    position_green_z.append(vel_p_green[2])
    position_yellow_x.append(vel_p_yellow[0])
    position_yellow_y.append(vel_p_yellow[1])
    position_yellow_z.append(vel_p_yellow[2])
    arm_p_idx_list.append(arm_p_idx_i)
    if iff_frame:
      print('----',np.round(iff_frame))
      iff_list.append(iff_frame)
    else:
      iff_list.append(np.nan)



    # ### contact quantities visualization
    # contact_quant_plot = {'cont_flag':[cont_flag_i*50],'cont_area':[cont_area_i/100.],
    #                       'depth':[depth_mean/10.],'vel_abs':[vel_abs_i/10.],'vel_vert':[vel_vert_i/10.],
    #                       'vel_late':[vel_late_i/10.],'vel_longi':[vel_longi_i/10.]}
    # df = pd.DataFrame(data=contact_quant_plot)    
    # sns.set(style="whitegrid", font="Arial", font_scale=3)
    # clrs = ['grey','darkseagreen','darkseagreen','darkseagreen','darkseagreen','darkseagreen','darkseagreen']
    # ax = sns.barplot(data=df,order=['cont_flag', 'cont_area','depth','vel_abs','vel_vert','vel_late','vel_longi'],palette=clrs)
    # ax.set_ylim(-50,50)
    # ax.set_xticklabels(('F','A','D','V','Vvt','Vlt','Vlg'))
    # # ax.set_ylabel('cm')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # plt.subplots_adjust(bottom=.14,left=.15,top=.88)
    # plt.draw()
    # plt.pause(0.00000001)
    # plt.clf()

    ########### visualization ############################
    # for some version of open3d you may need `viewer.update_geometry(mesh)`
    viewer.update_geometry()
    viewer.poll_events()

    frame_show = frame[(int(frame.shape[0]/2)-500):(int(frame.shape[0]/2)+500),
                      (int(frame.shape[1]/2)-500):(int(frame.shape[1]/2)+500)]
    frame_show = np.flip(frame_show, -1).copy() # BGR to RGB
    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_show, (500, 500)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()


    ########## 
    frame_idx += 1
    hasFrame, frame = capture.read()
    if frame_idx >= frame_n:
      hasFrame = False
    clock.tick(30)
    if keyboard.is_pressed("esc"):
      break

  print(frame_idx)
  capture.release()
  cv2.destroyAllWindows()

  Contact_Quan_df = pd.DataFrame({'t':t,'Contact_Flag':contact_flag,'Contact_Area':contact_area,
                                  'Depth':depth,'Vel_Abs':vel_abs,'Vel_Vert':vel_vert,
                                  'Vel_Late':vel_late,'Vel_Longi':vel_longi})
  Contact_Quan_df.to_csv(ContactQuantities_dir+'.csv', index = False, header=True)
  Contact_Quan_All_df = pd.DataFrame({'t':t,'Contact_Flag':contact_flag,'Contact_Area':contact_area,'Depth':depth,
                    'Position_x':position_x,'Position_y':position_y,'Position_z':position_z,
                    'Position_index_x':position_index_x,'Position_index_y':position_index_y,'Position_index_z':position_index_z,
                    'Position_red_x':position_red_x,'Position_red_y':position_red_y,'Position_red_z':position_red_z,
                    'Position_green_x':position_green_x,'Position_green_y':position_green_y,'Position_green_z':position_green_z,
                    'Position_yellow_x':position_yellow_x,'Position_yellow_y':position_yellow_y,'Position_yellow_z':position_yellow_z,
                    'arm_points':arm_p_idx_list, 'IFF':iff_list})
  Contact_Quan_All_df.to_csv(ContactQuantities_dir+'All.csv', index = False, header=True)


  return arm_p_idx




def find_neural_delay(dir_1, video_idx):

  dict_sync = {'2022-06-14/unit1/': -2.8, '2022-06-14/unit2/': -2, '2022-06-14/unit3/': -2.9,
               '2022-06-15/unit1/': -2.7, '2022-06-15/unit2/': -2.8, '2022-06-15/unit3/': -2.7, '2022-06-15/unit4/': -2.8,
               '2022-06-16/unit1/': -2.9, '2022-06-16/unit2/': -2.75,            
               '2022-06-17/unit2/': -2.8, '2022-06-17/unit3/': -2.3, '2022-06-17/unit4/': -2.8, '2022-06-17/unit5/': -2.7,
               '2022-06-22/unit1/': -2.7, '2022-06-22/unit2/': -2.6, '2022-06-22/unit4/': -2.7}
  neural_delay = -dict_sync[dir_1]

  if dir_1 == '2022-06-17/unit2/':
      pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
  if dir_1 == '2022-06-17/unit5/':
      pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
  if dir_1 == '2022-06-22/unit1/':
      pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.6, (8,9,10,11):0.9, (12,13,14,15):1.2}
  if dir_1 == '2022-06-14/unit1/':
      pulse_duration0 = {(0,1,2,3):0}
  if dir_1 == '2022-06-14/unit3/':
      pulse_duration0 = {(0,1):0, (2,3):0.5, (4,5):0.3, (6,7,8,9):0.9}
  if dir_1 == '2022-06-14/unit2/':
      pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3}
  if dir_1 == '2022-06-15/unit1/':
      pulse_duration0 = {(0,1):0, (2,2):-0.3, (3,4,5,6):0.3, (7,8):0.8}
  if dir_1 == '2022-06-15/unit2/':
      pulse_duration0 = {(0,1,2,3):0, (4,5):0.3}
  if dir_1 == '2022-06-15/unit3/' or dir_1 == '2022-06-16/unit2/':
      pulse_duration0 = {(0,1):0}
  if dir_1 == '2022-06-16/unit1/' or dir_1 == '2022-06-17/unit4/':
      pulse_duration0 = {(0,1,2,3):0}
  if dir_1 == '2022-06-17/unit3/':
      pulse_duration0 = {(0,1):0, (2,3,4):0.3}
  if dir_1 == '2022-06-15/unit4/':
      pulse_duration0 = {(0,0):-2.1, (1,2):0}
  if dir_1 == '2022-06-22/unit4/':
      pulse_duration0 = {(0,1,2):0, (3,4,5,6):0.3}
  if dir_1 == '2022-06-22/unit2/':  
      pulse_duration0 = {(0,0):0, (1,2):-0.2}

  for k, v in pulse_duration0.items():
    for video_idx in k:
        neural_delay += v  

  return neural_delay

def find_dir_names(date, unit):
  dir_0 = "D:/OneDrive - University of Virginia/Projects/Hand_tracking/MNG_tracking/data/"

  if date == '06-17' and unit == 'unit2':
    dir_1 = '2022-06-17/unit2/' 
    dir_11 = '2022-06-17_ST16-02/'
    ply_name = 'NoIR_2022-06-17_15-44-19_controlled-touch-MNG_ST16_2_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/'+dir_1

  if date == '06-14' and unit == 'unit3':
    dir_1 = '2022-06-14/unit3/'  
    dir_11 = '2022-06-14_ST13-03/'
    ply_name = 'NoIR_2022-06-14_17-39-15_controlled-touch-MNG_ST13_3_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-14' and unit == 'unit2':
    dir_1 = '2022-06-14/unit2/'
    dir_11 = '2022-06-14_ST13-02/'
    ply_name = 'NoIR_2022-06-14_17-19-59_controlled-touch-MNG_ST13_2_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit5':
    dir_1 = '2022-06-17/unit5/' 
    dir_11 = '2022-06-17_ST16-05/'
    ply_name = 'NoIR_2022-06-17_17-36-59_ST16_unit5_01_love'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-16' and unit == 'unit1':
    dir_1 = '2022-06-16/unit1/' 
    dir_11 = '2022-06-16_ST15-01/'
    ply_name = 'NoIR_2022-06-16_14-56-13_controlled-touch-MNG_ST15_1_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/'+dir_1

  if date == '06-16' and unit == 'unit2':
    dir_1 = '2022-06-16/unit2/'
    dir_11 = '2022-06-16_ST15-02/'
    ply_name = 'NoIR_2022-06-16_17-11-56_ST15_unit2_01_happiness'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit3':
    dir_1 = '2022-06-17/unit3/'
    dir_11 = '2022-06-17_ST16-03/'
    ply_name = 'NoIR_2022-06-17_16-52-18_controlled-touch-MNG_ST16_3_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-17' and unit == 'unit4':
    dir_1 = '2022-06-17/unit4/'
    dir_11 = '2022-06-17_ST16-04/'
    ply_name = 'NoIR_2022-06-17_17-20-51_controlled-touch-MNG_ST16_4_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit1':
    dir_1 = '2022-06-15/unit1/' 
    dir_11 = '2022-06-15_ST14-01/'
    ply_name = 'NoIR_2022-06-15_13-54-40_controlled-touch-MNG_ST14_01_block2'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit2':
    dir_1 = '2022-06-15/unit2/'
    dir_11 = '2022-06-15_ST14-02/'
    ply_name = 'NoIR_2022-06-15_15-09-09_controlled-touch-MNG_ST14_2_block3'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' # changed to this name
    Video_Dir_0 = dir_0+'videos_mp4/'+dir_1
    handMesh_Dir_0 = dir_0+'ArmPLY/'+dir_1

  if date == '06-15' and unit == 'unit3':
    dir_1 = '2022-06-15/unit3/'
    dir_11 = '2022-06-15_ST14-03/'
    ply_name = 'NoIR_2022-06-15_16-35-35_controlled-touch-MNG_ST14_3_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-15' and unit == 'unit4':
    dir_1 = '2022-06-15/unit4/'
    dir_11 = '2022-06-15_ST14-04/'
    ply_name = 'NoIR_2022-06-15_16-51-43_controlled-touch-MNG_ST14_4_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit1':
    dir_1 = '2022-06-22/unit1/'  #unit1-emotion
    dir_11 = '2022-06-22_ST18-01/'
    ply_name = 'NoIR_2022-06-22_14-52-47_controlled-touch-MNG_ST18_1_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit2':
    dir_1 = '2022-06-22/unit2/'
    dir_11 = '2022-06-22_ST18-02/'
    ply_name = 'NoIR_2022-06-22_16-23-10_controlled-touch-MNG_ST18_2_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+ply_name+'-Normal.txt'
    # N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-22' and unit == 'unit4':
    dir_1 = '2022-06-22/unit4/' 
    dir_11 = '2022-06-22_ST18-04/'
    ply_name = 'NoIR_2022-06-22_17-26-19_controlled-touch-MNG_ST18_4_block2'
    N_Dir = dir_0+'ArmPLY/'+dir_1+'gravity_v.txt'
    handMesh_Dir_0 = dir_0+'ArmPLY/handmesh/'

  if date == '06-14' and unit == 'unit1':
    dir_1 = '2022-06-14/unit1/'
    dir_11 = '2022-06-14_ST13-01/'
    ply_name = 'NoIR_2022-06-14_15-42-43_controlled-touch-MNG_ST13_1_block1'
    N_Dir = dir_0+'ArmPLY/'+dir_1+ply_name+'-Normal.txt' 
    handMesh_Dir_0 = dir_0+'ArmPLY/'+dir_1

  return dir_0, dir_1, dir_11, ply_name, N_Dir, handMesh_Dir_0

def find_parameters(date, unit, i, handMesh_Dir_0):
      
  Other_Shifts = [[np.dot([0,0],30),np.array([0,0,0])]]
  Root_Shift0 = np.array([0,0,0])
  neural_shift = 0

  if date == '06-17' and unit == 'unit2':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    # function_opt = 'constantMesh'

    # if i!=6:  # 6,7 stroking weak
    #   continue        
    if i%2 == 0: #15-44-19
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,0,-1,0.3]; Root_Shift0 = np.array([0,0,10])
      if i == 0:
        Root_Shift0 = np.array([0,0,0])
        Shift_Time1 = np.dot([57,65,81,88],30);Root_Shift1 = np.array([0,0,8])   
        Other_Shifts = [[Shift_Time1,Root_Shift1]]
      if i == 6: 
        Root_Shift0 = np.array([-5,5,10])
    if i%2 == 1: #15-46-06
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,0,0,1]; Root_Shift0 = np.array([0,0,0]) 
      if i == 5:
         Hand_Rotate0 = [-1,0.3,0,1]

  if date == '06-14' and unit == 'unit3':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3 
    function_opt = 'constantMesh' 
    Crop_Position = [-350,-300]
    Crop_Size = 544

    # if i!=0: 
    #   return None,None,None,None,None,None,None,True 

    if i%2==0:
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [0,-1,0.5,1.5]; Root_Shift0 = np.array([0,0,5])  
      if i==0:
        Hand_Rotate0 = [0,-1,0.5,1.5]; Root_Shift0 = np.array([0,0,-5])  
        Shift_Time1 = np.dot([0,7,30,33,56.5,59.4],30);Root_Shift1 = np.array([0,0,-25])  
        Other_Shifts = [[Shift_Time1,Root_Shift1]]
      if i==2:#17-45-04
        Color_idx = 2
        handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
        Hand_Rotate0 = [0,-1,0,0.5]; Root_Shift0 = np.array([15,0,-15])  
        Shift_Time5 = np.dot([5,13.5],30); Root_Shift5 = np.array([10,0,-15])  
        Shift_Time1 = np.dot([40.5,49.5],30); Root_Shift1 = np.array([30,0,-15])  
        Shift_Time2 = np.dot([49.5,57.5],30); Root_Shift2 = np.array([35,0,-10])  
        Shift_Time3 = np.dot([65,82],30); Root_Shift3 = np.array([25,0,-15])  
        Shift_Time4 = np.dot([82,105],30); Root_Shift4 = np.array([10,0,-25])  
        Other_Shifts = [[Shift_Time1,Root_Shift1],[Shift_Time2,Root_Shift2],
                        [Shift_Time3,Root_Shift3],[Shift_Time4,Root_Shift4],
                        [Shift_Time5,Root_Shift5]]
    if i%2==1: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices0.txt' 
      Hand_Rotate0 = [0.2,-1,0,1]; Root_Shift0 = np.array([10,0,10]) 
      if i==1:
        Hand_Rotate0 = [0.2,-1,0,1]; Root_Shift0 = np.array([10,0,5]) 
      if i==3:  #17-46-52
        Shift_Time1 = np.dot([58,104],30); Root_Shift1 = np.array([10,0,10])  
        Other_Shifts = [[Shift_Time1,Root_Shift1]]
    
  if date == '06-14' and unit == 'unit2':
    ArmNParas = [200000,290]
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    if i!=0:
      return None,None,None,None,None,None,None,True    

    if i%2==0: #17-19-59
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,-1,0.5,0]; Root_Shift0 = np.array([0,0,-2])  
    if i%2==1: #17-21-48
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.1,0.2,1]; Root_Shift0 = np.array([0,0,-5])  

    if i==3:
      Root_Shift0 = np.array([0,0,10]) 
    if i==5 or i==7:
      Hand_Rotate0 = [-1,-0.1,0.2,1.2]; Root_Shift0 = np.array([0,0,5]) 

  if date == '06-17' and unit == 'unit5':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    # if i<8: 
    #   return None,None,None,None,None,None,None,True      

    if i%2==0: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,-1,-0.5,0.8]; Root_Shift0 = np.array([-5,0,10])  
      if i == 0:
        Root_Shift0 = np.array([0,0,5])   
    if i%2==1: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.4,0.2,1]; Root_Shift0 = np.array([0,0,10])  
      if i == 1 or i == 11:
        Shift_Time1 = np.dot([0,4.8],30); Root_Shift1 = np.array([0,0,-150])        
        Other_Shifts = [[Shift_Time1,Root_Shift1]] 

  if date == '06-16' and unit == 'unit1':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'
    # if i!=2:
    #   return None,None,None,None,None,None,None,True   
    if i%2==0: #14-56-13
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [0,-1,0,0.8]; Root_Shift0 = np.array([0,0,0])  
    if i%2==1: #14-57-36
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.4,0.2,1.4]; Root_Shift0 = np.array([0,0,5])  
    if i == 2: Root_Shift0 = np.array([0,0,-3])  

  if date == '06-16' and unit == 'unit2':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'
    # neural_shift = 0.74
 
    if i==0: #17-18-16
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [0,-1,0,0.8]; Root_Shift0 = np.array([0,0,10])  
      Shift_Time1 = np.dot([980,1530], 1); Root_Shift1 = np.array([0,0,10])        
      Other_Shifts = [[Shift_Time1,Root_Shift1]]

  if date == '06-17' and unit == 'unit3':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'
    # if i!=4: continue   

    if i in [0, 1, 2, 4]: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,0,-1,0.3]; Root_Shift0 = np.array([0,0,0])#16-52-18
    if i == 3: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,0,0,1]; Root_Shift0 = np.array([0,0,0]) #16-54-05 
    if i==1 or i==4:
      Hand_Rotate0 = [0,-1,-1,0.4]; Root_Shift0 = np.array([0,0,3]) #16-55-53
      if i == 4: Root_Shift0 = np.array([0,0,5])
    if i == 0:
      Shift_Time1 = np.dot([80,130],30);Root_Shift1 = np.array([0,0,2])   
      Other_Shifts = [[Shift_Time1,Root_Shift1]]

  if date == '06-17' and unit == 'unit4':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    # if i!=1:
    #   continue    
    if i==0: #17-20-51
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [0,-1,0,0.8]; Root_Shift0 = np.array([0,0,0])  
    if i==1: #17-22-15
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.1,0.2,0.9]; Root_Shift0 = np.array([0,0,5])

  if date == '06-15' and unit == 'unit1':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    if i<=5:
      return None,None,None,None,None,None,None,True 

    if i in [0, 2, 3, 5, 7]: #13-53-17
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [0,-1,0.5,1]; Root_Shift0 = np.array([0,0,-2])  
      
    if i in [1, 4, 6, 8]: #13-54-40
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,0,0.2,1.2]; Root_Shift0 = np.array([0,0,8])  

    if i == 0:
      Color_idx = 2; Root_Shift0 = np.array([30,20,-45])
      Shift_Time1 = np.dot([24,48],30); Root_Shift1 = np.array([30,20,-35])        
      Other_Shifts = [[Shift_Time1,Root_Shift1]] 
      # neural_shift = -0.15
    # if i == 1:
      # neural_shift = 0.5
    if i == 2:
      Root_Shift0 = np.array([0,0,0])
      # neural_shift = 1.3
    if i == 3:
      Shift_Time1 = np.dot([0,7],30); Root_Shift1 = np.array([0,0,-145])        
      Other_Shifts = [[Shift_Time1,Root_Shift1]]  
      # neural_shift = 1.2
    if i == 5:
      Root_Shift0 = np.array([0,0,5])
    if i == 7:
      Root_Shift0 = np.array([0,0,5]); Hand_Rotate0 = [0,-0.2,0,0.5]

  if date == '06-15' and unit == 'unit2':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    ArmNParas = [200000,497]#296
    function_opt = 'constantMesh'  

    if i in [0, 2, 4, 5]: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices.txt'#new one captured by camera
      Hand_Rotate0 = [-0.5,-.3,0.3,1.5]; Root_Shift0 = np.array([0,0,10])  #14-58-51  i==1

    if i in [1, 3]: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,-1,0,0.5]; Root_Shift0 = np.array([5,0,-5])  #15-09-09   i==6

    if i == 3: Hand_Rotate0 = [-0.65,-.3,0.3,1.5]; Root_Shift0 = np.array([0,0,0])

  if date == '06-15' and unit == 'unit3':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    ArmNParas = [200000,361]#491
    function_opt = 'constantMesh' 
    # neural_shift = -8

    if i==0: #16-35-35
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,-1,0,1]; Root_Shift0 = np.array([5,0,-5])  
      Shift_Time1 = np.dot([0,8.1],30); Root_Shift1 = np.array([0,0,-150])        
      Other_Shifts = [[Shift_Time1,Root_Shift1]]  
    if i==1: #16-37-23
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.9,0.2,1.9]; Root_Shift0 = np.array([5,0,-5])  

  if date == '06-15' and unit == 'unit4':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    ArmNParas = [200000,24]#120
    function_opt = 'constantMesh'

    if i%2==0: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [-0.5,-0.2,-0.5,1]; Root_Shift0 = np.array([5,15,-5]) 
    if i%2==1: #16-53-30
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.3,0.2,1.7]; Root_Shift0 = np.array([-20,25,-10])  
    if i == 0: #16-51-43
      Shift_Time1 = np.dot([12.83,14.5],30); Root_Shift1 = np.array([0,0,-150])        
      Other_Shifts = [[Shift_Time1,Root_Shift1]]  
    if i == 2:
      Hand_Rotate0 = [-0.5,-1,-0.5,1]; Root_Shift0 = np.array([-15,15,-5]) 

  if date == '06-22' and unit == 'unit1':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    # if i != 3: continue   

    if i%2==0: #14-52-47
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [0,-1,0,1]; Root_Shift0 = np.array([0,0,0])    
    if i%2==1: #14-54-35
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1.1,-1,0.5,1.7]; Root_Shift0 = np.array([0,0,8])
      # if i == 3:
      #   Hand_Rotate0 = [-1.2,-1,0.5,1.7]; Root_Shift0 = np.array([-15,0,5])

  if date == '06-22' and unit == 'unit2':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    ArmNParas = [20000000,100]
    function_opt = 'constantMesh'

    # if i!=2: 
    #   continue   

    if i%2==0: #16-23-10
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices1.txt'
      Hand_Rotate0 = [-1,0,-1,0.3]; Root_Shift0 = np.array([10,0,-12])
    if i%2==1: #16-24-58
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,-0.3,0,1.7]; Root_Shift0 = np.array([0,0,-10])  

  if date == '06-22' and unit == 'unit4':
    left = False
    rotate = False
    Color_idx = 3 # rb0 g1 y2 auto-1, blue on index finger 3  
    function_opt = 'constantMesh'

    # if i!=6: 
    #   continue   

    if i in [0, 2, 3, 5]: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices1.txt'
      Hand_Rotate0 = [-1,0,0,1.6]; Root_Shift0 = np.array([0,0,15])   #17-26-19 
    if i in [1, 4, 6]: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices0.txt'
      Hand_Rotate0 = [-1,-1,0.7,1]; Root_Shift0 = np.array([0,0,20])  #17-28-07

  if date == '06-14' and unit == 'unit1':
    left = False
    rotate = False
    Color_idx = 2 # r0 g1 y2 auto-1    
    ArmNParas = [200000,66]

    # if i!=2:
    #   continue

    if i==1 or i==3: 
      handMesh_Dir = handMesh_Dir_0 + 'wholeHandVertices.txt'
      Hand_Rotate0 = [0,-1,0.5,1.5]
      Root_Shift0 = np.array([50,-10,-100])   #rightup,frontdown,rightdownback      
    if i==0 or i==2: 
      handMesh_Dir = handMesh_Dir_0 + 'singleFingerVertices.txt'
      Hand_Rotate0 = [0,-1,0.5,0.6]
      Root_Shift0 = np.array([25,20,-75]) #20,20,-90

    if i == 0:
      Root_Shift0 = np.array([50,-10,-90])
      Shift_Time1 = np.dot([0,5.4],30);Root_Shift1 = np.array([50,-10,-170])   
      Shift_Time2 = np.dot([80.6,103],30);Root_Shift2 = np.array([50,-10,-60])     
      Other_Shifts = [[Shift_Time1,Root_Shift1],[Shift_Time2,Root_Shift2]]
      # neural_shift = 0.2
    if i==1:
      Color_idx = 1
      Shift_Time1 = np.dot([5,12.5],30);Root_Shift1 = np.array([50,-10,-80])
      Shift_Time2 = np.dot([0,5],30);Root_Shift2 = np.array([50,-10,-170])   
      Shift_Time3 = np.dot([80.6,103],30);Root_Shift3 = np.array([50,-10,-60])     
      Other_Shifts = [[Shift_Time1,Root_Shift1],[Shift_Time2,Root_Shift2],[Shift_Time3,Root_Shift3]]
      # neural_shift = 0.6
    if i == 2:
      Shift_Time1 = np.dot([27,120],30);Root_Shift1 = np.array([25,20,-80]) 
      Other_Shifts = [[Shift_Time1,Root_Shift1]]
    if i == 3: 
      Root_Shift0 = np.array([50,0,-50])
      Shift_Time1 = np.dot([30,120],30);Root_Shift1 = np.array([25,-10,-25]) 
      Shift_Time2 = np.dot([0,9],30);Root_Shift2 = np.array([25,10,-60]) 
      Other_Shifts = [[Shift_Time1,Root_Shift1],[Shift_Time2,Root_Shift2]]

  return Root_Shift0,Other_Shifts,Hand_Rotate0,Color_idx,left, handMesh_Dir, neural_shift, False




def run_contact_tracking(date, unit):

  Shift_Time1 = np.dot([0,0,0,0,0,0,0,0],30)  
  Root_Shift1 = np.array([0,0,0])
  Shift_Time2 = np.dot([0,0,0,0],30)  
  Root_Shift2 = np.array([0,0,0])   
  Shift_Time3 = np.dot([0,0],30)  
  Root_Shift3 = np.array([0,0,0])        
  Other_Shifts = [[Shift_Time1,Root_Shift1],[Shift_Time2,Root_Shift2],[Shift_Time3,Root_Shift3]]  
  Rotate_Time1 =  np.array([np.dot([0,0],30),np.dot([0,0],30)])   
  Hand_Rotate1 = [0,0,1,0.1]
  video_start = int(0*30)
  rotate = False
  mask=[slice(0,0),slice(0,0),slice(0,0),slice(0,0)]
  ArmNParas = [2,20]
  function_opt = 'test'
 
  dir_0, dir_1, dir_11, ply_name, N_Dir, handMesh_Dir_0 = find_dir_names(date, unit)
  ArmPLY_Dir = dir_0+'ArmPLY/'+dir_1+ply_name+'-arm_only.ply' 
  Video_Dir_0 = dir_0+'videos_mp4/'+dir_1
  all_arm_p_idx = []

  for i in range(len(os.listdir(Video_Dir_0))):
    name = os.listdir(Video_Dir_0)[i]
    print(name[:-4])
    Video_Dir = dir_0+'videos_mp4/'+dir_1+name[:-4]+'.mp4'
    CorrectedJointsColor_Dir = dir_0+'CorrectedJoints/'+dir_1+name[:-4]+'-CorrectedJoints_color.csv'
    if not os.path.exists(dir_0+'contact_quantities_RF/'+dir_1):
      os.makedirs(dir_0+'contact_quantities_RF/'+dir_1)
    ContactQuantities_Dir = dir_0+'contact_quantities_RF/'+dir_1+name[:-4]+'-ContQuant'

    Root_Shift0,Other_Shifts,Hand_Rotate0,Color_idx,left,handMesh_Dir,neural_shift,flag = find_parameters(date, unit, i, handMesh_Dir_0)
    if flag: continue

    neural_dir_0 = dir_0 + 'synched_neural_video_data/' + dir_11
    df_neural = pd.read_csv(neural_dir_0+os.listdir(neural_dir_0)[i])
    df_neural = df_neural[['t', 'Nerve_freq']]

    arm_p_idx = Receptive_field_track(Video_Dir,ArmPLY_Dir,CorrectedJointsColor_Dir,
                Root_Shift0,Other_Shifts,Hand_Rotate0,Color_idx,ArmNParas,N_Dir,left,video_start,
                handMesh_Dir,ContactQuantities_Dir, df_neural)
    all_arm_p_idx = all_arm_p_idx + arm_p_idx

    with open(dir_0+'ArmPLY/'+dir_1+'/all_contact_points_'+str(i)+'.txt', 'w') as f:
      f.write("\n".join([str(i) for i in all_arm_p_idx]))



def vis_all_RF(date, unit, video_idx, iff_vis_thre, left=False):

  dir_0, dir_1, dir_11, ply_name, N_Dir, handMesh_Dir_0 = find_dir_names(date, unit)
  ArmPLY_dir = dir_0+'ArmPLY/'+dir_1+ply_name+'-arm_only.ply' 
  contact_dir_0 = dir_0 + 'contact_quantities_RF/' + dir_1

  i = 0
  dict_contact = {}
  for name in os.listdir(contact_dir_0):
    if name[-7:-4] == 'All':
      contact_dir = contact_dir_0 + name
      df_contact_i = pd.read_csv(contact_dir)
      dict_contact[i] = df_contact_i
      i += 1

  p_idx_all = []
  for i in video_idx:
    df_contact_i = dict_contact[i]
    rf_data = df_contact_i.loc[df_contact_i['IFF'] >= iff_vis_thre, 'arm_points'].values
    print(rf_data)
    for rf_temp in rf_data:
      p_idx_all += literal_eval(rf_temp)
  print(p_idx_all)

  c = Counter(p_idx_all)
  p_idx, p_num = c.keys(), c.values()
  p_num_max = max(p_num)

  ############ output visualization ############
  window_width = 1000
  arm_pcd = o3d.io.read_point_cloud(ArmPLY_dir)
  if left == True:
    arm_points = np.asarray(arm_pcd.points)
    arm_points_left = []
    for point in arm_points:
      point[0] = -point[0]
      arm_points_left.append(point)
    arm_pcd.points = o3d.utility.Vector3dVector(np.array(arm_points_left))
    arm_points = np.asarray(arm_pcd.points)
  # viewer.add_geometry(arm_pcd)

  arm_rf_pcd = o3d.geometry.PointCloud()
  arm_rf = []
  arm_rf_color = []
  for i, n in zip(p_idx, p_num):
    if n/p_num_max > 0.05:
      arm_rf.append(arm_pcd.points[int(i)])
      arm_rf_color.append([0, 0, n/p_num_max])
      # print(arm_pcd.points[int(i)], n/p_num_max)
  arm_rf_pcd.points = o3d.utility.Vector3dVector(np.array(arm_rf))
  arm_rf_pcd.colors = o3d.utility.Vector3dVector(np.array(arm_rf_color))

  # viewer.add_geometry(arm_rf_pcd)
  o3d.visualization.draw_geometries([arm_rf_pcd, arm_pcd])

def plot_iff_depth(date, unit):
  dir_0, dir_1, dir_11, ply_name, N_Dir, handMesh_Dir_0 = find_dir_names(date, unit)
  Video_Dir_0 = dir_0+'videos_mp4/'+dir_1
  for i in range(len(os.listdir(Video_Dir_0))):
    name = os.listdir(Video_Dir_0)[i]
    print(name[:-4])
    ContactQuantities_dir = dir_0+'contact_quantities_RF/'+dir_1+name[:-4]+'-ContQuantAll.csv'
    df_contact = pd.read_csv(ContactQuantities_dir)
    t = list(range(len(df_contact.index)))

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col')
    fig.suptitle(name[:-4])  
    axes[0].plot(t, df_contact['Depth'])
    axes[1].plot(t, df_contact['IFF'])
    sns.despine(trim=True)
    plt.tight_layout()
  plt.show()


def set_parameters(dir_1):
  contactThre = [100,1,100,100,50,100]

  if dir_1 == '2022-06-17/unit2/':
    contactThre = [100,1,100,100,50,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
    # smoothWinSize01 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 9, (6,7,10,11): 33}
    # smoothWinSize02 = {(0,1,2,3,12,13,14,15): 5, (4,5,8,9): 5, (6,7,10,11): 11}    
    smoothWinSize01 = {(0,1,2,3,12,13,14,15): 151, (4,5,8,9): 301, (6,7,10,11): 1001}
    smoothWinSize02 = {(0,1,2,3,12,13,14,15): 151, (4,5,8,9): 151, (6,7,10,11): 301}

  if dir_1 == '2022-06-17/unit5/':
    contactThre = [100,0.5,125,100,55,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3, (8,9,10,11):0.7, (12,13,14,15):1.2}
    smoothWinSize01 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 301, (6,7,14,15): 1001}
    smoothWinSize02 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 151, (6,7,14,15): 301}

  if dir_1 == '2022-06-22/unit1/':
    contactThre = [100,1,100,100,55,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.6, (8,9,10,11):0.9, (12,13,14,15):1.2}
    smoothWinSize01 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 301, (6,7,14,15): 1001}
    smoothWinSize02 = {(0,1,2,3,8,9,10,11): 151, (4,5,12,13): 151, (6,7,14,15): 301}

  if dir_1 == '2022-06-14/unit1/':
    contactThre = [100,1,100,40,50,100]  # default: a_thre=100,d_thre=1,vabs_thre=200,vv_thre=100,vlt_thre=100,vlg_thre=100
    pulse_duration0 = {(0,1,2,3):0}
    smoothWinSize01 = {(0,1): 151, (2,3): 301}
    smoothWinSize02 = {(0,1): 151, (2,3): 151}

  if dir_1 == '2022-06-14/unit3/':
    pulse_duration0 = {(0,1):0, (2,3):0.5, (4,5):0.3, (6,7,8,9):0.9}
    smoothWinSize01 = {(2,3,4,5): 151, (0,1,6,7): 301, (8,9): 1001}
    smoothWinSize02 = {(2,3,4,5): 151, (0,1,6,7): 151, (8,9): 301}

  if dir_1 == '2022-06-14/unit2/':
    pulse_duration0 = {(0,1,2,3):0, (4,5,6,7):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3): 301, (4,5,6,7): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3): 151, (4,5,6,7): 301}

  if dir_1 == '2022-06-15/unit1/':
    pulse_duration0 = {(0,1):0, (2,2):-0.3, (3,4,5,6):0.3, (7,8):0.8}
    smoothWinSize01 = {(0,1,7): 151, (2,3,4,8): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1,7): 151, (2,3,4,8): 151, (5,6): 301}

  if dir_1 == '2022-06-15/unit2/':
    pulse_duration0 = {(0,1,2,3):0, (4,5):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3,4): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3,4): 151, (5,6): 301}

  if dir_1 == '2022-06-15/unit3/' or dir_1 == '2022-06-16/unit2/':
    pulse_duration0 = {(0,1):0}
    smoothWinSize01 = {(0,1): 151}
    smoothWinSize02 = {(0,1): 151}

  if dir_1 == '2022-06-16/unit1/' or dir_1 == '2022-06-17/unit4/':
    pulse_duration0 = {(0,1,2,3):0}
    smoothWinSize01 = {(0,1): 301, (2,3): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3): 301}

  if dir_1 == '2022-06-17/unit3/':
    pulse_duration0 = {(0,1):0, (2,3,4):0.3}
    smoothWinSize01 = {(0,0): 151, (1,2,3): 301, (4,4): 1001}
    smoothWinSize02 = {(0,0): 151, (1,2,3): 151, (4,4): 301}

  if dir_1 == '2022-06-15/unit4/':
    pulse_duration0 = {(0,0):-2.1, (1,2):0}
    smoothWinSize01 = {(0,1): 151, (2,2): 301}
    smoothWinSize02 = {(0,1): 151, (2,2): 151}

  if dir_1 == '2022-06-22/unit4/':
    pulse_duration0 = {(0,1,2):0, (3,4,5,6):0.3}
    smoothWinSize01 = {(0,1): 151, (2,3,4): 301, (5,6): 1001}
    smoothWinSize02 = {(0,1): 151, (2,3,4): 151, (5,6): 301}

  if dir_1 == '2022-06-22/unit2/':  
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

def fillNaN(data):
  t = np.arange(len(data))
  good = ~np.isnan(data)
  f = interp1d(t[good], data[good], bounds_error=False)
  filledData = np.where(np.isnan(data), f(t), data)
  filledData[np.isnan(filledData)] = 0
  return filledData

def combine_all_trials_semi_control_new(date, unit,split_trial=False,new_axes=False):

  dir_0, dir_1, dir_11, _, _, _ = find_dir_names(date, unit)

  Output_Dir = dir_0+'combined_data/manual_axes_3Dposition_RF/'
  video_Dir_0 = dir_0+'videos_mp4/'+dir_1  
  neural_Dir_0 = dir_0+'synched_neural_video_data/'+dir_11   
  stimuli_Dir_0 = dir_0+'stimuli_log/'+dir_1   
  cq_Dir_0 = dir_0+'contact_quantities_RF/'+dir_1
  position_3d_Dir_0 = dir_0+'contact_quantities_RF/'+dir_1

  combine_df = pd.DataFrame()
  t_start = 0.0

  pulse_duration, smoothWinSize1, smoothWinSize2, cThre = set_parameters(dir_1)

  for i in range(len(os.listdir(video_Dir_0))):  # iterate through all file
    name = os.listdir(video_Dir_0)[i]
    print(name[:-4])
    ContactQuantities_dir = cq_Dir_0+name[:-4]+'-ContQuant.csv'
    position_3d_dir = position_3d_Dir_0+name[:-4]+'-ContQuantAll.csv'
    i_neural = os.listdir(neural_Dir_0)[i]
    NeuralData_dir = neural_Dir_0+i_neural
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
    # print(len(t_spike), sum(np.array(f_plot) != 0))

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
    os.chdir(stimuli_Dir_0)
    for i_stimuli in os.listdir():
      if i_stimuli.split('_')[-1] != 'log.csv': continue
      stimuli_name = i_stimuli[:16]
      video_name = name[5:][:16]
      print(stimuli_name, video_name)
      if stimuli_name == video_name:
        print('yes')
        stimuli_dir = stimuli_Dir_0+i_stimuli
        stimuli_log_df = pd.read_csv(stimuli_dir, names=['time', 'event', 'vel', 'finger', 'force'], header=None) #, usecols=[0,1]
        stimuli_log_df = stimuli_log_df.iloc[1:].reset_index()
        stimuli_log_df['time'] = stimuli_log_df['time'].apply(pd.to_numeric)
        print(stimuli_log_df)
    os.chdir(video_Dir_0)

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
        trial_temp_idx = (dir_1, trial_df['block_id'].values[0], trial_id)
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
      path = dir_0+'contact_IFF_trial/'+dir_1
      isExist = os.path.exists(path)
      if not isExist:
         os.makedirs(path)
      trial_df.to_csv(path+name[5:-4]+'.csv', index = False, header=True)

  if not split_trial:
    combine_df.to_csv(Output_Dir+dir_1[:10]+'-'+name[46:50]+'-'+dir_1[11:-1]+'-semicontrol.csv', index = False, header=True)

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
  #   fig.savefig(dir_0+'plots/'+dir_1+dir_plot+str(ai)+'.png')

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
  fig.savefig(dir_0+'plots/'+dir_1+dir_plot+'.png')
  # plt.show()




if __name__ == '__main__':

  date = '06-17'; unit = 'unit2'
  # video_idx = [0, 2, 4, 6, 8, 10, 12, 14]
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-14'; unit = 'unit3'
  # video_idx = [0, 2, 4, 6, 8]
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-14'; unit = 'unit2'
  video_idx = [0, 2, 4, 6]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-17'; unit = 'unit5'
  # video_idx = [0, 2, 4, 6, 8, 10, 12, 14]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-16'; unit = 'unit1'
  # video_idx = [0, 2]
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)  
  combine_all_trials_semi_control_new(date,unit)

  date = '06-16'; unit = 'unit2'
  # video_idx = [0]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-17'; unit = 'unit3'
  # video_idx = [0, 1, 2, 4]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-17'; unit = 'unit4'
  # video_idx = [0]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-15'; unit = 'unit1'
  # video_idx = [0, 2, 4, 6]
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)  
  combine_all_trials_semi_control_new(date,unit)

  date = '06-15'; unit = 'unit2'
  # video_idx = [0, 2, 4, 5]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-15'; unit = 'unit3'
  # video_idx = [0]
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)  
  combine_all_trials_semi_control_new(date,unit)

  date = '06-15'; unit = 'unit4'
  # video_idx = [0, 2]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-22'; unit = 'unit1'
  # video_idx = [0, 2, 4, 6, 8, 10, 12, 14]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-22'; unit = 'unit2'
  # video_idx = [0, 2]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-22'; unit = 'unit4'
  # video_idx = [0, 2, 3, 5]
  # run_contact_tracking(date, unit)
  combine_all_trials_semi_control_new(date,unit)

  date = '06-14'; unit = 'unit1'
  # run_contact_tracking(date, unit)
  # plot_iff_depth(date, unit)  
  combine_all_trials_semi_control_new(date,unit)

  # video_idx = list(range(10))
  # # video_idx = [2]
  # vis_all_RF(date, unit, video_idx, iff_vis_thre=5)



  # plt.show()


