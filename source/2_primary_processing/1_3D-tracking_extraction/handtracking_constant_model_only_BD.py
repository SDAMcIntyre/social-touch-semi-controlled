import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
from sympy import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'imported'))
from imported.hand_mesh import HandMesh
from imported.utils import imresize  # OneEuroFilter, imresize
from imported.kinematics import mpii_to_mano
from imported.wrappers import ModelPipeline
import imported.config as config



def create_mockup_dataframe_row(time, frame_idx, is_capture_failed=False):
    contact_quantities = {
        "contact_detected": False,
        "area": np.nan,
        "depth": np.nan,
        "location": [np.nan, np.nan, np.nan],
        "contact_arm_pointcloud": np.nan
    }
    palm_position_xyz = [np.nan, np.nan, np.nan]
    index_position_xyz = [np.nan, np.nan, np.nan]
    velocities_avLl = [np.nan, np.nan, np.nan, np.nan]
    return create_row(time, frame_idx, contact_quantities, palm_position_xyz, index_position_xyz, velocities_avLl, is_capture_failed)



def create_row(time, frame_idx, contact_quantities, palm_position_xyz, index_position_xyz, velocities_avLl, is_capture_failed=False):
    return {
        "time": time,
        "frame_id": frame_idx,
        "capture_read_failed": is_capture_failed,

        "contact_detected": contact_quantities["contact_detected"],
        "contact_area": contact_quantities["area"],
        "contact_depth": contact_quantities["depth"],
        "contact_arm_pointcloud": contact_quantities["contact_arm_pointcloud"],
        
        "contact_position_x": contact_quantities["location"][0],
        "contact_position_y": contact_quantities["location"][1],
        "contact_position_z": contact_quantities["location"][2],
        
        "palm_position_x": palm_position_xyz[0],
        "palm_position_y": palm_position_xyz[1],
        "palm_position_z": palm_position_xyz[2],
        
        "index_position_x": index_position_xyz[0],
        "index_position_y": index_position_xyz[1],
        "index_position_z": index_position_xyz[2],

        "hand_velocity_abs": velocities_avLl[0],
        "hand_velocity_vertical": velocities_avLl[1],
        "hand_velocity_lateral": velocities_avLl[2],
        "hand_velocity_longitudinal": velocities_avLl[3],
        }



def Contact_quantities_ConstantHandGesture(video_dir,ArmPLY_dir,CorrectedJointsColor_dir,
  root_shift0,Other_Shifts,hand_rotate0,
  color_idx,ArmNParas,N_dir,left,handMesh_Dir, show=True, show_result=True):
  """
  calibration using colored finger nail and 3d calibrated root joint position
  """
  capture = cv2.VideoCapture(video_dir)
  capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
  if not capture.isOpened():
    print(f"video cannot be be opened: {video_dir}")
    return None
  capture_nframe = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
  #palm_vertex_idx = np.where((hand_mesh.verts[:,0]>-0.045)&(hand_mesh.verts[:,0]<-0.035)&
  #           (hand_mesh.verts[:,2]>-0.01)&(hand_mesh.verts[:,2]<0.01)&(hand_mesh.verts[:,1]>0)) # palm green point
  # vertex of the hand mesh locating the palm and index (used for hand XYZ position estimation)
  palm_vertex_idx = np.where((-0.045 < hand_mesh.verts[:,0]) & (hand_mesh.verts[:,0] < -0.035)   # x value of vertices
                                      &( 0    < hand_mesh.verts[:,1])                                     # y value of vertices
                                      &(-0.01 < hand_mesh.verts[:,2]) & (hand_mesh.verts[:,2] < 0.01))    # z value of vertices
  index_vertex_idx = np.where((0.065 < hand_mesh.verts[:,0]) & (0.02 < hand_mesh.verts[:,2]))  # index finger tip
  
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
  arm_intesect_pcd = o3d.geometry.PointCloud()
  arm_intesect_pcd.paint_uniform_color([1, 0, 0])
  points = [arm_points[int(n_arm_points/ArmNParas[0])+ArmNParas[1]],
            arm_points[int(n_arm_points/ArmNParas[0])+ArmNParas[1]]+150*arm_Normal]
  lines=[[0,1]]
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(points)
  line_set.lines = o3d.utility.Vector2iVector(lines)

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
  v_handMesh += np.abs(np.min(v_handMesh, axis=0))
  if left == False:
    v_handMesh[:, 1] *= -1
    v_handMesh[:, 2] *= -1
    #v_handMesh[:,1] = -v_handMesh[:,1]
    #v_handMesh[:,0] = -v_handMesh[:,0]

  ########## Contact quantities ############
  # output dataframe
  columns = [
      'time', 'frame_id', 'capture_read_failed', 
      'contact_detected', 'contact_arm_pointcloud', 'contact_area', 'contact_depth', 
      'contact_position_x',   'contact_position_y',   'contact_position_z', 
      'palm_position_x',      'palm_position_y',      'palm_position_z', 
      'index_position_x',     'index_position_y',     'index_position_z', 
      'hand_velocity_abs', 'hand_velocity_vertical', 'hand_velocity_lateral', 'hand_velocity_longitudinal'
  ]
  df = pd.DataFrame(columns=columns)

  ############ input and model visualization ############
  if show:
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
      width=window_width + 1, height=window_width + 1,
      window_name='Minimal Hand - output'
    )
    viewer.add_geometry(mesh)
    viewer.add_geometry(marker_sphere)
    viewer.add_geometry(arm_pcd)
    viewer.add_geometry(arm_intesect_pcd)
    viewer.add_geometry(line_set)

    view_control = viewer.get_view_control()
    view_control.rotate(0,1000)
    view_control.scale(.001)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./render_option.json')
    viewer.update_renderer()
    pygame.init()
    display = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Minimal Hand - input')
    clock = pygame.time.Clock()
    # Introduce a paused state, initially False
    paused = False
    running = True
  ############ somatosensory result visualization ############
  if show_result:
    fig, ax = plt.subplots()
    fig.set_size_inches(8,5)
    plt.ion()

  end_of_video = False
  first_frame = True
  palm_position_xyz_prev = 0

  while not end_of_video:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
          running = False
      if event.type == pygame.KEYDOWN:
          # Toggle pause mode when SPACE is pressed
          if event.key == pygame.K_SPACE:
              paused = not paused
              if paused:
                  print("--- Paused: You can now rotate the geometry manually. Press SPACE to resume. ---")
              else:
                  print("--- Resumed ---")
    
    # I. load current frame
    if not paused:
      if first_frame:
        time = 0
        frame_idx = 0
      else:
        time += 1./30  # Azure Kinect Fs is 30 Hz
        frame_idx += 1
      has_frame, frame = capture.read()
      if not has_frame:
          if capture.get(cv2.CAP_PROP_POS_FRAMES) >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
              print("End of video file reached.")
              end_of_video = True
              continue
          else:
              # Append the new row to the output DataFrame using loc
              df.loc[len(df)] = create_mockup_dataframe_row(time, frame_idx, is_capture_failed=True)
              print(f"frame {frame_idx}: capture read failed.")
              has_frame = True
              continue
    
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
    print(f'---------------------------------------')   
    print(f'{os.path.basename(video_dir)}')
    print(f'Frame: {str(frame_idx).zfill(len(str(capture_nframe)))}/{capture_nframe}')
    print(f'color_idx = {color_idx}, root_shift = {root_shift}')   

    v = np.matmul(view_mat0, v_handMesh.T).T    
    v[:,0] = -v[:,0]
    red_p = corrected_joints_color[frame_idx,0:3]
    green_p = corrected_joints_color[frame_idx,3:6]
    yellow_p = corrected_joints_color[frame_idx,6:9]
    # root_p = corrected_joints[frame_idx]  
    fingernail_p = np.mean(v[index_idx[0],:],axis=0)
    palm_p = np.mean(v[palm_vertex_idx[0],:],axis=0)
    #### calibration using root & color 
    v0 = v
    # v = v0 * 1000 + root_p + root_shift.T ;p_temp = root_p
    if color_idx==0: p_temp=red_p
    if color_idx==1: p_temp=green_p
    if color_idx==2: p_temp=yellow_p
    if color_idx==3: 
      p_temp=red_p
      v = v0 * 1000 + p_temp - fingernail_p*1000 + root_shift.T  
    else:  
      v = v0 * 1000 + p_temp - palm_p*1000 + root_shift.T
    # v = mesh_smoother.process(v)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    v_no_thumb = np.delete(v, thumb_idx[0], axis=0)
    marker_sphere.translate(p_temp, relative=False)
    
    # IV. Calculate contact quantites
    hand_inside = []
    arm_intersect = []
    hand_arm_dis = []
    arm_norm = []
    for i_hand in range(v_no_thumb.shape[0]):
      hand_arm_vector = v_no_thumb[i_hand]-arm_pcd.points
      hand_arm_min_idx = np.argmin(np.sum(np.abs(hand_arm_vector)**2,axis=-1))
      if np.dot(hand_arm_vector[hand_arm_min_idx],arm_pcd.normals[hand_arm_min_idx]) <= 0:
        arm_point_color = arm_pcd.colors[hand_arm_min_idx]
        if arm_point_color[0] > 0.9 and arm_point_color[1] > 0.9 and arm_point_color[2] > 0.9: # for hnad mesh with missing parts
          continue 
        hand_inside.append(v_no_thumb[i_hand])
        arm_intersect.append(arm_pcd.points[hand_arm_min_idx])
        hand_arm_dis.append(np.linalg.norm(hand_arm_vector[hand_arm_min_idx]))
        arm_norm.append(arm_pcd.normals[hand_arm_min_idx])
    arm_Longitudinal = [0,1,0]-np.dot([0,1,0],arm_Normal)*arm_Normal #same direction as camera y axis
    arm_Longitudinal = arm_Longitudinal/np.linalg.norm(arm_Longitudinal)
    arm_Lateral = np.cross(arm_Longitudinal,arm_Normal) # opposite direction of camera x axis
    arm_Lateral = arm_Lateral/np.linalg.norm(arm_Lateral)   
           
    # (1/2) tactile data
    if np.array(arm_intersect).size == 0:
      contact_quantities = {
            "contact_detected": 0,
            "depth": 0,
            "area": 0,
            "location": (np.nan, np.nan, np.nan),
            "contact_arm_pointcloud": []
            }
      arm_intesect_pcd.clear()
    else:
      arm_intersect_uniq = np.unique(np.array(arm_intersect), axis=0)
      contact_quantities = {
            "contact_detected": 1,
            "depth": np.mean(hand_arm_dis),
            "area": arm_intersect_uniq.shape[0]*arm_point_dis**2,
            "location": tuple(np.mean(arm_intersect_uniq, axis=0)),
            "contact_arm_pointcloud": arm_intersect_uniq
            }
      arm_intesect_pcd.points = o3d.utility.Vector3dVector(np.array(arm_intersect))
      arm_intesect_pcd.paint_uniform_color([1, 0, 0])
      print(np.array(arm_intersect).shape)
      print(arm_intersect_uniq.shape)

    # (2/2) proprioceptive data
    palm_position_xyz = np.mean(v[palm_vertex_idx[0], :], axis=0)
    index_position_xyz = np.mean(v[index_vertex_idx[0], :], axis=0)
    if first_frame:
      vel_i = 0
      velocities_avLl = [0, 0, 0, 0]
    else:
      vel_i = (palm_position_xyz - palm_position_xyz_prev) / (1./30)
      velocities_avLl  = [np.linalg.norm(vel_i), np.dot(vel_i, arm_Normal), np.dot(vel_i, arm_Lateral), np.dot(vel_i, arm_Longitudinal)]  # absolute, vertical, Lateral, longitudinal

    # V. Append the new data to the output DataFrame using loc
    df_frame = create_row(time, frame_idx, contact_quantities, palm_position_xyz, index_position_xyz, velocities_avLl)
    df.loc[len(df)] = df_frame

    # VI. routine before loading next frame
    palm_position_xyz_prev = palm_position_xyz
    if first_frame:
      first_frame = False

    ########### visualization ############################
    if show_result and False:
      # contact quantities visualization
      contact_quant_plot = {'cont_flag': [df_frame["contact_detected"]*50],
                            'cont_area': [df_frame["contact_area"]/100.],
                            'depth':     [df_frame["contact_depth"]/10.],
                            'vel_abs':   [df_frame["hand_velocity_abs"]/10.],
                            'vel_vert':  [df_frame["hand_velocity_vertical"]/10.],
                            'vel_late':  [df_frame["hand_velocity_lateral"]/10.],
                            'vel_longi': [df_frame["hand_velocity_longitudinal"]/10.]
                            }
      df_plot = pd.DataFrame(data=contact_quant_plot)    
      sns.set(style="whitegrid", font="Arial", font_scale=3)
      clrs = ['grey','darkseagreen','darkseagreen','darkseagreen','darkseagreen','darkseagreen','darkseagreen']
      ax = sns.barplot(data=df_plot, order=['cont_flag', 'cont_area','depth','vel_abs','vel_vert','vel_late','vel_longi'],palette=clrs)
      ax.set_ylim(-50,50)
      ax.set_xticklabels(('F','A','D','V','Vvt','Vlt','Vlg'))
      ax.set_title(f'{os.path.basename(video_dir)}\nFrame: {str(frame_idx).zfill(len(str(capture_nframe)))}/{capture_nframe}')
      # ax.set_ylabel('cm')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.spines['left'].set_visible(False)
      plt.subplots_adjust(bottom=.14,left=.15,top=.88)
      plt.draw()
      plt.pause(0.00000001)
      plt.clf()

    if show:
      # for some version of open3d you may need `viewer.update_geometry(mesh)`
      viewer.update_geometry(mesh)
      viewer.poll_events()
      # update the processed frame
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
      clock.tick(30)


  print(frame_idx)
  capture.release()
  cv2.destroyAllWindows()
  plt.close('all')

  # ignore the velocity columns (as the position XYZ and velocities have two different reference systems {kinect camera or arm orientation})
  columns_to_remove = ['hand_velocity_abs', 'hand_velocity_vertical', 'hand_velocity_lateral', 'hand_velocity_longitudinal']
  df = df.drop(columns=columns_to_remove)
  return df

