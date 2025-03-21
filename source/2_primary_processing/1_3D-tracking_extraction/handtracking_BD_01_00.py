import copy
import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pandas as pd
import pygame
import seaborn as sns
from sympy import *
import sys

from transforms3d.axangles import axangle2mat

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'imported'))
from imported.hand_mesh import HandMesh
from imported.utils import *  # OneEuroFilter, imresize
from imported.kinematics import mpii_to_mano
from imported.wrappers import ModelPipeline
import imported.config as config

# Function to calculate the center of a geometry
def calculate_center(geometry):
    bbox = geometry.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    return center


def create_hand_mesh(hand_mesh_model_path, hand_rotate = [0, 1, 0, np.pi/8.]):
    view_matrix = axangle2mat(hand_rotate[:3], hand_rotate[3])
    hand_mesh = HandMesh(hand_mesh_model_path)
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_matrix, hand_mesh.verts.T).T * 1000)
    mesh.compute_vertex_normals()
    return mesh, hand_mesh

def create_arm_point_cloud(arm_ply_path, left):
    arm_pointcloud = o3d.io.read_point_cloud(arm_ply_path)
    if left:
        arm_points = np.asarray(arm_pointcloud.points)
        arm_points[:, 0] = -arm_points[:, 0]
        arm_pointcloud.points = o3d.utility.Vector3dVector(arm_points)
    arm_pointcloud.estimate_normals()
    arm_pointcloud.orient_normals_to_align_with_direction([0., 0., -1.])
    return arm_pointcloud

def extract_handstickers(handStickersLocs_fname_abs, left, filter=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(handStickersLocs_fname_abs)
    
    # remove any leading and trailing spaces from the column names. 
    df.columns = df.columns.str.strip()

    # Reduce the DataFrame to only the columns of interest
    if filter:
        # Define the columns of interest
        column_of_interest = ['blue_x_mm', 'blue_y_mm', 'blue_z_mm',
                            'green_x_mm', 'green_y_mm', 'green_z_mm',
                            'yellow_x_mm', 'yellow_y_mm', 'yellow_z_mm']
        df = df[column_of_interest]

    if left:
        df['blue_x_mm'] = -df['blue_x_mm']
        df['green_x_mm'] = -df['green_x_mm']
        df['yellow_x_mm'] = -df['yellow_x_mm']
        df['blue_pixel_x'] = 1920-df['blue_pixel_x']
        df['green_pixel_x'] = 1920-df['green_pixel_x']
        df['yellow_pixel_x'] = 1920-df['yellow_pixel_x']

    return df


def display_init(title='input'):
    pygame.init()
    display = pygame.display.set_mode((500, 500))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    return display

def display_update(display, frame, pointsuv=None, connections=None):
    # frame = np.flip(frame_show, -1).copy()  # BGR to RGB
    
    # Resize the frame to 500x500 to match the pygame window size
    frame = imresize(frame, (500, 500))

    # Plot red dots for each XY value in pointsuv
    if pointsuv is not None:
        # scale the XY points to the image (they have been assessed on a 32x32 image)
        scale_x = 500 / 32
        scale_y = 500 / 32
        pointsuv = (pointsuv * np.array([scale_x, scale_y])).astype(int)

        for point in pointsuv:
            y, x = point
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Change pixels within a diameter of 5 pixels around (x, y) to red
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < frame.shape[1] and 0 <= ny < frame.shape[0]:
                            frame[ny, nx] = [255, 0, 0]  # Red color in RGB

        for connection in connections:
            pointv0, pointu0 = pointsuv[connection[0]]
            pointv1, pointu1 = pointsuv[connection[1]]
            # Draw line by directly modifying pixels
            x0, y0 = int(pointu0), int(pointv0)
            x1, y1 = int(pointu1), int(pointv1)
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                if 0 <= x0 < frame.shape[1] and 0 <= y0 < frame.shape[0]:
                    frame[y0, x0] = [0, 255, 0]  # Set pixel to connection color
                if x0 == x1 and y0 == y1:
                    break
                e2 = err * 2
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

    display.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))
    pygame.display.update()


def viewer3D_init(mesh, arm_pointcloud, marker_sphere, ArmNParas=[2,2], title='output', window_width=1000):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=window_width + 1, height=window_width + 1, window_name=title)
    viewer.add_geometry(mesh)
    viewer.add_geometry(marker_sphere)
    viewer.add_geometry(arm_pointcloud)
    
    arm_intesect_pcd = o3d.geometry.PointCloud()
    arm_intesect_pcd.paint_uniform_color([1, 0, 0])
    viewer.add_geometry(arm_intesect_pcd)
    
    arm_points = np.asarray(arm_pointcloud.points)
    n_arm_points = arm_points.shape[0]
    points = [arm_points[int(n_arm_points / ArmNParas[0]) + ArmNParas[1]],
    arm_points[int(n_arm_points / ArmNParas[0]) + ArmNParas[1]]]
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    viewer.add_geometry(line_set)
    
    # Update the view control to center on the passive skin
    view_control = viewer.get_view_control()
    view_control.rotate(0, 1000)
    view_control.scale(.001)
    
    render_option = viewer.get_render_option()
    json_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_option.json')
    render_option.load_from_json(json_path_abs)

    viewer.update_renderer()

    print(f"-------------------------------")
    print(f"Center of mesh: {calculate_center(mesh)}")
    print(f"Center of marker_sphere: {calculate_center(marker_sphere)}")
    print(f"Center of arm_pointcloud: {calculate_center(arm_pointcloud)}")
    print(f"Center of arm_intesect_pcd: {calculate_center(arm_intesect_pcd)}")
    print(f"Center of line_set: {calculate_center(line_set)}")
    print(f"-------------------------------")


    return viewer

def viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, marker_sphere=None, hand_adj_vertex_value=None, arm_pointcloud=None, verbose=False):
    # if verbose: print(f"-------------------------------")
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(hand_vertices)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    if verbose: print(f"Center of mesh: {calculate_center(mesh)}")

    if marker_sphere is not None:
        marker_sphere.translate(hand_adj_vertex_value, relative=False)
        viewer.update_geometry(marker_sphere)   
        if verbose: print(f"Center of marker_sphere: {calculate_center(marker_sphere)}")
    
    if arm_pointcloud is not None and verbose:
        print(f"Center of arm_pointcloud: {calculate_center(arm_pointcloud)}")
    
    if verbose: print(f"-------------------------------")
    viewer.poll_events()

def create_marker_sphere(radius=5, color=[0, 0, 1]):
    marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker_sphere.paint_uniform_color(color)
    return marker_sphere


def overwrite_shift(frame_idx, hardcoded_shift, specific_sections_shift):
    for shift_param in specific_sections_shift:
        if shift_param[0].size > 2:
            periods = np.split(shift_param[0], shift_param[0].size / 2)
        else:
            periods = np.array([shift_param[0]])
        for period in periods:
            if period[0] <= frame_idx <= period[1]:
                hardcoded_shift = shift_param[1]
    return hardcoded_shift


def preprocess_video_frame(frame, crop_box_center, crop_size, is_left_hand=False, is_bgr=True):
    # Ensure crop_offset and crop_size are within bounds
    crop_x_start = max(0, int(crop_box_center[0] - crop_size / 2))
    crop_x_end = min(frame.shape[1], crop_x_start + crop_size)
    crop_y_start = max(0, int(crop_box_center[1] - crop_size / 2))
    crop_y_end = min(frame.shape[0], crop_y_start + crop_size)
    frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    # BGR to RGB
    if is_bgr:
        frame = np.flip(frame, -1)  
    # flip image if the hand is a right hand (for handmesh)
    if not is_left_hand:
        frame = np.flip(frame, axis=1) 
    # reduce the image to speed up handmesh
    frame = imresize(frame, (128, 128))
    return frame


def hand_mismatch_detected(pointsuv, connections, LRatio, crop_size):
    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    # hand detection, remove mismatched hand
    flag_J5913 =    (np.dot((pointsuv[9] -pointsuv[5]), (pointsuv[13]-pointsuv[9]))) > 0
    flag_J91317 =   (np.dot((pointsuv[13]-pointsuv[9]), (pointsuv[17]-pointsuv[13]))) > 0
    flag_J059 =     (np.dot((pointsuv[5] -pointsuv[0]), (pointsuv[9] -pointsuv[0]))) > 0
    flag_J0913 =    (np.dot((pointsuv[9] -pointsuv[0]), (pointsuv[13]-pointsuv[0]))) > 0
    flag_J01317 =   (np.dot((pointsuv[13]-pointsuv[0]), (pointsuv[17]-pointsuv[0]))) > 0
    flag_J056 =     (np.dot((pointsuv[5] -pointsuv[0]), (pointsuv[6] -pointsuv[0]))) > 0
    flag_J0910 =    (np.dot((pointsuv[9] -pointsuv[0]), (pointsuv[10]-pointsuv[0]))) > 0
    flag_J01314 =   (np.dot((pointsuv[13]-pointsuv[0]), (pointsuv[14]-pointsuv[0]))) > 0
    flag_J01718 =   (np.dot((pointsuv[17]-pointsuv[0]), (pointsuv[18]-pointsuv[0]))) > 0
    flag_J012 =     (np.dot((pointsuv[2] -pointsuv[1]), (pointsuv[1] -pointsuv[0]))) > 0
    flag_J123 =     (np.dot((pointsuv[2] -pointsuv[1]), (pointsuv[3] -pointsuv[2]))) > 0
    flag_J234 =     (np.dot((pointsuv[4] -pointsuv[3]), (pointsuv[3] -pointsuv[2]))) > 0
    flag_J015 =     (np.dot((pointsuv[1] -pointsuv[0]), (pointsuv[5] -pointsuv[0]))) > 0
    flag_length = True
    for connection in connections:
      v0 = pointsuv[connection[0]][0]
      u0 = pointsuv[connection[0]][1]
      v1 = pointsuv[connection[1]][0]
      u1 = pointsuv[connection[1]][1]
      # print(np.linalg.norm(np.array([(32-u0)*LRatio,v0*LRatio])-np.array([(32-u1)*LRatio,v1*LRatio])))
      if np.linalg.norm(np.array([(32-u0)*LRatio,v0*LRatio])-np.array([(32-u1)*LRatio,v1*LRatio])) > crop_size/2:
        flag_length = False
    flag_mismatch = not(flag_J059 and flag_J0913 and flag_J01317 and #flag_J5913 and flag_J91317 and
                        flag_J056 and flag_J0910 and flag_J01314 and flag_J01718 and flag_J012 and 
                        flag_J123 and flag_J234 and flag_J015 and flag_length)
    return flag_mismatch



def update_hand_posture(hand_mesh, theta_mpii, matrix_rotation_viewer, rotate, sticker_vertex_idx, sticker_kinect_xyz_value, hardcoded_shift, viewer=None, mesh=None):
    # viewer3D_update(viewer, mesh, hand_mesh, hand_vertices)
    
    theta_mano = mpii_to_mano(theta_mpii)
    hand_vertices = hand_mesh.set_abs_quat(theta_mano)  # shape [V, 3]
    print("viewer3D of updated hand mesh:")
    viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)

    view_matrix = axangle2mat(matrix_rotation_viewer[:3], matrix_rotation_viewer[3])  # align different coordinate systems
    hand_vertices = np.matmul(view_matrix, hand_vertices.T).T
    print("viewer3D update after view_matrix:")
    viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)

    if rotate:
        view_mat_r = axangle2mat([0, 0, 1], -np.pi / 2)
        hand_vertices = np.matmul(view_mat_r, hand_vertices.T).T
        print("viewer3D update after rotate:")
        viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)
    
    #hand_vertices[:, 0] = -hand_vertices[:, 0]
    print("viewer3D update after -:")
    viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)

    hand_vertices *= 1e3  # meter to mm (hand mesh -> Kinect unit)
    print("viewer3D update after *= 1e3:")
    viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)
     
    if True:
        sticker_hand_mesh_xyz_value = np.mean(hand_vertices[sticker_vertex_idx[0], :], axis=0)
        #sticker_kinect_xyz_value /= 1e3
        hand_vertices += sticker_kinect_xyz_value - sticker_hand_mesh_xyz_value
        print("viewer3D update after sticker adjusted:")
        viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)

    hand_vertices += hardcoded_shift.T
    print("viewer3D update after hardcoded shift:")
    viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, verbose=True)

    return hand_vertices



def calculate_contact_quantities(hand_vertices, arm_point_cloud, avg_neighbor_arm_point_dist):
    hand_inside = []
    arm_intersect = []
    hand_arm_distance = []
    arm_normals = []
    # Iterate over each vertex in the hand mesh (excluding the thumb)
    for vertex in hand_vertices:
        # Calculate the vector from the current hand vertex to each point in the arm point cloud
        arm_points_with_vertex_origin = arm_point_cloud.points - vertex
        # Calculate the squared Euclidean distances (cost efficient) to each point in the arm point cloud
        eucl_distances = np.sum(np.abs(arm_points_with_vertex_origin) ** 2, axis=-1)
        # Find the index of the closest point based on the minimum squared distance
        closest_arm_point_idx = np.argmin(eucl_distances)
        
        # Check if the vertex is inside the arm mesh based on the dot product with the normal
        if np.dot(arm_points_with_vertex_origin[closest_arm_point_idx], arm_point_cloud.normals[closest_arm_point_idx]) <= 0:
            # Skip vertices that correspond to points with high color values (indicating missing parts)
            if np.all(arm_point_cloud.colors[closest_arm_point_idx] > 0.9):
                continue
            # Append the current vertex, closest arm point, and normal of the closest arm point
            hand_inside.append(vertex)
            arm_intersect.append(arm_point_cloud.points[closest_arm_point_idx])
            arm_normals.append(arm_point_cloud.normals[closest_arm_point_idx])
            # Calculate and append the distance between the hand vertex and the closest arm point
            hand_arm_distance.append(np.linalg.norm(arm_points_with_vertex_origin[closest_arm_point_idx]))

    if not arm_intersect:
        return {
            "contact_detected": 0,
            "depth": 0,
            "area": 0,
            "location": (np.nan, np.nan, np.nan)
            }
    
    arm_intersect_uniq = np.unique(np.array(arm_intersect), axis=0)
    depth_mean = np.mean(hand_arm_distance)
    cont_area_i = arm_intersect_uniq.shape[0] * avg_neighbor_arm_point_dist ** 2
    loc_xyz = tuple(np.mean(arm_intersect_uniq, axis=0))

    return {
        "contact_detected": 1,
        "depth": depth_mean,
        "area": cont_area_i,
        "location": loc_xyz
        }


def create_mockup_dataframe_row(time, frame_idx, is_capture_failed=False, is_hand_failed=False):
    return {
        "time": time,
        "frame_id": frame_idx,

        "capture_read_failed": is_capture_failed,
        "hand_posture_failed": is_hand_failed,

        "contact_detected": False,
        "contact_area": np.nan,
        "contact_depth": np.nan,
        
        "contact_position_x": np.nan,
        "contact_position_y": np.nan,
        "contact_position_z": np.nan,
        
        "palm_position_x": np.nan,
        "palm_position_y": np.nan,
        "palm_position_z": np.nan,
        
        "index_position_x": np.nan,
        "index_position_y": np.nan,
        "index_position_z": np.nan,
        }


def create_row(time, frame_idx, contact_quantities, palm_position_xyz, index_position_xyz):
    return {
        "time": time,
        "frame_id": frame_idx,

        "capture_read_failed": False,
        "hand_posture_failed": False,

        "contact_detected": contact_quantities["contact_detected"],
        "contact_area": contact_quantities["area"],
        "contact_depth": contact_quantities["depth"],
        
        "contact_position_x": contact_quantities["location"][0],
        "contact_position_y": contact_quantities["location"][1],
        "contact_position_z": contact_quantities["location"][2],
        
        "palm_position_x": palm_position_xyz[0],
        "palm_position_y": palm_position_xyz[1],
        "palm_position_z": palm_position_xyz[2],
        
        "index_position_x": index_position_xyz[0],
        "index_position_y": index_position_xyz[1],
        "index_position_z": index_position_xyz[2],
        }



def extract_touch_per_frame(kinect_capture, arm_pointcloud, stickers_locs_df, params):
    crop_position = params["video"]["crop_position"]
    crop_size = params["video"]["crop_size"]
    color_idx = params["hand_stickers"]["color_idx"]
    left = params["hand_used"]["left"]
    rotate = params["hand_mesh"]["rotate"]
    hardcoded_shift = params["hand_mesh"]["hardcoded_shift"]
    specific_sections_shift = params["hand_mesh"]["specific_sections_shift"]
    matrix_rotation_viewer = params["viewer"]["matrix_rotation_viewer"]

    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
    mesh, hand_mesh = create_hand_mesh(config.HAND_MESH_MODEL_PATH, matrix_rotation_viewer)

    # constant variables used for processing
    # Calculate the average distance to the nearest neighbor for points in the arm point cloud (useful for contact area)
    avg_dist_nnarm = np.mean(arm_pointcloud.compute_nearest_neighbor_distance())
    # constant for XYZ hand adjustement of the hand mesh using the selected hand sticker
    if color_idx == 0: 
        # blue point (fingertip)
        selected_sticker_xyzcolnames = ['blue_x_mm', 'blue_y_mm', 'blue_z_mm']
        # /!\ palm green point (NOT BLUE)
        hand_adj_vertex_idx = np.where((-0.045 < hand_mesh.verts[:,0]) & (hand_mesh.verts[:,0] < -0.035)    # x value of vertices
                                        &( 0    < hand_mesh.verts[:,1])                                     # y value of vertices
                                        &(-0.01 < hand_mesh.verts[:,2]) & (hand_mesh.verts[:,2] < 0.01))    # z value of vertices
    elif color_idx == 1: 
        # green point (palm)
        selected_sticker_xyzcolnames = ['green_x_mm', 'green_y_mm', 'green_z_mm']
        hand_adj_vertex_idx = np.where((-0.045 < hand_mesh.verts[:,0]) & (hand_mesh.verts[:,0] < -0.035)    # x value of vertices
                                        &( 0    < hand_mesh.verts[:,1])                                     # y value of vertices
                                        &(-0.01 < hand_mesh.verts[:,2]) & (hand_mesh.verts[:,2] < 0.01))    # z value of vertices
    elif color_idx == 2: 
        # yellow point (palm)
        selected_sticker_xyzcolnames = ['yellow_x_mm', 'yellow_y_mm', 'yellow_z_mm']
        # /!\ palm green point (NOT YELLOW)
        hand_adj_vertex_idx = np.where((-0.045 < hand_mesh.verts[:,0]) & (hand_mesh.verts[:,0] < -0.035)    # x value of vertices
                                        &( 0    < hand_mesh.verts[:,1])                                     # y value of vertices
                                        &(-0.01 < hand_mesh.verts[:,2]) & (hand_mesh.verts[:,2] < 0.01))    # z value of vertices
    
    # columns to center the frame on the hand for better hand posture estimation
    palm_sticker_pixelcolnames = ['green_pixel_x', 'green_pixel_y']
    # vertex of the hand mesh locating the palm and index (used for hand XYZ position estimation)
    palm_position_vertex_idx = np.where((-0.045 < hand_mesh.verts[:,0]) & (hand_mesh.verts[:,0] < -0.035)   # x value of vertices
                                        &( 0    < hand_mesh.verts[:,1])                                     # y value of vertices
                                        &(-0.01 < hand_mesh.verts[:,2]) & (hand_mesh.verts[:,2] < 0.01))    # z value of vertices
    index_position_vertex_idx = np.where((0.065 < hand_mesh.verts[:,0]) & (0.02 < hand_mesh.verts[:,2]))  # index finger tip
    thumb_position_vertex_idx = np.where(0.05 < hand_mesh.verts[:,2])
    
    # visualization and control
    marker_sphere = create_marker_sphere()
    viewer = viewer3D_init(mesh, arm_pointcloud, marker_sphere, title='output')
    display = display_init(title='input')

    # HandMesh Model Posture Estimator
    model = ModelPipeline()

    # output dataframe
    columns = [
        'time', 'frame_id', 
        'capture_read_failed', 'hand_posture_failed', 
        'contact_detected', 'contact_area', 'contact_depth',
        'contact_position_x',   'contact_position_y',   'contact_position_z', 
        'palm_position_x',      'palm_position_y',      'palm_position_z', 
        'index_position_x',     'index_position_y',     'index_position_z'
    ]
    df = pd.DataFrame(columns=columns)

    has_frame = True
    first_frame = True
    while has_frame:
        if first_frame:
            time = 0
            frame_idx = 0
            first_frame = False
        else:
            time += 1./30  # Azure Kinect Fs is 30 Hz
            frame_idx += 1
        
        # I. get the formatted frame
        has_frame, frame = kinect_capture.read()
        if not has_frame:
            if kinect_capture.get(cv2.CAP_PROP_POS_FRAMES) >= kinect_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                print("End of video file reached.")
                continue
            else:
                # Append the new row to the output DataFrame using loc
                df.loc[len(df)] = create_mockup_dataframe_row(time, frame_idx, is_capture_failed=True)
                print(f"frame {frame_idx}: capture read failed.")
                has_frame = True
                continue
        palm_xy = stickers_locs_df.loc[frame_idx, palm_sticker_pixelcolnames].values.flatten()
        print(f'palm_xy: {palm_xy}')
        if np.any(np.isnan(palm_xy)):
            frame = preprocess_video_frame(frame, crop_position, crop_size, left)
        else:
            frame = preprocess_video_frame(frame, palm_xy, crop_size, left)
        display_update(display, frame)

        # II. Extract the hand skeleton from the frame
        xyz, pointsuv, theta_mpii = model.process(frame)  # points, pointsuv, theta_mpii
        if hand_mismatch_detected(pointsuv, connections, int(frame.shape[0] / 32), crop_size):
            print(f"frame {frame_idx}: hand posture failed.")
            # model failed at detecting the hand, append the new row to the output DataFrame using loc
            df.loc[len(df)] = create_mockup_dataframe_row(time, frame_idx, is_hand_failed=True)
            continue
        display_update(display, frame, pointsuv, connections)

        # III. adjust the hand position using hand sticker and manual shift/rotation
        shift_curr_frame = overwrite_shift(frame_idx, hardcoded_shift, specific_sections_shift)  # switch potentially the manual adjustement
        hand_adj_vertex_value = stickers_locs_df.loc[frame_idx, selected_sticker_xyzcolnames].values.flatten()
        print(f'hand_xyz: {hand_adj_vertex_value}')
        if np.any(np.isnan(hand_adj_vertex_value)):
            continue
        hand_vertices = update_hand_posture(hand_mesh, theta_mpii, matrix_rotation_viewer, rotate, hand_adj_vertex_idx, hand_adj_vertex_value, shift_curr_frame, viewer=viewer, mesh=mesh)

        # Update visualization, use viewer.run() / viewer.stop() in debug mode console to interact with the viewer3D  
        viewer3D_update(viewer, mesh, hand_mesh, hand_vertices, marker_sphere, hand_adj_vertex_value, arm_pointcloud, verbose=True)

        # IV. Calculate tactile features of arm/hand interaction 
        hand_vertices_no_thumb = np.delete(hand_vertices, thumb_position_vertex_idx[0], axis=0)  # remove thumb for contact quantities (? Shan)
        contact_quantities = calculate_contact_quantities(hand_vertices_no_thumb, arm_pointcloud, avg_dist_nnarm)
        palm_position_xyz  = np.mean(hand_vertices[palm_position_vertex_idx[0], :], axis=0)
        index_position_xyz  = np.mean(hand_vertices[index_position_vertex_idx[0], :], axis=0)
        if contact_quantities['contact_detected']:
            print(f"frame {frame_idx}: contact detected!")

        # V. Append the new data to the output DataFrame using loc
        df.loc[len(df)] = create_row(time, frame_idx, contact_quantities, palm_position_xyz, index_position_xyz)

    print("done.")

    return df


if __name__ == '__main__':
  
    # input database path
    input_folder_path = "/home/basil/Documents/tactile-quantities_dataset_unit-test/dataset/"
    # input data paths
    video_fname_abs = os.path.join(input_folder_path, '2022-06-14_ST13-03_semicontrolled_block-order01_kinect.mp4')
    sticker_fname_abs = os.path.join(input_folder_path, '2022-06-14_ST13-03_kinect_handstickers_markers_locs.csv')
    arm_ply_fname_abs = os.path.join(input_folder_path, '2022-06-14_ST13-03_kinect_arm.ply')

    # processing parameters relative to the video
    parameters = {
        "video": {
            "crop_position": [1920/2, 1080/2],
            "crop_size": 400
        },
        "hand_stickers": {
            "color_idx": 0
        },
        "hand_used": {
            "left": False
        },
        "hand_mesh": {
            "rotate": False,
            "hardcoded_shift": np.array([10, 0, 0]), # variable storing the general manual adjustement for the video
            "specific_sections_shift": [  # variable storing manual adjustement for sections (in second) in the video
                [np.dot([25.5, 26.8], 30), np.array([10, 0, -15])],
                [np.dot([33.0, 33.7], 30), np.array([10, 0, -15])],
                [np.dot([45.7, 46.5], 30), np.array([10, 0, 10])],
                [np.dot([56.8, 58.0], 30), np.array([10, 0, 10])]
            ]
        },
        "viewer": {
            "matrix_rotation_viewer": [0, 1, 0, np.pi / 8.]
        }
    }

    # loading input data (arm point cloud, hand mesh model, RGB video for hand posture estimation)
    arm_pointcloud = create_arm_point_cloud(arm_ply_fname_abs, parameters["hand_used"]["left"])
    stickers_locs_df = extract_handstickers(sticker_fname_abs, parameters["hand_used"]["left"])
    kinect_capture = cv2.VideoCapture(video_fname_abs)
    kinect_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not kinect_capture.isOpened():
        sys.exit(f"program stops: kinect video couldn't be opened <{video_fname_abs}>")

    df = extract_touch_per_frame(kinect_capture, arm_pointcloud, stickers_locs_df, parameters)


    # Release the video capture object and close all OpenCV windows
    kinect_capture.release()
    cv2.destroyAllWindows()






