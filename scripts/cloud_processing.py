#!usr/bin/env python3
import os
import time
import torch
import numpy as np
import open3d as o3d
from robot_pcd import robotPointCloud

def fitPlane(downpcd):
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000,
                                                 probability=0.9999)
    
    [a, b, c, d] = plane_model.cpu().numpy().tolist()
    plane_equation = [a, b, c, d]

    inlier_cloud = downpcd.select_by_index(inliers)
    inlier_cloud, _ = inlier_cloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=0.5)
    inlier_cloud, _ = inlier_cloud.remove_radius_outliers(nb_points=16 , search_radius=0.05)

    boundary_cloud, mask = inlier_cloud.compute_boundary_points(0.5, 30)
    boundary_cloud, _ = boundary_cloud.remove_statistical_outliers(nb_neighbors=5, std_ratio=1.0)
    boundary_cloud, _ = boundary_cloud.remove_radius_outliers(nb_points=5 , search_radius=0.05)

    return inlier_cloud, boundary_cloud

def filter_cloud(boundary_pt, inlier_pts, desired_dist, batch_size=2000):
    num_points = inlier_pts.shape[0]
    update_inlier = torch.ones(num_points, dtype=torch.bool, device='cuda:0') 

    for i in range(0, num_points, batch_size):
        end = min(i+batch_size, num_points)
        distances = torch.linalg.norm(inlier_pts[i:end] - boundary_pt, dim=1)
        close_mask = distances <= desired_dist
        update_inlier[i:end] = torch.logical_and(update_inlier[i:end], ~close_mask) 
    return update_inlier

def transform_cloud(pointcloud, translation, rotation):
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3, 3] = translation
    transform_tensor = torch.from_numpy(transform).float().to("cuda:0")

    n = pointcloud.shape[0]
    points = torch.hstack((pointcloud, torch.ones(n,1, device="cuda:0"))).T
    transformed_points = torch.matmul(transform_tensor, points)[:3].T
    return transformed_points

def nearest_search(filtered_inlier_cloud, target_point):
    pcd_tree = o3d.geometry.KDTreeFlann(filtered_inlier_cloud)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(target_point, 1)
    nearest_point = np.asarray(filtered_inlier_cloud.points)[idx[0]]
    return nearest_point

def ProcessCloud(pcd, robot_position, robot_orientation, target_coord):
    robot_pcd = robotPointCloud()
    pcd = np.concatenate((pcd, robot_pcd), 0)

    # read np point cloud and convert to CUDA
    pcd_cpu = o3d.t.geometry.PointCloud(pcd)
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_cpu.to_legacy())
    pcd_gpu = tensor_pcd.to(device=o3d.core.Device("cuda:0"))

    # point cloud pre-processing    
    downpcd = pcd_gpu.voxel_down_sample(voxel_size=0.02)
    downpcd.estimate_normals(max_nn=30, radius=0.1)

    # fit plane
    inlier_cloud, boundary_cloud = fitPlane(downpcd)

    # filter cloud
    inlier_cpu = inlier_cloud.to(o3d.core.Device('CPU:0'))
    inlier_tensor = torch.tensor(inlier_cpu.point.positions.numpy(), device='cuda:0')
    boundary_cpu = boundary_cloud.to(o3d.core.Device('CPU:0'))
    boundary_tensor = torch.tensor(boundary_cpu.point.positions.numpy(), device='cuda:0')

    # remove boundary points that are within the robot
    robot_origin = torch.tensor([0,0,-0.171], device="cuda:0")
    boundary_mask = filter_cloud(robot_origin, boundary_tensor, 0.3)
    boundary_tensor_filtered = boundary_tensor[boundary_mask]

    bound_filtered = boundary_tensor_filtered.detach().cpu().numpy()
    bound_filtered_pcd = o3d.geometry.PointCloud()
    bound_filtered_pcd.points = o3d.utility.Vector3dVector(bound_filtered)
    bound_filtered_pcd.paint_uniform_color([0, 0, 1])
    
    filtered_bound = torch.ones(inlier_tensor.shape[0], dtype=torch.bool, device='cuda:0')
    for boundary_pt in boundary_tensor_filtered:
        filtered_bound = torch.logical_and(filtered_bound, filter_cloud(boundary_pt, inlier_tensor, 0.2))
    filtered_inlier_cloud = inlier_tensor[filtered_bound]
    
    np_filtered = filtered_inlier_cloud.detach().cpu().numpy()
    np_filtered_pcd = o3d.geometry.PointCloud()
    np_filtered_pcd.points = o3d.utility.Vector3dVector(np_filtered)
    np_filtered_pcd.paint_uniform_color([0, 1, 0.5])
    
    # transfrom pcd to map frame
    transform_inlier_cloud = transform_cloud(filtered_inlier_cloud, robot_position, robot_orientation)

    # find nearest point to target
    np_inlier_cloud = transform_inlier_cloud.cpu().numpy()
    output_inlier_cloud = o3d.geometry.PointCloud()
    output_inlier_cloud.points = o3d.utility.Vector3dVector(np_inlier_cloud)

    nearest_point = nearest_search(output_inlier_cloud, target_coord)

    # Visualizations
    output_inlier_cloud.paint_uniform_color([1, 0, 0])

    def create_sphere_at_point(point, radius, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        return sphere
    
    sphere_radius = 0.05
    robot_color = [0, 0, 1]
    nearest_point_color = [0, 1, 0]
    target_color = [1, 0, 1]
    
    robot_sphere = create_sphere_at_point(robot_position, sphere_radius, robot_color)
    nearest_point_sphere = create_sphere_at_point(nearest_point, sphere_radius, nearest_point_color)
    target_sphere = create_sphere_at_point(target_coord, sphere_radius, target_color)

    # o3d.visualization.draw_geometries([output_inlier_cloud, robot_sphere, nearest_point_sphere, target_sphere])



    return np_inlier_cloud, nearest_point
