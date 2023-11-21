import time
import torch
import numpy as np
import open3d as o3d

def fitPlane(downpcd):
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.05,
                                            ransac_n=3,
                                            num_iterations=1000,
                                            probability=0.9999)

    inlier_cloud = downpcd.select_by_index(inliers)
    inlier_cloud, _ = inlier_cloud.remove_statistical_outliers(nb_neighbors=5, std_ratio=1.0)
    boundary_cloud, mask = inlier_cloud.compute_boundary_points(0.5, 60)
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

def nearest_search(filtered_inlier_cloud, target_point):
    pcd_tree = o3d.geometry.KDTreeFlann(filtered_inlier_cloud)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(target_point, 1)
    nearest_point = np.asarray(filtered_inlier_cloud.points)[idx[0]]
    return nearest_point

def ProcessCloud(np_pcd, target_coord):
    # read np point cloud and convert to CUDA
    pcd_cpu = o3d.geometry.PointCloud()
    pcd_cpu.points = o3d.utility.Vector3dVectory(np_pcd)
    pcd = pcd_cpu.to(o3d.core.Device("cuda:0"))

    # point cloud pre-processing    
    downpcd = pcd.voxel_down_sample(voxel_size=0.025)
    downpcd.estimate_normals(max_nn=30, radius=0.1)

    # fit plane
    inlier_cloud, boundary_cloud = fitPlane(downpcd)

    # filter cloud
    inlier_cpu = inlier_cloud.to(o3d.core.Device('CPU:0'))
    inlier_tensor = torch.tensor(inlier_cpu.point.positions.numpy(), device='cuda:0')
    boundary_cpu = boundary_cloud.to(o3d.core.Device('CPU:0'))
    boundary_tensor = torch.tensor(boundary_cpu.point.positions.numpy(), device='cuda:0')

    filtered_bound = torch.ones(inlier_tensor.shape[0], dtype=torch.bool, device='cuda:0')
    for boundary_pt in boundary_tensor:
        filtered_bound = torch.logical_and(filtered_bound, filter_cloud(boundary_pt, inlier_tensor, 0.1))
    filtered_inlier_tensor = inlier_tensor[filtered_bound]

    # find nearest point to target
    filtered_inlier_cloud = o3d.geometry.PointCloud()
    filtered_inlier_cloud.points = o3d.utility.Vector3dVector(filtered_inlier_tensor.cpu().numpy())

    return nearest_search(filtered_inlier_cloud, target_coord)

