import time
import torch
import numpy as np
import open3d as o3d

start_time = time.time()

# read xyz point cloud
pcd_cpu = o3d.t.io.read_point_cloud('pc_snapshot_uneven_tree_terrain_1.xyz')

# use CUDA
pcd = pcd_cpu.to(o3d.core.Device("cuda:0"))

# voxel down sample point cloud
downpcd = pcd.voxel_down_sample(voxel_size=0.025)

# estimate point cloud normals
downpcd.estimate_normals(max_nn=30, radius=0.1)

# fit plane
plane_model, inliers = downpcd.segment_plane(distance_threshold=0.05,
                                            ransac_n=3,
                                            num_iterations=1000,
                                            probability=0.9999)

# print plane equation (conversion to CPU required for numpy library)
# plane_model_cpu = plane_model.to(o3d.core.Device("CPU:0"))                
# [a, b, c, d] = plane_model_cpu.numpy().tolist()
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# get plane inlier cloud
inlier_cloud = downpcd.select_by_index(inliers)
inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = downpcd.select_by_index(inliers, invert=True)

# remove outliers in plane inlier_cloud
inlier_cloud, _ = inlier_cloud.remove_statistical_outliers(nb_neighbors=5, std_ratio=1.0)

# find boundaries of plane inlier cloud
boundary_cloud, mask = inlier_cloud.compute_boundary_points(0.1, 30)
# print(f"Detect {boundary_cloud.point.positions.shape[0]} boundary points from {inlier_cloud.point.positions.shape[0]} points.")
# boundaries = boundary_cloud.paint_uniform_color([0.0, 0.0, 1.0])


# filter inlier cloud to only have points 0.1m away from the boundary
inlier_cpu = inlier_cloud.to(o3d.core.Device('CPU:0'))
inlier_tensor = torch.tensor(inlier_cpu.point.positions.numpy(), device='cuda:0')

boundary_cpu = boundary_cloud.to(o3d.core.Device('CPU:0'))
boundary_tensor = torch.tensor(boundary_cpu.point.positions.numpy(), device='cuda:0')

dist = torch.tensor(0.1, device='cuda:0')
tolerance = 0.02

def filter_cloud(boundary_pt, inlier_pts, batch_size=5000):
    num_points = inlier_pts.shape[0]
    update_inlier = torch.ones(num_points, dtype=torch.bool, device='cuda:0') 

    for i in range(0, num_points, batch_size):
        end = min(i+batch_size, num_points)
        distances = torch.linalg.norm(inlier_pts[i:end] - boundary_pt, dim=1)
        close_mask = distances <= dist 
        update_inlier[i:end] = torch.logical_and(update_inlier[i:end], ~close_mask) 
        # print(update_inlier)
    return update_inlier

robot_tensor = torch.tensor([], device='cuda:0')
filtered_dist = torch.ones(inlier_tensor.shape[0], dtype=torch.bool, device='cuda:0')
filtered_bound = torch.logical_and(filtered_dist, filter_cloud(robot_tensor, inlier_tensor))
filtered_inlier_tensor = inlier_tensor[filtered_bound]



filtered_bound = torch.ones(inlier_tensor.shape[0], dtype=torch.bool, device='cuda:0')
for boundary_pt in boundary_tensor:
    filtered_bound = torch.logical_and(filtered_bound, filter_cloud(boundary_pt, inlier_tensor))

filtered_inlier_tensor = inlier_tensor[filtered_bound]

# output_pcd = inlier_cloud.to_legacy()
# o3d.io.write_point_cloud('surface_points_tensor.xyz', output_pcd)

end_time = time.time()
print(f'Execution time: {end_time - start_time}')

filtered_inlier_cloud = o3d.geometry.PointCloud()
filtered_inlier_cloud.points = o3d.utility.Vector3dVector(filtered_inlier_tensor.cpu().numpy())
filtered_inlier_cloud.paint_uniform_color([0, 1, 0])
# o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), boundaries.to_legacy()])
o3d.visualization.draw_geometries([filtered_inlier_cloud, boundary_cloud.to_legacy()])







