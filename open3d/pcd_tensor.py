import time
import numpy as np
import open3d as o3d

start_time = time.time()

# read xyz point cloud
pcd_cpu = o3d.t.io.read_point_cloud('pc_snapshot_uneven_tree_terrain_1.xyz')

# use CUDA
pcd = pcd_cpu.to(o3d.core.Device("cuda:0"))

# voxel down sample point cloud
downpcd = pcd.voxel_down_sample(voxel_size=0.03)

# estimate point cloud normals
downpcd.estimate_normals(max_nn=30, radius=0.1)

# fit plane
plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,
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
# outlier_cloud = downpcd.select_by_index(inliers, invert=True)

# remove outliers in plane inlier_cloud
inlier_cloud, _ = inlier_cloud.remove_statistical_outliers(nb_neighbors=5, std_ratio=1.0)

# find boundaries of plane inlier cloud
boundaries, mask = inlier_cloud.compute_boundary_points(0.1, 30)
print(f"Detect {boundaries.point.positions.shape[0]} bnoundary points from {inlier_cloud.point.positions.shape[0]} points.")
boundaries = boundaries.paint_uniform_color([0.0, 0.0, 1.0])

# inlier_cloud_cpu = inlier_cloud.to(o3d.core.Device("CPU:0"))
# outlier_cloud_cpu = outlier_cloud.to(o3d.core.Device("CPU:0"))


# o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), boundaries.to_legacy()])


end_time = time.time()
print(f'Execution time: {end_time - start_time}')




