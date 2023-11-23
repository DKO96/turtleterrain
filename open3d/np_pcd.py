import time
import torch
import numpy as np
import open3d as o3d

# pcd = o3d.t.geometry.PointCloud()
# print(pcd, "\n")

# # To create a point cloud on CUDA, specify the device.
# pcd = o3d.t.geometry.PointCloud(o3d.core.Device("cuda:0"))
# print(pcd, "\n")

# # Create a point cloud from open3d tensor with dtype of float32.
# pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor([[0, 0, 0], [1, 1, 1]], o3d.core.float32))
# print(pcd, "\n")

# # Create a point cloud from open3d tensor with dtype of float64.
# pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor([[0, 0, 0], [1, 1, 1]], o3d.core.float64))
# print(pcd, "\n")

# # Create a point cloud from numpy array. The array will be copied.
pcd = o3d.t.geometry.PointCloud(
    np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
# print(pcd, "\n")

# # Create a point cloud from python list.
# pcd = o3d.t.geometry.PointCloud([[0., 0., 0.], [1., 1., 1.]])
# print(pcd, "\n")

# # Error creation. The point cloud must have shape of (N, 3).
# try:
#     pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor([0, 0, 0, 0], o3d.core.float32))
# except:
#     print(f"Error creation. The point cloud must have shape of (N, 3).")




legacy_pcd = pcd.to_legacy()
# print(f'legacy_pcd: {legacy_pcd}')

tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd.to_legacy())
print(f'tensor_pcd: {tensor_pcd}')

gpu_pcd = tensor_pcd.to(device=o3d.core.Device("cuda:0"))
print(f'gpu_pcd: {gpu_pcd}')

# Convert from legacy point cloud with data type of float64.
# tensor_pcd_f64 = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd, o3d.core.float64)
# print(tensor_pcd_f64, "\n")
