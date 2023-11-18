import os
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import KDTree

import surface_reconstruction

def filter_vertical_pts(downpcd, tolerance):
    # convert numpy array to tensor
    tensor = torch.from_numpy(np.asarray(downpcd.normals))

    # use gpu if available
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")

    # check z-axis of normals
    is_horizontal = torch.abs(tensor[:,2] - 1) < tolerance
    is_horizontal_np = is_horizontal.cpu().numpy()

    # Filter the point cloud points
    points = np.asarray(downpcd.points)
    horizontal_points = points[is_horizontal_np]

    # Create a new point cloud for the filtered points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(horizontal_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd

def get_planar_patches(pcd):
    # find planar patches
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=75,
        coplanarity_deg=75,
        outlier_ratio=0.8,
        min_plane_edge_length=0.0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))

    print("Detected {} patches".format(len(oboxes)))

    # get points within the planar patches
    geometries = []
    inlier_arr = []
    for obox in oboxes:
        geometries.append(obox)
        inlier_pcd = pcd.crop(obox)
        inlier_pts = np.asarray(inlier_pcd.points)
        inlier_arr.append(inlier_pts)

        inlier_pcd.paint_uniform_color(obox.color)
        geometries.append(inlier_pcd)

        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)

    return inlier_arr, geometries

def find_nearest_pt(inlier_arr, target):
    plane_pts = np.vstack(inlier_arr)

    # find nearest point to target
    target = np.array([8, 0, 0.1])
    tree = KDTree(plane_pts)
    dist, index = tree.query(target)
    nearest_pt = plane_pts[index]

    print(f'nearest point: {nearest_pt}')
    print(f'nearest point dist: {dist}')

    return nearest_pt 

def main():
    # load point cloud
    downpcd = o3d.io.read_point_cloud('pc_snapshot_uneven_tree_terrain_1.xyz')
    # downpcd = pointcloud.voxel_down_sample(voxel_size=0.03)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # filter vertical points
    tolerance = 0.5
    pcd = filter_vertical_pts(downpcd, tolerance) 

    # get points in planar patches
    inlier_arr, geometries = get_planar_patches(pcd)

    # surface reconstruction
    inlier_pts = np.vstack(inlier_arr)
    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inlier_pts)
    inlier_pcd.estimate_normals()

    # mesh = surface_reconstruction.BPA(inlier_pcd)
    # mesh = surface_reconstruction.Poisson(inlier_pcd)
    mesh, pcd = surface_reconstruction.Alpha(inlier_pcd, 0.2)
    print(f'pcd shape: {type(pcd)}')

    # export pcd points 
    reg_pcd = o3d.geometry.PointCloud()
    reg_pcd.points = pcd.points
    o3d.io.write_point_cloud('surface_points.xyz',reg_pcd)

    # mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=10)
    # mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out, pcd])

    # # find nearest point in plane to target
    # target = np.array([8.0, 0.0, 0.1])
    # nearest_pt = find_nearest_pt(inlier_arr, target)

    # # visualization
    # marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    # marker.translate(nearest_pt)
    # marker.paint_uniform_color([1,0,0])

    # geometries.append(marker)

    # # o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()







