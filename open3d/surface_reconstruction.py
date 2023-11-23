import os
import time
import numpy as np
import open3d as o3d



def BPA(pcd):
    print('BPA method')
    # Define the radius for BPA
    radii = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

    # Perform the Ball Pivoting Algorithm
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd, o3d.utility.DoubleVector(radii))
    return mesh 

def Poisson(pcd):
    print('Poisson method')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

# def Alpha(pcd, alpha):
#     print('Alpha method')
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#     mesh.compute_vertex_normals()
#     return mesh

def Alpha(pcd, alpha, outlier_removal=True, cluster_analysis=True):
    print('Alpha method')
    # Outlier removal
    if outlier_removal:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Cluster analysis (optional)
    if cluster_analysis:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
        pcd = pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])
    # Alpha shape reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    # # Define the color for the point cloud and mesh
    # pcd.paint_uniform_color([1, 0.706, 0])  # Orange for point cloud
    # mesh.paint_uniform_color([0, 0.651, 0.929])  # Blue for mesh
    # # Create a visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # Add the point cloud and mesh
    # vis.add_geometry(pcd)
    # vis.add_geometry(mesh)
    # # Run the visualizer
    # vis.run()
    # vis.destroy_window()
    return mesh, pcd

def Delaunay(pcd):
    print('Delaunay method')
    # Project to 2D
    points = np.asarray(pcd.points)
    points_2d = points[:, :2]

    # Perform Delaunay triangulation in 2D
    hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)
    hull.compute_vertex_normals()
    return hull


def main():
    # load point cloud
    ply_path = os.path.expanduser("pc_snapshot_uneven_tree_terrain.xyz")
    pcd = o3d.io.read_point_cloud(ply_path)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.estimate_normals()

    start_time = time.time()
    mesh = BPA(pcd)
    # mesh = Poisson(pcd)
    # mesh = Alpha(pcd)
    # mesh = Delaunay(pcd)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()
