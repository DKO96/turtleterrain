import open3d as o3d

# load point cloud
pcd = o3d.io.read_point_cloud('pc_snapshot_uneven_tree_terrain.xyz')
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.estimate_normals()

o3d.visualization.draw_geometries([pcd])