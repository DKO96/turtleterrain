#!/usr/bin/env python3
import rclpy
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import KDTree

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from path_planning import PathPlanner

class O3DNode(Node):
    def __init__(self):
        super().__init__('open3d_node')
        # subscription for pointcloud
        self.pcd_subscription = self.create_subscription(
            Float64MultiArray,
            'xyz_pointcloud',
            self.waypoint_generator,
            10
        )

        #subscription for current pose from amcl node
        self.amcl_subscription = self.create_subscription(
            PoseStamped,
            'current_pose',
            self.current_pose_callback,
            10
        )

        #subscription for goal location from target_point topic
        self.target_subscription = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.target_callback,
            10
        )

        self.current_pose = None
        self.target_pose = None
    
    def current_pose_callback(self, msg):
        self.current_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 msg.pose.position.z])
    
    def target_callback(self, msg):
        self.target_pose = np.array([msg.pose.position.x,
                                msg.pose.position.y,
                                msg.pose.position.z])

    def reshape_pcd(self, msg):
        points_dim = msg.layout.dim[0].size
        coords_dim = msg.layout.dim[1].size
        pc_array = np.array(msg.data).reshape((points_dim, coords_dim))
        return pc_array

    def filter_vertical_pts(self, pc_array, tolerance):
        # prepare point cloud
        downpcd = o3d.geometry.PointCloud()
        downpcd.points = o3d.utility.Vector3dVector(pc_array)
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

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

    def get_planar_patches(self, pcd):
    # find planar patches
        oboxes = pcd.detect_planar_patches(
            normal_variance_threshold_deg=60,
            coplanarity_deg=75,
            outlier_ratio=0.75,
            min_plane_edge_length=0.0,
            min_num_points=0,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # print("Detected {} patches".format(len(oboxes)))

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

    def find_nearest_pt(self, inlier_arr, target):
        # find nearest point to target from inlier pts
        plane_pts = np.vstack(inlier_arr)
        target = np.array([8, 0, 0.1])
        tree = KDTree(plane_pts)
        dist, index = tree.query(target)
        nearest_pt = plane_pts[index]

        # print(f'nearest point: {nearest_pt}')
        # print(f'nearest point dist: {dist}')

        return nearest_pt 

    def waypoint_generator(self, msg):
        if self.target_pose is None:
            self.get_logger().info('No target pose recieved.')
            return 

        # Reshape 1d pcd to n x 3 array
        pc_array = self.reshape_pcd(msg)

        # Filter vertical points 
        tolerance = 0.25
        pcd = self.filter_vertical_pts(pc_array, tolerance)

        # Get Planar patches
        inlier_arr, geometries = self.get_planar_patches(pcd)

        # find nearest point in plane to target
        nearest_pt = self.find_nearest_pt(inlier_arr, self.target_pose)

        



        # self.get_logger().info(f'current_pose: {self.current_pose}')
        # self.get_logger().info(f'nearest_pt: {nearest_pt}')
        # self.get_logger().info('shutting down node')
        # self.pcd_subscription.destroy()
        # self.amcl_subscription.destroy()
        # self.target_subscription.destroy()
        # rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)

    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()















