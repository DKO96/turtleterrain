#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from path_planning import PathPlanner
from cloud_processing import ProcessCloud

class O3DNode(Node):
    def __init__(self):
        super().__init__('open3d_node')
        self.target_pose = None
        self.amcl_pose = None
        self.odom_pose = None
        self.current_pose = None
        
        # subscription for pointcloud
        self.pcd_subscription = self.create_subscription(
            Float64MultiArray,
            'xyz_pointcloud',
            self.waypoint_generator,
            10)

        # subscription for current pose from amcl_pose and odom
        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_position',
            self.amcl_pose_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_pose_callback,
            10)
        
        # subscription for goal location from target_point topic
        self.target_subscription = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.target_callback,
            10)

        # publisher for waypoints
        self.waypoint_publisher = self.create_publisher(
            Float64MultiArray,
            'waypoint_publisher',
            10)

    def target_callback(self, msg):
        self.target_pose = np.array([msg.pose.position.x,
                                     msg.pose.position.y,
                                     msg.pose.position.z])
    
    def amcl_pose_callback(self, msg):
        self.amcl_pose = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y,
                                   msg.pose.pose.position.z])
        self.update_current_pose()
    
    def odom_pose_callback(self, msg):
        self.odom_pose = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y,
                                   msg.pose.pose.position.z])
        self.update_current_pose()

    def update_current_pose(self):
        if self.amcl_pose is not None and self.odom_pose is not None:
            self.current_pose = (self.amcl_pose + self.odom_pose) / 2
        else:
            self.current_pose = self.odom_pose

    def listener_callback(self, msg):
        num_waypoints = msg.shape[0]
        num_coords = msg.shape[1]
        dim1 = MultiArrayDimension(label='waypoints', size=num_waypoints, stride=num_waypoints*num_coords)
        dim2 = MultiArrayDimension(label='coordinates', size=num_coords, stride=num_coords)

        waypoint_msg = Float64MultiArray()
        waypoint_msg.layout.dim = [dim1, dim2]
        waypoint_msg.layout.data_offset = 0
        waypoint_msg.data = msg.flatten().tolist()

        self.waypoint_publisher.publish(waypoint_msg)

    def reshape_pcd(self, msg):
        points_dim = msg.layout.dim[0].size
        coords_dim = msg.layout.dim[1].size
        pc_array = np.array(msg.data).reshape((points_dim, coords_dim))
        return pc_array

    def waypoint_generator(self, msg):
        if self.target_pose is None or self.current_pose is None:
            self.get_logger().info('Waiting for target_pose and current_pose to be available.')
            return

        pcd_array = self.reshape_pcd(msg)
        pcd, nearest_point = ProcessCloud(pcd_array, self.target_pose)
        waypoints = PathPlanner(pcd, self.current_pose, nearest_point)
        self.get_logger().info(f'waypoints: \n{waypoints}')
        self.listener_callback(waypoints)


def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)

    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

