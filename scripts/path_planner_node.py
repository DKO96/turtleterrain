#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from path_planning import PathPlanner
from cloud_processing import ProcessCloud

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

    def waypoint_generator(self, msg):
        pcd_array = self.reshape_pcd(msg)
        pcd, nearest_point = ProcessCloud(pcd_array, self.target_pose)
        waypoints = PathPlanner(pcd, self.current_pose, nearest_point)


def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)

    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

