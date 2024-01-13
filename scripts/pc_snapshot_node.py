#!/usr/bin/env python3
import os
import time
import rclpy
import numpy as np
import open3d as o3d

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray


class O3DNode(Node):
    def __init__(self):
        super().__init__('snapshot_node')

        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        # self.o3d_pcd = o3d.geometry.PointCloud()

        self.subscription = self.create_subscription(
            Float64MultiArray,
            'xyz_pointcloud',
            self.o3d_visualizer,
            10
        )

    def o3d_visualizer(self, msg):
        points_dim = msg.layout.dim[0].size
        coords_dim = msg.layout.dim[1].size
        pc_array = np.array(msg.data).reshape((points_dim, coords_dim))

        file_path = os.path.expanduser('~/Documents/Turtleterrain/src/turtleterrain/open3d/Images/')
        np.savetxt(file_path + 'steps_1.xyz', pc_array)
        self.get_logger().info(f'xyz format shape: {pc_array}')
        self.get_logger().info('shutting down node')
        self.subscription.destroy()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)

    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


