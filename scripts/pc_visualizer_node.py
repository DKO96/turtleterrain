#!/usr/bin/env python3
import time
import rclpy
import numpy as np
import open3d as o3d

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray


class O3DNode(Node):
    def __init__(self):
        super().__init__('open3d_node')

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

        np.savetxt('pc_snapshot.xyz', pc_array)
        self.get_logger().info(f'xyz format shape: {pc_array}')
        self.get_logger().info('shutting down node')
        self.subscription.destroy()
        rclpy.shutdown()


        # self.vis.remove_geometry(self.o3d_pcd)
        # self.o3d_pcd = o3d.geometry.PointCloud(
        #                     o3d.utility.Vector3dVector(pc_array))
        # self.vis.add_geometry(self.o3d_pcd)
        # self.vis.poll_events()
        # self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)

    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


