#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from path_planning import PathPlanner
from cloud_processing import ProcessCloud
from robot_pcd import robotPointCloud

class O3DNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        self.goal_reached = False
        self.target_pose = None
        self.amcl_pose = None
        self.prev_waypoint_goal = None
        self.current_position = None
        self.current_orientation = None
        self.prev_position = None

        # subscription for pointcloud
        self.pcd_subscriber = self.create_subscription(
            Float64MultiArray,
            'xyz_pointcloud',
            self.get_waypoints,
            10)

        # subscription for current pose from amcl_pose and odom
        self.amcl_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_position',
            self.amcl_pose_callback,
            10)
        
        # subscription for goal location from target_point topic
        self.target_subscriber = self.create_subscription(
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
                                   msg.pose.pose.position.z,
                                   msg.pose.pose.orientation.x,
                                   msg.pose.pose.orientation.y,
                                   msg.pose.pose.orientation.z,
                                   msg.pose.pose.orientation.w])
        self.update_current_pose()

    def update_current_pose(self):
        Q = self.amcl_pose[-4:] / np.linalg.norm(self.amcl_pose[-4:])       
        x, y, z, w = Q[0], Q[1], Q[2], Q[3]

        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*z*w
        r02 = 2*x*z + 2*y*w

        r10 = 2*x*y + 2*z*w
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*x*w

        r20 = 2*x*z - 2*y*w
        r21 = 2*y*z + 2*x*w
        r22 = 1 - 2*x*x - 2*y*y

        self.current_orientation = np.array([[-r00, -r01, r02],
                                             [-r10, -r11, r12],
                                             [r20, r21, r22]])
        self.current_position = np.array(self.amcl_pose[:3])
        
    def publish_waypoints(self, msg):
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
        pcd_array = self.reshape_pcd(msg)
        pcd, nearest_point = ProcessCloud(pcd_array, self.current_position, self.current_orientation, self.target_pose)
        waypoints = PathPlanner(pcd, self.current_position, nearest_point, self.target_pose)
        # print(f'generated waypoints: {waypoints}')
        self.publish_waypoints(waypoints)    
        self.prev_waypoint_goal = nearest_point
    
    def distance_to_waypoint_end(self):
        if self.prev_waypoint_goal is not None and self.current_position is not None: 
            return np.linalg.norm(self.prev_waypoint_goal - self.current_position)
    
    def distance_advanced(self):
        if self.prev_position is not None and self.current_position is not None: 
            return np.linalg.norm(self.prev_position - self.current_position)

    def distance_to_target(self):
        return(np.linalg.norm(self.current_position - self.target_pose))

    def get_waypoints(self, msg):
        if self.target_pose is None or self.current_position is None:
            self.get_logger().info('Waiting for target_pose and current_pose to be available.')
            return

        if self.distance_to_target() < 0.1:
            print('Target location reached')
            rclpy.shutdown()

        if self.prev_waypoint_goal is None or self.distance_advanced() > 0.25:
            self.waypoint_generator(msg)
            self.prev_position = self.current_position

def main(args=None):
    rclpy.init(args=args)
    open3d_node = O3DNode()
    rclpy.spin(open3d_node)
    open3d_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
