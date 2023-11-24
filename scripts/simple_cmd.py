#!/usr/bin/env python3 
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from robot_navigator import BasicNavigator, TaskResult
from nav2_msgs.action import NavigateToPose

class PathFollower(Node):
    def __init__(self, navigator):
        super().__init__('path_follower_node')
        self.navigator = navigator
        self.waypoint_subscriber = self.create_subscription(
            Float64MultiArray,
            'waypoint_publisher',
            self.waypoint_callback,
            10)

        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')        
        self.path_received = False
        self.navigating = False

    def reshape_path(self, msg):
        num_waypoints = msg.layout.dim[0].size
        num_coords = msg.layout.dim[1].size
        waypoints_arr = np.array(msg.data).reshape((num_waypoints, num_coords))
        return waypoints_arr

    def create_path(self, path_coord, frame_id='map'):
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = rclpy.time.Time().to_msg()
        for coords in path_coord:
            waypoint = PoseStamped()
            waypoint.header.frame_id = frame_id
            waypoint.pose.position.x = coords[0]
            waypoint.pose.position.y = coords[1]
            waypoint.pose.position.z = coords[2]
            waypoint.pose.orientation.w = 1.0  
            path.poses.append(waypoint)
        return path

    def waypoint_callback(self, msg):
        print('New waypoints received')
        self.path_received = True

        if not self.navigator.isTaskComplete():
            self.navigator.cancelTask()
        

        path_coord = self.reshape_path(msg)
        path = self.create_path(path_coord)

        self.follow_path(path)
    
    def follow_path(self, path):
        self.path_received = False
        # smoothed_path = self.navigator.smoothPath(path)
        # self.navigator.followPath(smoothed_path)
        self.navigator.followPath(path)
        
        i = 0
        while not self.navigator.isTaskComplete():
            i += 1
            feedback = self.navigator.getFeedback()
            if feedback and i % 5 == 0:
                print(
                    'Estimated distance remaining to goal position: '
                    + '{0:.3f}'.format(feedback.distance_to_goal)
                    + '\nCurrent speed of the robot: '
                    + '{0:.3f}'.format(feedback.speed)
                )

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Goal succeeded!')
            self.flag = True
        elif result == TaskResult.CANCELED:
            print('Goal was canceled!')
        elif result == TaskResult.FAILED:
            print('Goal failed!')
        else:
            print('Goal has an invalid return status!')


def main():
    rclpy.init()
    navigator = BasicNavigator()
    path_follower = PathFollower(navigator)
    rclpy.spin(path_follower)
    navigator.lifecycleShutdown()

if __name__ == '__main__':
    main()