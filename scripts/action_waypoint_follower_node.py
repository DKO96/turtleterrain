#!/usr/bin/env python3
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import FollowWaypoints
from action_msgs.msg import GoalStatus
from std_msgs.msg import Float64MultiArray


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')
        self._action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')

        self.waypoint_subscriber = self.create_subscription(
            Float64MultiArray,
            'waypoint_publisher',
            self.waypoint_callback,
            10)

    def send_waypoints(self, waypoints):
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'current waypoint: {feedback}')
        # Handle feedback here (e.g., current waypoint)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().info('Navigation failed')

    def waypoint_callback(self, msg):
        num_waypoints = msg.layout.dim[0].size
        num_coords = msg.layout.dim[1].size
        waypoint_array = np.array(msg.data).reshape


    def create_waypoint(x, y, z):
        waypoint = PoseStamped()
        waypoint.header.stamp = rclpy.time.Time().to_msg()
        waypoint.header.frame_id = "map"  # Assuming the map frame of reference

        waypoint.pose.position = Point(x=x, y=y, z=z)
        waypoint.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        return waypoint

def main(args=None):
    rclpy.init(args=args)
    waypoint_follower = WaypointFollower()

    # Define your waypoints here
    waypoints = [
        # Populate with PoseStamped waypoints
        WaypointFollower.create_waypoint(0.3, 0.3, -0.175),
        WaypointFollower.create_waypoint(0.4, 0.4, -0.175),
        WaypointFollower.create_waypoint(0.5, 0.5, -0.175),
        WaypointFollower.create_waypoint(0.6, 0.6, -0.175),
        WaypointFollower.create_waypoint(0.7, 0.7, -0.175),
        WaypointFollower.create_waypoint(0.8, 0.8, -0.175),
        WaypointFollower.create_waypoint(0.9, 0.9, -0.175),
        WaypointFollower.create_waypoint(1.0, 1.0, -0.175),
        
        # WaypointFollower.create_waypoint(1.0, 1.0, -0.175),
        # WaypointFollower.create_waypoint(0.9, 0.9, -0.175),
        # WaypointFollower.create_waypoint(0.8, 0.8, -0.175),
        # WaypointFollower.create_waypoint(0.7, 0.7, -0.175),
        # WaypointFollower.create_waypoint(0.6, 0.6, -0.175),
        # WaypointFollower.create_waypoint(0.5, 0.5, -0.175),
        # WaypointFollower.create_waypoint(0.4, 0.4, -0.175),
        # WaypointFollower.create_waypoint(0.3, 0.3, -0.175),
    ]

    waypoint_follower.send_waypoints(waypoints)
    rclpy.spin(waypoint_follower)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
