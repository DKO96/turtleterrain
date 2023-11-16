#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class Nav2WaypointFollower(Node):

    def __init__(self, waypoints):
        super().__init__('path_follower_node')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.waypoints = waypoints
        self.current_waypoint_index = 0

    def send_next_waypoint(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('All waypoints reached')
            return

        waypoint = self.waypoints[self.current_waypoint_index]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = waypoint

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.current_waypoint_index += 1
        self.send_next_waypoint()

    def feedback_callback(self, feedback_msg):
        # Implement any feedback handling if needed
        pass

def create_waypoint(x, y, z):
    waypoint = PoseStamped()
    waypoint.header.frame_id = 'map'
    waypoint.pose.position.x = x
    waypoint.pose.position.y = y
    waypoint.pose.orientation.w = 1.0  # Facing forward
    return waypoint

def main(args=None):
    rclpy.init(args=args)

    # Define your waypoints here
    waypoints = [
        # create_waypoint(1.2, -1.2, 0.0),  
        # create_waypoint(0.6, -1.2, 0.0),  
        # create_waypoint(0.0, -1.2, 0.0),  
        # create_waypoint(-0.6, -1.2, 0.0),  
        # create_waypoint(-1.2, -1.2, 0.0),  
        
        create_waypoint(-1.2, -1.2, 0.0),  
        create_waypoint(-0.6, -1.2, 0.0),  
        create_waypoint(0.0, -1.2, 0.0),  
        create_waypoint(0.6, -1.2, 0.0),  
        create_waypoint(1.2, -1.2, 0.0),  
    ]

    waypoint_follower = Nav2WaypointFollower(waypoints)
    waypoint_follower.send_next_waypoint()

    rclpy.spin(waypoint_follower)
    waypoint_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



