#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class TargetPublisher(Node):

    def __init__(self):
        super().__init__('target_publisher')
        self.target_publisher = self.create_publisher(
            PoseStamped,
            'target_pose',
            10
        )

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # postion and direction of pose
        msg.pose.position.x = 1.0
        msg.pose.position.y = 1.0
        msg.pose.position.z = 0.0
        # msg.pose.orientation.x = 0.0
        # msg.pose.orientation.y = 0.0
        # msg.pose.orientation.z = 0.0
        # msg.pose.orientation.w = 1.0
        
        self.target_publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg}')


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = TargetPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()