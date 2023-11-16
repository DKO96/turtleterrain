#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class PoseSubscriber(Node):

    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10)
        self.subscription  # prevent unused variable warning

    def pose_callback(self, msg):
        self.get_logger().info('Current Pose: "%s"' % msg.pose.pose)
        self.get_logger().info(f'msg.pose.pose.position.x: {msg.pose.pose.position.x}')
        self.get_logger().info(f'msg.pose.pose.position.y: {msg.pose.pose.position.y}')
        self.get_logger().info(f'msg.pose.pose.position.z: {msg.pose.pose.position.z}')

def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = PoseSubscriber()
    rclpy.spin(pose_subscriber)
    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



