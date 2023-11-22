#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class AmclSubscriber(Node):

    def __init__(self):
        super().__init__('amcl_pose_pub')
        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10)

        self.amcl_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            'amcl_position',
            10)

        self.last_pose = None
        self.new_pose = None
        self.timer = self.create_timer(0.1, self.timer_callback)



    def pose_callback(self, msg):
        # self.get_logger().info('Current Pose: "%s"' % msg.pose.pose)
        # self.get_logger().info(f'msg.pose.pose.position.x: {msg.pose.pose.position.x}')
        # self.get_logger().info(f'msg.pose.pose.position.y: {msg.pose.pose.position.y}')
        # self.get_logger().info(f'msg.pose.pose.position.z: {msg.pose.pose.position.z}')

        self.last_pose = msg
        self.new_pose = True

    def timer_callback(self):
        if self.last_pose is not None and not self.new_pose:
            # self.get_logger().info('Publishing stored Pose')
            self.amcl_publisher.publish(self.last_pose)

        self.new_pose = False


        



def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = AmclSubscriber()
    rclpy.spin(pose_subscriber)
    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



