#!/usr/bin/env python3 
import rclpy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from robot_navigator import BasicNavigator, TaskResult


def create_custom_path(custom_path_coordinates, frame_id='map'):
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = rclpy.time.Time().to_msg()

    for coords in custom_path_coordinates:
        waypoint = PoseStamped()
        waypoint.header.frame_id = frame_id
        waypoint.pose.position.x = coords[0]
        waypoint.pose.position.y = coords[1]
        waypoint.pose.position.z = coords[2]
        waypoint.pose.orientation.w = 1.0  # Assuming no orientation change
        path.poses.append(waypoint)
    
    return path


def main():
    rclpy.init()

    navigator = BasicNavigator()

    path_coordinates = np.array([[1.0, 1.0, 0.0],
                                 [1.1, 1.1, 0.0],
                                 [1.2, 1.2, 0.0],
                                 [1.3, 1.3, 0.0],
                                 [1.4, 1.4, 0.0],
                                 [1.5, 1.5, 0.0],
                                 [1.6, 1.6, 0.0],
                                 [1.7, 1.7, 0.0],
                                 [1.8, 1.8, 0.0],
                                 [1.9, 1.9, 0.0],
                                 [2.0, 2.0, 0.0],
                                ])

    path = create_custom_path(path_coordinates)



    smoothed_path = navigator.smoothPath(path)
    print(f'smoothed_path: {smoothed_path}')

    # Follow path
    navigator.followPath(smoothed_path)

    i = 0
    while not navigator.isTaskComplete():
        ################################################
        #
        # Implement some code here for your application!
        #
        ################################################

        # Do something with the feedback
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            print(
                'Estimated distance remaining to goal position: '
                + '{0:.3f}'.format(feedback.distance_to_goal)
                + '\nCurrent speed of the robot: '
                + '{0:.3f}'.format(feedback.speed)
            )

    # Do something depending on the return code
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Goal succeeded!')
    elif result == TaskResult.CANCELED:
        print('Goal was canceled!')
    elif result == TaskResult.FAILED:
        print('Goal failed!')
    else:
        print('Goal has an invalid return status!')

    navigator.lifecycleShutdown()

    exit(0)


if __name__ == '__main__':
    main()