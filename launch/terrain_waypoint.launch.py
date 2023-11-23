#!usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtleterrain'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    nav2_bringup_dir = os.path.join(get_package_share_directory('nav2_bringup'))
    map_path = os.path.join(get_package_share_directory('turtleterrain'), 'map', 'flat_tree_terrain.yaml')
    param_path = os.path.join(get_package_share_directory('turtleterrain'), 'param', 'terrain.yaml') 
    rviz_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'rviz', 'nav2_default_view.rviz')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_dir = LaunchConfiguration('map', default=map_path)
    param_dir = LaunchConfiguration('params_file', default=param_path)

    # World and map files
    world = os.path.join(get_package_share_directory('turtleterrain'), 'worlds', 'turtlebot3_world.world')

    # Gazebo commands
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')))

    # Robot state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # RViz command
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Navigation stack
    # bringup_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([nav2_bringup_dir, '/bringup_launch.py']),
    #     launch_arguments={'map': map_dir,
    #                       'use_sim_time': use_sim_time,
    #                       'params_file': param_dir}.items())

    # bringup_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
    #     launch_arguments={'map': map_dir,
    #                       'use_sim_time': use_sim_time,
    #                       'params_file': param_dir}.items())
    bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
            launch_arguments={'map': map_dir}.items())
    
    # Demo autonomy task
    demo_cmd = Node(
        package='nav2_simple_commander',
        executable='example_waypoint_follower',
        emulate_tty=True,
        output='screen')

    # Launch description
    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(rviz_node)
    ld.add_action(bringup_cmd)
    # ld.add_action(demo_cmd)

    return ld
