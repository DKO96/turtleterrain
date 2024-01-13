# Terrain Assessment Algorithm for Autonomous Robot Navigation

## Project Overview

This project introduces a terrain assessment algorithm tailored for autonomous robot navigation, primarily developed for educational purposes as part of a class project. It processes raw 3D point cloud data obtained from a LiDAR sensor for terrain assessment, which bypasses the need for conventional surface reconstruction or discretization methods.

### Key Features
- **Efficient Data Handling:** Incorporates voxel downsampling to enhance processing efficiency.
- **Terrain Accuracy:** Utilizes plane fitting with outlier rejection to ensure accurate terrain mapping.
- **Obstacle Awareness:** Employs boundary detection techniques for obstacle recognition.
- **Localization:** Integrates point cloud transformation for accurate robot positioning.
- **Path Planning:** Utilizes the Bidirectional A* algorithm for its effectiveness in graph-like structures.

## Simulation and Testing
The algorithm was tested in a simulated environment using the TurtleBot3 robot model, with ROS2 in the Gazebo Simulator. The tests focused on two scenarios: navigation over variable inclines and maneuvering through obstacle-laden trails, to assess uneven terrain navigation and obstacle avoidance capabilities.

## Challenges and Learning Outcomes
While the algorithm showed potential in navigating complex terrains, several challenges were encountered:
- Navigation stack latency.
- Need for nuanced obstacle detection.
- Occasional boundary detection inaccuracies.
- Requirement for a more effective path smoother.

These challenges have been valuable for learning and understanding the complexities of autonomous navigation systems.

## Educational Purpose
This project was developed for learning purposes and as part of a class project. As such, it may not be fully functional.
