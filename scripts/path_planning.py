#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import heapq
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline

class Node:
    def __init__(self, G=0, H=0, coordinate=None, parent=None):
        self.coordinate = coordinate
        self.parent = parent
        self.G = G
        self.H = H
        self.F = G + H

    def __lt__(self, other):
        return self.F < other.F

def heuristic(node, goal):
    return np.linalg.norm(node - goal)

def gCost(fixed_node, update_node):
    gcost = fixed_node.G + heuristic(fixed_node.coordinate, update_node.coordinate)  
    return gcost
    
def reconstructPath(forward_node, backward_node, start_node, goal_node):
    current, start, start_path = forward_node, start_node, []
    while current != start_node:
        start_path.append(current.coordinate) 
        current = current.parent
    start_path.append(start.coordinate)
    start_path.reverse() 

    current, goal, goal_path = backward_node, goal_node, []
    while current != goal_node:
        goal_path.append(current.coordinate) 
        current = current.parent
    goal_path.append(goal.coordinate)

    return start_path + goal_path[1:]

def neighbourSearch(node, nn, points, threshold=1e-4):
    distances, indices = nn.kneighbors(node.coordinate.reshape(1,-1))
    indices = indices.reshape(-1)
    return indices

def path_update(open_heap, node_dict, close_set, current_node, target_coord, points, nn):
    neighbours = neighbourSearch(current_node, nn, points)
    for n_idx in neighbours:
        neighbour_coord = points[n_idx]
        neighbour_node = Node(coordinate=neighbour_coord, 
                              H=heuristic(neighbour_coord, target_coord),
                              parent=current_node)
        neighbour_node.G = gCost(current_node, neighbour_node)

        if tuple(neighbour_coord) in close_set:
            continue

        if tuple(neighbour_coord) not in node_dict or \
            neighbour_node.G < node_dict[tuple(neighbour_node.coordinate)].G:
            heapq.heappush(open_heap, neighbour_node)
            node_dict[tuple(neighbour_coord)] = neighbour_node
    return open_heap, node_dict, close_set

def bidirectional_a_star(start, end, nn, points):
    origin = Node(coordinate=start, H=heuristic(start, end))
    goal = Node(coordinate=end, H=heuristic(end, start))

    # from start to goal
    origin_open_heap = []
    heapq.heappush(origin_open_heap, origin)
    origin_dict = {}
    origin_dict[tuple(origin.coordinate)] = origin
    origin_close = set()

    # from goal to start
    goal_open_heap = []
    heapq.heappush(goal_open_heap, goal)
    goal_dict = {}
    goal_dict[tuple(goal.coordinate)] = goal
    goal_close = set()

    while origin_open_heap and goal_open_heap:
        # from start to goal
        forward_current_node = heapq.heappop(origin_open_heap)
        forward_current_coord = tuple(forward_current_node.coordinate)

        if forward_current_coord in origin_dict and \
            forward_current_node.G <= origin_dict[forward_current_coord].G:
        
            origin_close.add(forward_current_coord)

            if forward_current_coord in goal_close:
                backward_node = goal_dict[forward_current_coord]
                return reconstructPath(forward_current_node, backward_node, origin, goal)
            
            origin_open_heap, origin_dict, origin_close = path_update(origin_open_heap, origin_dict,
                                                                      origin_close, forward_current_node,
                                                                      goal.coordinate, points, nn)

        # from goal to start
        backward_current_node = heapq.heappop(goal_open_heap)
        backward_current_coord = tuple(backward_current_node.coordinate)

        if backward_current_coord in goal_dict and \
            backward_current_node.G <= goal_dict[backward_current_coord].G:
        
            goal_close.add(backward_current_coord)

            if backward_current_coord in origin_close:
                forward_node = origin_dict[backward_current_coord]
                return reconstructPath(forward_node, backward_current_node, origin, goal)
        
            goal_open_heap, goal_dict, goal_close = path_update(goal_open_heap, goal_dict,
                                                                goal_close, backward_current_node,
                                                                origin.coordinate, points, nn)
    return None

def cubicSplineSmoother(path, num_points):
    path_arr = np.array(path)
    x = path_arr[:,0]
    y = path_arr[:,1]
    z = path_arr[:,2]
    t = np.arange(len(path))

    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    spline_z = CubicSpline(t, z)

    t_reduced = np.linspace(t.min(), t.max(), num_points)
    waypoints = np.column_stack((
        spline_x(t_reduced),
        spline_y(t_reduced),
        spline_z(t_reduced)
    ))
    return waypoints[1:]

def PathPlanner(points, start, end, target):
    nn = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(points) 
    path = bidirectional_a_star(start, end, nn, points)

    print(f'path: {path}')

    num_points = 30
    waypoints = cubicSplineSmoother(path, num_points)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.array([1,1,1])

    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    points_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    def create_sphere_at_point(point, radius, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        return sphere
    
    sphere_radius = 0.02  # Adjust the size of the spheres
    nearest_point_color = [0, 1, 0]
    robot_color = [0, 0, 1]
    path_color = [1, 0, 0]  # Red for path
    waypoint_color = [1, 0.5, 0]  # Orange for waypoints

    robot_sphere = create_sphere_at_point(start, sphere_radius, robot_color)
    nearest_point_sphere = create_sphere_at_point(end, sphere_radius, nearest_point_color)
    path_spheres = [create_sphere_at_point(p, sphere_radius, path_color) for p in path]
    waypoint_spheres = [create_sphere_at_point(p, sphere_radius, waypoint_color) for p in waypoints]

    vis.add_geometry(points_pcd)
    for sphere in path_spheres:
        vis.add_geometry(sphere)
    for sphere in waypoint_spheres:
        vis.add_geometry(sphere)
    vis.add_geometry(robot_sphere)
    vis.add_geometry(nearest_point_sphere)

    # Run the visualizer
    vis.run()

    # Close the visualizer window
    vis.destroy_window()



    return np.asarray(waypoints)









