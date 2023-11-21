import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
from sklearn.neighbors import NearestNeighbors

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

def neighbourSearch(node, nn, points):
    distances, indices = nn.kneighbors(node.coordinate.reshape(1,-1))
    indices = indices.reshape(-1)
    return indices[indices != np.where((points == node.coordinate).all(axis=1))[0][0]]

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


def main():
    # test set
    # samples = 10000
    # np.random.seed(0)
    # x = np.random.rand(samples) * 1000
    # y = np.random.rand(samples) * 1000
    # z = np.random.rand(samples) * 1000
    # points_xy = np.vstack((x,y))
    # points = np.vstack((points_xy,z)).T
    # points[0] = np.array([0,0,0])
    # points[-1] = np.array([1000,1000,1000])

    points = np.loadtxt('surface_points_tensor.xyz')
    print(f'# of points: {len(points)}')

    start = points[0] 
    end = points[-1]

    start_time = time.time()
    nn = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(points) 
    path = bidirectional_a_star(start, end, nn, points)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    if path:
        # Converting the path into a format suitable for plotting
        path_points = np.array(path)
    
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot for the points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', marker='o', label='Point Cloud')
    
        # Line plot for the path
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', linewidth=2, label='Path')
    
        ax.set_title("3D A* Pathfinding")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.legend()
        ax.grid(True)
        plt.show()



if __name__ == "__main__":
    main() 