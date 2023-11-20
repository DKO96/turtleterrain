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
    
def neighbourSearch(node, nn, points):
    distances, indices = nn.kneighbors(node.coordinate.reshape(1,-1))
    indices = indices.reshape(-1)
    return indices[indices != np.where((points == node.coordinate).all(axis=1))[0][0]]

def getPath(end_node, start_node):
    current, start, path = end_node, start_node, []
    while current is not None:
        path.append(current.coordinate) 
        current = current.parent
    path.append(start.coordinate)
    return list(reversed(path))

def a_star(start, end, nn, points):
    origin = Node(coordinate=start, H=heuristic(start, end))
    goal = Node(coordinate=end, H=heuristic(end, start))

    origin_open_heap = []
    heapq.heappush(origin_open_heap, origin)
    origin_open_dict = {}
    origin_open_dict[tuple(origin.coordinate)] = origin

    origin_close = set()

    while origin_open_heap:
        current_node = heapq.heappop(origin_open_heap)
        current_coord = tuple(current_node.coordinate)
        if current_coord not in origin_open_dict or current_node.G > origin_open_dict[current_coord].G:
            continue

        if np.array_equal(current_node.coordinate, goal.coordinate):
            return getPath(current_node, origin)
        
        origin_close.add(current_coord)

        neighbours = neighbourSearch(current_node, nn, points)
        for n_idx in neighbours:
            neighbour_coord = points[n_idx]
            neighbour_node = Node(coordinate=neighbour_coord, 
                                  H=heuristic(neighbour_coord, end),
                                  parent=current_node)
            neighbour_node.G = gCost(current_node, neighbour_node)

            if tuple(neighbour_coord) in origin_close:
                continue

            if tuple(neighbour_coord) not in origin_open_dict or neighbour_node.G < origin_open_dict[tuple(neighbour_node.coordinate)].G:
                heapq.heappush(origin_open_heap, neighbour_node)
                origin_open_dict[tuple(neighbour_coord)] = neighbour_node

    return None




def main():
    # test set
    samples = 10000
    np.random.seed(0)
    x = np.random.rand(samples) * 100
    y = np.random.rand(samples) * 100
    points = np.vstack((x,y)).T
    points[0] = np.array([0,0])
    points[-1] = np.array([100,100])

    start = points[0] 
    end = points[-1]

    start_time = time.time()
    nn = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(points) 
    path = a_star(start, end, nn, points)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    # print(path)

    

    if path:
        # Converting the path into a format suitable for plotting
        path_points = np.array(path)

        plt.figure(figsize=(9, 16))
        plt.scatter(points[:, 0], points[:, 1], color='blue', marker='o', label='Point Cloud')
        plt.plot(path_points[:, 0], path_points[:, 1], color='red', linewidth=2, label='Path')
        plt.title("Bidirectional A* Pathfinding")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    main() 