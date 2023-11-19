import numpy as np
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
    return indices

def getPath():
    pass

def a_star(start, end, nn, points):
    origin = Node(coordinate=start, H=heuristic(start, end))
    goal = Node(coordinate=end, H=heuristic(end, start))
    # nodes = {start: origin, end: goal}
    
    origin_open, origin_close = [], []
    heapq.heappush(origin_open, origin)

    while origin_open:
        current_node = heapq.heappop(origin_open)
        if np.array_equal(current_node.coordinate, goal.coordinate):
            return getPath(current_node)
        else:
            heapq.heappush(origin_close, current_node)
            neighbours = neighbourSearch(current_node, nn, points)
        
        for n_idx in neighbours[0]:
            neighbour_node = Node(coordinate=points[n_idx], 
                                  H=heuristic(points[n_idx], end),
                                  parent=current_node)
            neighbour_node.G = gCost(current_node, neighbour_node)
            print(f'n.G = {neighbour_node.G}') 



    pass


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

    nn = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(points) 
    path = a_star(start, end, nn, points)

if __name__ == "__main__":
    main() 