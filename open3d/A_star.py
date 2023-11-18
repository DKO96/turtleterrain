import numpy as np
import matplotlib.pyplot as plt
import heapq
from sklearn.neighbors import NearestNeighbors


class Node():
    def __init__(self, data, cost=0, heuristic=0):
        self.data = data
        self.cost = cost
        self.heuristic = heuristic
    
    @property
    def total_cost(self):
        return self.cost + self.heuristic
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

def get_neighbours(pcd, nn, current, k):
    current_reshaped = np.array(current).reshape(1,-1)
    distances, indices = nn.kneighbors(current_reshaped, n_neighbors=k+1)
    return pcd[indices[0][1:]]
    

def reconstruct_path(came_from, current):
    total_path = [current]
    while tuple(current) in came_from:
        current = came_from[tuple(current)]
        total_path.append(current)
    total_path.reverse()
    return total_path

def a_star(pcd, nn, start, goal):
    open_set = []
    start_node = Node(start, cost=0, heuristic=(start, goal))
    heapq.heappush(open_set, start_node)

    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): euclidean_dist(start, goal)}

    while open_set:
        current_node = heapq.heappop(open_set) 
        current = current_node.data

        if np.array_equal(current, goal):
            return reconstruct_path(came_from, current)
        
        for neighbour in get_neighbours(pcd, nn, current, k=5):
            neighbour_tuple = tuple(neighbour)
            tentative_g_score = g_score[tuple(current)] + euclidean_dist(current, neighbour)

            if neighbour_tuple not in g_score or tentative_g_score < g_score[neighbour_tuple]:
                came_from[neighbour_tuple] = tuple(current)
                g_score[neighbour_tuple] = tentative_g_score
                f_score[neighbour_tuple] = tentative_g_score + euclidean_dist(neighbour, goal)
                heapq.heappush(open_set, Node(neighbour, cost=g_score[neighbour_tuple], heuristic=f_score[neighbour_tuple]))

    return False


def main():
    # pcd = np.loadtxt('surface_points.xyz')

    # test
    np.random.seed(0)
    x = np.random.rand(10000) * 100
    y = np.random.rand(10000) * 100
    pcd = np.vstack((x,y)).T
    pcd[0] = np.array([0,0])
    pcd[-1] = np.array([100,100])
    print(pcd)

    nn = NearestNeighbors(algorithm='auto').fit(pcd)

    start = tuple(pcd[0])
    goal = tuple(pcd[-1]) 

    path = a_star(pcd, nn, start, goal)
    print(path)

     # Plotting the point cloud
    plt.figure(figsize=(9, 16))
    plt.scatter(pcd[:, 0], pcd[:, 1], color='blue', marker='o', label='Point Cloud')

    if path:
        # Converting the path into a format suitable for plotting
        path_points = np.array(path)
        plt.plot(path_points[:, 0], path_points[:, 1], color='red', linewidth=2, label='Path')

    plt.title("A* Pathfinding")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()



