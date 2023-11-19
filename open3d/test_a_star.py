import numpy as np
from sklearn.neighbors import NearestNeighbors

class Node:
    def __init__(self, G=0, H=0, coordinate=None, parent=None):
        self.coordinate = coordinate
        self.parent = parent
        self.G = G
        self.H = H
        self.F = G + H

    def reset_f(self):
        self.F = self.G + self.H

def euclidean_dist(node_coordinate, goal):
    return np.linalg.norm(node_coordinate - goal)

def nearest_neighbour(node, nodes, nn):
    # Using NearestNeighbors for efficient nearest neighbor search
    distances, indices = nn.kneighbors([node.coordinate], n_neighbors=1)
    return nodes[indices[0][0]]

def find_path(open_list, closed_list, goal, nn):
    new_open_list = open_list.copy()  # Create a copy for iteration
    for node in new_open_list:
        temp = nearest_neighbour(node, closed_list, nn)  # Pass nn for nearest neighbour search
        for element in temp:
            if element in closed_list:
                continue
            new_g = node.G + euclidean_dist(node.coordinate, element.coordinate)  # Accumulate cost from start
            if element in open_list:
                ind = open_list.index(element)
                if new_g < open_list[ind].G:
                    open_list[ind].G = new_g
                    open_list[ind].reset_f()
                    open_list[ind].parent = node
            else:
                ele_node = Node(coordinate=element, parent=node,
                                G=new_g, H=euclidean_dist(element.coordinate, goal))
                open_list.append(ele_node)
        open_list.remove(node)
        closed_list.append(node)
    return open_list, closed_list

def get_path(org_list, goal_list, coordinate):
    path = []
    # Trace back from the intersection point to the start
    current = [node for node in org_list if np.array_equal(node.coordinate, coordinate)][0]
    while current:
        path.append(current.coordinate)
        current = current.parent
    path.reverse()  # Reverse to start from the beginning

    # Trace from the intersection point to the goal
    current = [node for node in goal_list if np.array_equal(node.coordinate, coordinate)][0]
    current = current.parent  # Skip the intersection point as it's already added
    while current:
        path.append(current.coordinate)
        current = current.parent

    return path

def searching_control(start, end, nn):
    origin = Node(coordinate=start, H=euclidean_dist(start, end))
    goal = Node(coordinate=end, H=euclidean_dist(end, start))
    origin_open = [origin]
    origin_close = []
    goal_open = [goal]
    goal_close = []
    target_goal = end
    while True:
        origin_open, origin_close = find_path(origin_open, origin_close, target_goal, nn)
        if not origin_open:
            break
        target_origin = min(origin_open, key=lambda x: x.F).coordinate

        goal_open, goal_close = find_path(goal_open, goal_close, target_origin, nn)
        if not goal_open:
            break
        target_goal = min(goal_open, key=lambda x: x.F).coordinate

        if check_node_coincide(origin_close, goal_close):
            return get_path(origin_close, goal_close, target_goal)

    return None  # Return None if no path is found

def draw_control(org_closed, goal_closed, flag, start, end):
    stop_loop = 0  # stop sign for the searching
    org_closed_ls = node_to_coordinate(org_closed)
    org_array = np.array(org_closed_ls)
    goal_closed_ls = node_to_coordinate(goal_closed)
    goal_array = np.array(goal_closed_ls)
    path = None
    if flag == 0:
        node_intersect = check_node_coincide(org_closed, goal_closed)
        if node_intersect:  # a path is find
            path = get_path(org_closed, goal_closed, node_intersect[0])
            stop_loop = 1
            print('Path found!')
    elif flag == 1:  # start point blocked first
        stop_loop = 1
        print('There is no path to the goal! Start point is blocked!')
    elif flag == 2:  # end point blocked first
        stop_loop = 1
        print('There is no path to the goal! End point is blocked!')
    return stop_loop, path

def node_to_coordinate(node_list):
    coordinate_list = [node.coordinate for node in node_list]
    return coordinate_list

def check_node_coincide(close_ls1, close_ls2):
    cl1 = node_to_coordinate(close_ls1)
    cl2 = node_to_coordinate(close_ls2)
    intersect_ls = [node for node in cl1 if node in cl2]
    return intersect_ls

def main():
    samples = 10000
    np.random.seed(0)
    x = np.random.rand(samples) * 100  # Ensure x is an array of 'samples' random values
    y = np.random.rand(samples) * 100  # Ensure y is an array of 'samples' random values
    points = np.vstack((x, y)).T
    points[0] = np.array([0, 0])
    points[-1] = np.array([100, 100])

    start = points[0] 
    end = points[-1]

    nn = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(points) 

    path = searching_control(start, end, nn)
    print(path)

    # Plotting the point cloud
    # plt.figure(figsize=(9, 16))
    # plt.scatter(points[:, 0], points[:, 1], color='blue', marker='o', label='Point Cloud')
    # plt.title("A* Pathfinding")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
