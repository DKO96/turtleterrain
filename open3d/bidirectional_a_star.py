import numpy as np
import matplotlib.pyplot as plt
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

def heuristic(node_coordinate, goal):
    return np.linalg.norm(node_coordinate - goal)

def gcost(fixed_node, update_node_coordinate):
    # move from start node to updated node
    gcost = fixed_node.G + heuristic(fixed_node.coordinate, update_node_coordinate)
    return gcost

def nearest_neighbour(node, nn, close_coordinate_list):
    node = node.coordinate.reshape(1, -1)
    distances, indices = nn.kneighbors(node)
    neighbours = [nn._fit_X[idx] for idx in indices[0] if nn._fit_X[idx] not in close_coordinate_list]
    return neighbours

def find_node_index(coordinate, node_list):
    ind = 0
    for node in node_list:
        if node.coordinate == coordinate:
            target_node = node
            ind = node_list.index(target_node)
            break
    return ind

def find_index_of_array(array_list, array_to_find):
    for i, array in enumerate(array_list):
        if np.array_equal(array, array_to_find):
            return i
    return -1  # Return -1 if the array is not found

def find_path(open_list, close_list, goal, nn):
    # searching for the path, update open and closed list
    flag = len(open_list)
    for i in range(flag):
        node = open_list[0]
        open_coordinate_list = [node.coordinate for node in open_list]
        close_coordinate_list = [node.coordinate for node in close_list]
        temp = nearest_neighbour(node, nn, close_coordinate_list)
        for element in temp:
            if element in close_list:
                continue
            elif any(np.array_equal(element, coord) for coord in open_coordinate_list):
                # if node in open list, update g value
                ind = find_index_of_array(open_coordinate_list, element)
                new_g = gcost(node, element)
                if new_g <= open_list[ind].G:
                    open_list[ind].G = new_g
                    open_list[ind].reset_f()
                    open_list[ind].parent = node
            else:   # new coordinate, create corresponding node
                ele_node = Node(coordinate=element, parent=node,
                                G=gcost(node, element), H=heuristic(element, goal))
                open_list.append(ele_node)
        open_list.remove(node)
        close_list.append(node)
        open_list.sort(key=lambda x: x.F)
    return open_list, close_list

def node_to_coordinate(node_list):
    coordinate_list = [node.coordinate for node in node_list]
    return coordinate_list

def check_node_coincide(close_ls1, close_ls2):
    cl1 = node_to_coordinate(close_ls1)
    cl2 = node_to_coordinate(close_ls2)
    intersect_ls = [node for node in cl1 if any(np.array_equal(node, other_node) for other_node in cl2)]
    return intersect_ls

def get_path(org_list, goal_list, coordinate):
    pass

def draw_control(org_closed, goal_closed, flag, start, end):
    """
    control the plot process, evaluate if the searching finished
    flag == 0 : draw the searching process and plot path
    flag == 1 or 2 : start or end is blocked, draw the border line
    """
    stop_loop = 0  # stop sign for the searching
    org_closed_ls = node_to_coordinate(org_closed)
    org_array = np.array(org_closed_ls)
    goal_closed_ls = node_to_coordinate(goal_closed)
    goal_array = np.array(goal_closed_ls)
    path = None
    # if show_animation:  # draw the searching process
    #     draw(org_array, goal_array, start, end, bound)
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

def searching_control(start, end, nn):
    # initial origin node and end node
    origin = Node(coordinate=start, H=heuristic(start, end))
    goal = Node(coordinate=end, H=heuristic(end, start))
    # list for searching from origin to goal
    origin_open: list = [origin]
    origin_close: list = []
    # list for searching from goal to origin
    goal_open = [goal]
    goal_close: list = []
    # initial target
    target_goal = end
    # flag = 0 (not blocked) 1 (start point blocked) 2 (end point blocked)
    flag = 0
    path = None
    while True:
        # searching from start to end
        origin_open, origin_close = \
            find_path(origin_open, origin_close, target_goal, nn)
        if not origin_open:     # no path condition
            flag = 1    # origin node is blocked
            break
        # update target for searching from end to start
        target_origin = min(origin_open, key=lambda x: x.F).coordinate

        # searching from end to start
        goal_open, goal_close = \
            find_path(goal_open, goal_close, target_origin, nn)
        if not goal_open:     # no path condition
            flag = 2    # goal node is blocked
            break
        # update target for searching from start to end
        target_goal = min(goal_open, key=lambda x: x.F).coordinate
    
        # continue searching
        stop_sign, path = draw_control(origin_close, goal_close, flag, start, end)
        if stop_sign:
            break
    return path



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
    path = searching_control(start, end, nn)

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
