#!/usr/bin/env python3 
import numpy as np
import torch

def robotPointCloud(R=0.2):
    # a, b, c, d = plane_eqn

    np.random.seed(42)
    n = 6000
    r = R * np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi

    # x = robot_coord[0] + r * np.cos(theta) 
    # y = robot_coord[1] + r * np.sin(theta) 
    x = r * np.cos(theta) 
    y = r * np.sin(theta) 
    z = np.random.uniform(low=-0.2, high=-0.161, size=n) 
    # z = -(a * x + b * y + d) / c

    robot_points = np.vstack((x, y, z)).T
    return robot_points
    

if __name__ == "__main__":
    robotPointCloud(np.array([0.2, 0.2, 0.2]))