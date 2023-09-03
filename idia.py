import math
from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


your_mesh = mesh.Mesh.from_file('cone_ascii.stl')
#your_mesh = mesh.Mesh.from_file('cone_ascii.stl')
#your_mesh.normals = mesh.Mesh.normals

min_x, min_y, min_z = np.min(your_mesh.v0, axis=0)
max_x, max_y, max_z = np.max(your_mesh.v0, axis=0)

range_x = max_x - min_x
range_y = max_y - min_y
range_z = max_z - min_z

for i in range(len(your_mesh.v0)):
    your_mesh.v0[i][0] = (your_mesh.v0[i][0] - min_x) / range_x
    your_mesh.v0[i][1] = (your_mesh.v0[i][1] - min_y) / range_y
    your_mesh.v0[i][2] = (your_mesh.v0[i][2] - min_z) / range_z

for i in range(len(your_mesh.v1)):
    your_mesh.v1[i][0] = (your_mesh.v1[i][0] - min_x) / range_x
    your_mesh.v1[i][1] = (your_mesh.v1[i][1] - min_y) / range_y
    your_mesh.v1[i][2] = (your_mesh.v1[i][2] - min_z) / range_z

for i in range(len(your_mesh.v2)):
    your_mesh.v2[i][0] = (your_mesh.v2[i][0] - min_x) / range_x
    your_mesh.v2[i][1] = (your_mesh.v2[i][1] - min_y) / range_y
    your_mesh.v2[i][2] = (your_mesh.v2[i][2] - min_z) / range_z


triangles = your_mesh.vectors

def find_intersection(triangle, vector):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    d = -np.dot(normal, triangle[0])
    if np.dot(normal, vector) == 0:
        return None
    t = -(np.dot(normal, triangle[0]) + d) / np.dot(normal, vector)
    if t < 0:
        return None
    intersection_point = vector * t
    return intersection_point

def is_inside_triangle(point, triangle):
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    n = np.cross(u, v)
    w = point - v0
    alpha = np.dot(u, u) * np.dot(w, v) - np.dot(u, v) * np.dot(w, u)
    beta = np.dot(u, u) * np.dot(w, v) - np.dot(u, v) * np.dot(w, v)
    gamma = np.dot(u, v) * np.dot(w, u) - np.dot(v, v) * np.dot(w, v)
    alpha /= np.dot(n, n)
    beta /= np.dot(n, n)
    gamma /= np.dot(n, n)
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1


def rotate_vector(vector, axis, angle):
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a + (1 - cos_a) * axis[0]**2, (1 - cos_a) * axis[0] * axis[1] - sin_a * axis[2], (1 - cos_a) * axis[0] * axis[2] + sin_a * axis[1]],
        [(1 - cos_a) * axis[1] * axis[0] + sin_a * axis[2], cos_a + (1 - cos_a) * axis[1]**2, (1 - cos_a) * axis[1] * axis[2] - sin_a * axis[0]],
        [(1 - cos_a) * axis[2] * axis[0] - sin_a * axis[1], (1 - cos_a) * axis[2] * axis[1] + sin_a * axis[0], cos_a + (1 - cos_a) * axis[2]**2]
    ])
    return np.dot(rotation_matrix, vector)

all_intersection_points = []
rotating_vector = np.array([1.0, 0.0, 0.0])  
step_height = 0.0  

for angle_degrees in range(361):  
    intersection_points = []

    for triangle in triangles:
        intersection = find_intersection(triangle, rotating_vector)
        if intersection is not None and is_inside_triangle(intersection,triangle):
            intersection_points.append(intersection)

    all_intersection_points.append((angle_degrees, intersection_points))

    if angle_degrees % 360 == 0:
        step_height += 1.0
    rotating_vector = rotate_vector(rotating_vector, [0, 0, 1], 1) 

    rotating_vector[2] += step_height


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for angle, intersection_points in all_intersection_points:
    if intersection_points:
        intersection_points = np.array(intersection_points)
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], label=f'Angle {angle}°')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

plt.show()




