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




