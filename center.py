from stl import mesh
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#your_mesh = mesh.Mesh.from_file('Models\siege.stl')
your_mesh = mesh.Mesh.from_file('cone_ascii.stl')
center_of_mass = your_mesh.points.mean(axis=0)
vertices = your_mesh.points
hull = ConvexHull(vertices)
center_of_hull = np.mean(vertices[hull.vertices], axis=0)
central_axis = center_of_hull - center_of_mass
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(mesh.Poly3DCollection(your_mesh.vectors, alpha=0.5, linewidths=0.2, edgecolors='k'))
ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
          central_axis[0], central_axis[1], central_axis[2], color='r', label='Central Axis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()



