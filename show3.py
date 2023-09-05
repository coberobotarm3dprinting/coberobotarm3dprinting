import numpy as np
from stl import mesh
from mayavi import mlab

your_mesh = mesh.Mesh.from_file('your_model.stl')

vertices = your_mesh.vectors.reshape(-1, 3)

fig = mlab.figure()
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], your_mesh.vectors, color=(0.8, 0.8, 0.8), opacity=1.0)

mlab.light(azimuth=45, elevation=45)

mlab.show()
