import numpy as np
from stl import mesh
from mayavi import mlab

# Загрузка STL модели
your_mesh = mesh.Mesh.from_file('your_model.stl')

# Получение координат вершин треугольников
vertices = your_mesh.vectors.reshape(-1, 3)

# Отображение модели с настройкой освещения и невидимых поверхностей
fig = mlab.figure()
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], your_mesh.vectors, color=(0.8, 0.8, 0.8), opacity=1.0)

# Настройка освещения
mlab.light(azimuth=45, elevation=45)

# Отображение графика
mlab.show()
