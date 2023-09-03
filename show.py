import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

# Загрузка STL модели
#your_mesh = mesh.Mesh.from_file('cone_ascii.stl')
your_mesh = mesh.Mesh.from_file('cone_bin.stl')


# Создание 3D графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Итерация по треугольникам в STL модели
for triangle in your_mesh.vectors:
    # Создание трех вершин треугольника
    vertices = [list(vertex) for vertex in triangle]
    
    # Построение треугольника
    poly3d = [[vertices[0], vertices[1], vertices[2]]]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# Настройка осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Отображение графика
plt.show()

