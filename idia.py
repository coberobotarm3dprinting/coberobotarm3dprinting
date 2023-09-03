from stl import mesh
import numpy as np
import math

your_mesh = mesh.Mesh.from_file('your_model.stl')

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
rotating_vector = np.array([1.0, 0.0, 0.0])  # Исходный вектор, который будет вращаться
step_height = 0.0  # Высота подъема по оси поворота

for angle_degrees in range(361):  # Углы от 0 до 360 градусов
    intersection_points = []

    for triangle in triangles:
        intersection = find_intersection(triangle, rotating_vector)
        if intersection is not None:
            intersection_points.append(intersection)

    all_intersection_points.append((angle_degrees, intersection_points))

    if angle_degrees % 360 == 0:
        step_height += 1.0
    rotating_vector = rotate_vector(rotating_vector, [0, 0, 1], 1)  # Поворачиваем на 1 градус по оси Z

    rotating_vector[2] += step_height

# all_intersection_points содержит список точек пересечения для каждого угла и вектор, который поворачивается и поднимается по оси поворота.
