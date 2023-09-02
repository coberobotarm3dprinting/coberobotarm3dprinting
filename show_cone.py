import math
import numpy
from numpy import NaN, isnan
from stl.base import X, Y, Z
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.patches import Polygon
from scipy.spatial import distance

START_DIRECTION = Z
SHIFT_DIRECTION = Y
PATH_STEP = 0.3

def testInSurface(point, m, n):
	res = n[X]*(point[X] - m[X]) + n[Y]*(point[Y] - m[Y]) + n[Z]*(point[Z] - m[Z])
	#print('testInSurface: ', res)
	return res < 1.0e-8

def getNextPointY(point, normal, side, poligon, returnFirst):
	# если если плоскость задана уравнением 'Ax + By + Cz + D = 0', то вектор нормали к плоскости: n(A, B, C)
	# уравнение плоскости через точку M(x0,y0,z0) и нормаль n(n1,n2,n3): n1(x-x0) + n2(y-y0) + n3(z-z0) = 0
	# считаем, что обход начнем в плоскости Y: т.е. (y-y0) = 0
	# уравнение плоскости приводится к виду: n1(x-x0) + n3(z-z0) = 0
	# расстояние до следующей точки: sqrt((x-x0)^2 + (z-z0)^2) = PATH_STEP
	# для решения системы выражаем z через x: z = - n1(x-x0)/n3 + z0 = z0 - n1/n3(x-x0)
	# подставляем в уравнение расстояния (возведя обе стороны в квадрат):
	# (x-x0)^2 + (z-z0)^2 = PATH_STEP^2
	# (x-x0)^2 + (z0 - n1/n3(x-x0) - z0)^2 = PATH_STEP^2
	# (x-x0)^2 + (n1/n3)^2(x-x0)^2 = PATH_STEP^2
	# (x-x0)^2 = PATH_STEP^2/(1 + (n1/n3)^2)
	# x = PATH_STEP/sqrt((1 + (n1/n3)^2)) + x0

	y1 = point[Y]
	x1 = point[X] + PATH_STEP/math.sqrt(1 + pow(normal[X]/normal[Z],2))
	z1 = point[Z] + normal[X] / normal[Z] * (point[X] - x1)

	y2 = point[Y]
	x2 = point[X] - PATH_STEP/math.sqrt(1 + pow(normal[X]/normal[Z],2))
	z2 = point[Z] - normal[X] / normal[Z] * (point[X] - x1)

	# предварительный ответ
	x = x1
	y = y1
	z = z1

	# bug!
	if returnFirst:
		return [x, y, z]

	# найдём направление на третий угол
	if side != None:
		# print('last side:', side)
		v = [NaN, NaN, NaN]
		for i in range(3):
			# print('side[0]', side[0])
			# print('side[1]', side[1])
			# print('poligon', poligon[i*3 : i*3+3])
			if not (poligon[i*3 : i*3+3] == side[0]).all() and not (poligon[i*3 : i*3+3] == side[1]).all():
				v = poligon[i*3 : i*3+3]
				break
		# print('v:', v)
		if isnan(v[X]):
			raise(Exception('Third corner not found'))
		vecPV = [v[X] - point[X], v[Y] - point[Y], v[Z] - point[Z]]
		vecP1 = [x1 - point[X], y1 - point[Y], z1 - point[Z]]
		vecP2 = [x2 - point[X], y2 - point[Y], z2 - point[Z]]
		scal1 = vecPV[X]*vecP1[X] + vecPV[Y]*vecP1[Y] + vecPV[Z]*vecP1[Z]
		scal2 = vecPV[X]*vecP2[X] + vecPV[Y]*vecP2[Y] + vecPV[Z]*vecP2[Z]
		#print('scal1: ', scal1)
		#print('scal2: ', scal2)
		if scal2 > scal1:
			x = x2
			y = y2
			z = z2

	return [x, y, z]

def getNextPointZ(point, normal, direction):
	# если если плоскость задана уравнением 'Ax + By + Cz + D = 0', то вектор нормали к плоскости: n(A, B, C)
	# уравнение плоскости через точку M(x0,y0,z0) и нормаль n(n1,n2,n3): n1(x-x0) + n2(y-y0) + n3(z-z0) = 0
	# считаем, что обход начнем в плоскости Z: т.е. (z-z0) = 0
	# уравнение плоскости приводится к виду: n1(x-x0) + n2(y-y0) = 0
	# расстояние до следующей точки: sqrt((x-x0)^2 + (y-y0)^2) = PATH_STEP
	# для решения системы выражаем y через x: y = - n1(x-x0)/n2 + y0 = y0 - n1/n2(x-x0)
	# подставляем в уравнение расстояния (возведя обе стороны в квадрат):
	# (x-x0)^2 + (y-y0)^2 = PATH_STEP^2
	# (x-x0)^2 + (y0 - n1/n2(x-x0) - y0)^2 = PATH_STEP^2
	# (x-x0)^2 + (n1/n2)^2(x-x0)^2 = PATH_STEP^2
	# (x-x0)^2 = PATH_STEP^2/(1 + (n1/n2)^2)
	# x = PATH_STEP/sqrt((1 + (n1/n2)^2)) + x0

	z1 = point[Z]
	x1 = point[X] + PATH_STEP/math.sqrt(1 + pow(normal[X]/normal[Y],2))
	y1 = point[Y] + normal[X] / normal[Y] * (point[X] - x1)

	z2 = point[Z]
	x2 = point[X] - PATH_STEP/math.sqrt(1 + pow(normal[X]/normal[Y],2))
	y2 = point[Y] - normal[X] / normal[Y] * (point[X] - x1)

	# предварительный ответ
	x = x1
	y = y1
	z = z1

	# найдём направление на третий угол
	if direction != None:
		vecPV = direction
		vecP1 = [x1 - point[X], y1 - point[Y], z1 - point[Z]]
		vecP2 = [x2 - point[X], y2 - point[Y], z2 - point[Z]]
		scal1 = vecPV[X]*vecP1[X] + vecPV[Y]*vecP1[Y] + vecPV[Z]*vecP1[Z]
		scal2 = vecPV[X]*vecP2[X] + vecPV[Y]*vecP2[Y] + vecPV[Z]*vecP2[Z]
		# print('scal1: ', scal1)
		# print('scal2: ', scal2)
		if scal2 > scal1:
			x = x2
			y = y2
			z = z2

	return [x, y, z]

def isInside(point, vertices):
	# https://russianblogs.com/article/1726811329/
	v0 = vertices[2] - vertices[0]
	v1 = vertices[1] - vertices[0]
	v2 = point - vertices[0]
	dot00 = numpy.dot(v0,v0)
	dot01 = numpy.dot(v0,v1)
	dot02 = numpy.dot(v0,v2)
	dot11 = numpy.dot(v1,v1)
	dot12 = numpy.dot(v1,v2)
	inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)
	u = (dot11 * dot02 - dot01 * dot12) * inverDeno
	if u < -1.0e-8 or u > 1:
		# if u out of range, return directly
		#print('u out of range: ', u)
		return False

	v = (dot00 * dot12 - dot01 * dot02) * inverDeno
	if v < 0 or v > 1:
		# if v out of range, return directly
		#print('v out of range: ', v)
		return False

	return u + v <= 1


def getIntersection(A, B, C, D):
	# https://www.cyberforum.ru/geometry/thread2147227.html
	# ищем пересечение отрезков AB и CD
	s = ((A[Y] - C[Y]) + (B[Y] - A[Y])*((C[X] - A[X])/(B[X] - A[X])))/(D[Y] - C[Y] - (B[Y] - A[Y])*((D[X] - C[X])/(B[X] - A[X])))
	t = (C[X] - A[X])/(B[X] - A[X]) + s*(D[X] - C[X])/(B[X] - A[X])
	#print('s: ', s)
	#print('t: ', t)
	if s < 0 or s > 1 or t < 0 or t > 1:
		#print('return nan')
		return [NaN, NaN, NaN]

	return [C[X] + s*(D[X] - C[X]), C[Y] + s*(D[Y] - C[Y]), C[Z] + s*(D[Z] - C[Z])]

def getNextPoligon(obj, side, exceptIdx):
	for i in range(len(obj.v0)):
		if i == exceptIdx:
			continue
		one = (side[0] == obj.v0[i]).all() or (side[0] == obj.v1[i]).all() or (side[0] == obj.v2[i]).all()
		two = (side[1] == obj.v0[i]).all() or (side[1] == obj.v1[i]).all() or (side[1] == obj.v2[i]).all()
		if one and two:
			return i
	raise(Exception('Next poligon not found'))

# Load the STL files
cone = mesh.Mesh.from_file('cone_ascii.stl')
print('points:\n', cone.points)
print('vectors:\n', cone.vectors)
print('normals:\n', cone.normals)

# Find start point
yMin = 100.0
startPoligon = None
startPoligonIdx = None
startVertexIdx = None
for idx, pointSet in enumerate(cone.points):
	for vertexIdx in range(3):
		x, y, z = pointSet[vertexIdx * 3: vertexIdx * 3 + 3]
		if y <= yMin:
			yMin = y
			startPoligon = pointSet
			startPoligonIdx = idx
			startVertexIdx = vertexIdx

print('Y min: ', yMin)
print('start poligon idx: ', startPoligonIdx)
print('start vertex idx: ', startVertexIdx)
startPoint = [startPoligon[startVertexIdx*3], startPoligon[startVertexIdx*3 + 1], startPoligon[startVertexIdx*3 + 2]]
print('Start point: ', startPoint[X], startPoint[Y], startPoint[Z])
print('Start polygon normal: ', cone.normals[startPoligonIdx])

# Create path
path = []
path.append(startPoint)

currentPoint = startPoint
currentPoligonIdx = startPoligonIdx
prevPoligonIdx = None
lastSide = None
prevLastSide = None
stop = False
forceStop = False
intersectionNotFound = False
while not stop:
	print(f'=========== iteration {len(path)} ==============')
	n = getNextPointY(currentPoint, cone.normals[currentPoligonIdx], lastSide, cone.points[currentPoligonIdx], intersectionNotFound)
	#print('Next point: ', n)
	#print('Current poligon idx: ', currentPoligonIdx)
	if not testInSurface(n, currentPoint, cone.normals[currentPoligonIdx]):
		raise(Exception('Not in surface!'))

	oversize = not isInside(n, [cone.v0[currentPoligonIdx], cone.v1[currentPoligonIdx], cone.v2[currentPoligonIdx]])

	if oversize:
		print('Next point is out of triangle')
		# дальше ищем пересечение отрезков currentPoint-n и каждой из сторон полигона
		side = [cone.v0[currentPoligonIdx], cone.v1[currentPoligonIdx]]
		intersection = getIntersection(currentPoint, n, cone.v0[currentPoligonIdx], cone.v1[currentPoligonIdx])
		if isnan(intersection[X]):
			side = [cone.v1[currentPoligonIdx], cone.v2[currentPoligonIdx]]
			intersection = getIntersection(currentPoint, n, cone.v1[currentPoligonIdx], cone.v2[currentPoligonIdx])
		if isnan(intersection[X]):
			side = [cone.v2[currentPoligonIdx], cone.v0[currentPoligonIdx]]
			intersection = getIntersection(currentPoint, n, cone.v2[currentPoligonIdx], cone.v0[currentPoligonIdx])
		if isnan(intersection[X]):
			#raise Exception('Intersection not found')
			print('Intersection not found')
			if intersectionNotFound:
				break
			intersectionNotFound = True
			#path.append(n)
			continue
		print('Intersection found: ', intersection)
		n = intersection
		if not testInSurface(n, currentPoint, cone.normals[currentPoligonIdx]):
			raise(Exception('Not in surface!'))

		# change poligon
		prevPoligonIdx = currentPoligonIdx
		currentPoligonIdx = getNextPoligon(cone, side, currentPoligonIdx)
		prevLastSide = lastSide
		lastSide = side
		print('Next poligon Idx: ', currentPoligonIdx)

	# find closest point in previous path
	closest = None
	minDist = None
	for idx, p in enumerate(path):
		if idx == len(path) - 2:
			break
		p = path[idx]
		dist = distance.euclidean(n, p)
		if minDist == None or dist < minDist:
			minDist = dist
			closest = p
	print('Closest point distance: ', minDist)

	# если заехали туда, где уже были
	if oversize and minDist != None and minDist < PATH_STEP*0.1:
		# отодвинемся на следующий обход
		# двигаемся на STEP по SHIFT_DIRECTION = Y (тупо, но для прототипа сойдёт)
		# X оставляем тот же самый
		print('Too close - make shift')
		n = getNextPointZ(n, cone.normals[prevPoligonIdx], [0.0,1.0,1.0])
		if not testInSurface(n, currentPoint, cone.normals[prevPoligonIdx]):
			raise(Exception('Not in surface!'))
		currentPoligonIdx = prevPoligonIdx
		lastSide = prevLastSide
		#forceStop = True

	path.append(n)
	currentPoint = n
	stop = len(path) > 10000 or forceStop
	intersectionNotFound = False

path1 = []
for p in path:
	# увеличить масштаб в 100 раз
 	#path1.append([p[X] * 100, p[Y] * 100, p[Z] * 100])
 	path1.append([p[X], p[Y], p[Z]])

# Create a new plot
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

# показать исходную поверхность
#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cone.vectors))

# add start polygon
axes.plot([startPoligon[0], startPoligon[3]], [startPoligon[1],startPoligon[4]],zs=[startPoligon[2],startPoligon[5]], c="red")
axes.plot([startPoligon[3], startPoligon[6]], [startPoligon[4],startPoligon[7]],zs=[startPoligon[5],startPoligon[8]], c="red")
axes.plot([startPoligon[6], startPoligon[0]], [startPoligon[7],startPoligon[1]],zs=[startPoligon[8],startPoligon[2]], c="red")
axes.scatter(startPoligon[0], startPoligon[1], startPoligon[2],  c="yellow")
axes.scatter(startPoligon[3], startPoligon[4], startPoligon[5],  c="yellow")
axes.scatter(startPoligon[6], startPoligon[7], startPoligon[8],  c="yellow")
# add start vertex
axes.scatter(startPoligon[startVertexIdx*3], startPoligon[startVertexIdx*3 + 1], startPoligon[startVertexIdx*3 + 2],  c="orange")

for n in path1:
	# add path point
	axes.scatter(n[X], n[Y], n[Z], c="green")

# add last point
axes.scatter(path[len(path) - 1][X], path[len(path) - 1][Y], path[len(path) - 1][Z],  c="red")

# lastSide
#axes.scatter(lastSide[0][X], lastSide[0][Y], lastSide[0][Z],  c="orange")
#axes.scatter(lastSide[1][X], lastSide[1][Y], lastSide[1][Z],  c="orange")

# Auto scale to the mesh size
scale = cone.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
# масштаб * 100
#axes.auto_scale_xyz(scale*100, scale*100, scale*100)

# Show the plot to the screen
pyplot.show()


with open('path.csv', 'w') as f:
	for p in path1:
		f.write(f"{p[X]}, {p[Y]}, {p[Z]}, 0, -1, 0\n")


