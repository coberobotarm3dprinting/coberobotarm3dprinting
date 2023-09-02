import math
import numpy as np
import numpy
from numpy import NaN, isnan
from stl.base import X, Y, Z
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.patches import Polygon
from scipy.spatial import distance
from random import randrange

import statprof

PATH_STEP = 0.1
START_DIRECTION = [0, 0, 1]

def show():
	# Create a new plot
	figure = pyplot.figure()
	axes = figure.add_subplot(projection='3d')

	# показать исходную поверхность
	#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cone.vectors))

	# show axises
	axes.plot([0, 1], [0,0],zs=[0,0], c="green")
	axes.plot([0, 0], [0,1],zs=[0,0], c="red")
	axes.plot([0, 0], [0,0],zs=[0,1], c="blue")
	axes.text(1,0,0,"X", c="green")
	axes.text(0,1,0,"Y", c="red")
	axes.text(0,0,1,"Z", c="blue")

	# add start polygon
	axes.plot([startPoligon[0], startPoligon[3]], [startPoligon[1],startPoligon[4]],zs=[startPoligon[2],startPoligon[5]], c="red")
	axes.plot([startPoligon[3], startPoligon[6]], [startPoligon[4],startPoligon[7]],zs=[startPoligon[5],startPoligon[8]], c="red")
	axes.plot([startPoligon[6], startPoligon[0]], [startPoligon[7],startPoligon[1]],zs=[startPoligon[8],startPoligon[2]], c="red")
	axes.scatter(startPoligon[0], startPoligon[1], startPoligon[2],  c="yellow")
	axes.scatter(startPoligon[3], startPoligon[4], startPoligon[5],  c="yellow")
	axes.scatter(startPoligon[6], startPoligon[7], startPoligon[8],  c="yellow")
	# add start vertex
	axes.scatter(startPoligon[startVertexIdx*3], startPoligon[startVertexIdx*3 + 1], startPoligon[startVertexIdx*3 + 2],  c="orange")

	# random points
	#for n in randomPoints:
	#	axes.scatter(n[X], n[Y], n[Z], c="green")

	# step points
	#for n in stepPoints:
	#	axes.scatter(n[X], n[Y], n[Z], c="green", s=1)

	# last successfull step points
	#for n in pointsByStep[-1]:
	#	axes.scatter(n[X], n[Y], n[Z], c="green", s=1)

	# left points (no more than 100)
	shown = 0
	for i in range(0, len(pointsLeft), 1 + int(len(pointsLeft)/100)):
		n = pointsLeft[i]
		axes.scatter(n[X], n[Y], n[Z], c="yellow", s=1)
		shown += 1
		if shown > 100:
			break

	# path points
	#for n in path:
	#	axes.scatter(n[X], n[Y], n[Z], c="blue", s=1)

	# path lines
	prev = [None,None,None]
	for n in path:
		if prev[0] != None:
			axes.plot([prev[X], n[X]], [prev[Y],n[Y]],zs=[prev[Z],n[Z]], c="blue")
		prev = n

	# Auto scale to the mesh size
	scale = cone.points.flatten()
	axes.auto_scale_xyz(scale, scale, scale)

	# Show the plot to the screen
	pyplot.show()

def getRandomPoint(polygon):
	# Получение случайной точки, равномерно распределенной в треугольнике ABC
	# через вычисление барицентрических координат
	# http://steps3d.narod.ru/snippets.html#rndtriangle
	a = randrange(0,1000) / 1000
	b = randrange(0,1000) / 1000
	if ( a + b > 1 ):
		a = 1 - a
		b = 1 - b
	c = 1 - a - b
	baricentric = numpy.array([a, b, c])

	# переводим в cartesian-координаты
	# https://stackoverflow.com/questions/56328254/how-to-make-the-conversion-from-barycentric-coordinates-to-cartesian-coordinates
	triangle = numpy.transpose(numpy.array([[polygon[0], polygon[1], polygon[2]],[polygon[3], polygon[4], polygon[5]],[polygon[6], polygon[7], polygon[8]]]))
	return triangle.dot(baricentric)

def getStepPoints(points, current, step):
	stepPoints = []
	left = []
	for idx, p in enumerate(points):
		# fast method
		if abs(p[X] - current[X]) <= step and abs(p[Y] - current[Y]) <= step and abs(p[Z] - current[Z]) <= step:
			stepPoints.append(p)

			# accurate method
			v = p - current
			if np.sqrt(v.dot(v)) > PATH_STEP:
				left.append(p)
		else:
			left.append(p)
	return (stepPoints, left)


def findBestPoint(stepPoints, prevPoint, gravityVector):
	def distance(p):
		v = p - prevPoint
		d = np.sqrt(v.dot(v))
		return d

	a = np.array([0,0,0])
	b = np.array(gravityVector)
	def gravity(p):
		ap = p-a
		ab = b-a
		projection = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
		magnitude = np.sqrt(projection.dot(projection))
		return magnitude

	def sortByDistAndGrav(p):
		return p[1]*0.5 - p[2]*0.5

	sortdata = []
	for idx, p in enumerate(stepPoints):
		sortdata.append([idx, distance(p), gravity(p)])

	#sortedByDistance = sorted(stepPoints, key=distance)
	#sortedByGravity = sorted(stepPoints, key=gravity)
	s = sorted(sortdata, key=sortByDistAndGrav)
	return stepPoints[s[-1][0]]

def removeByIndices(arr, indices):
	newarr = []
	for idx, p in enumerate(arr):
		if idx not in indices:
			newarr.append(p)
	return newarr

def getAngle(v1, v2):
	(x1,y1,z1) = v1
	(x2,y2,z2) = v2
	dot = x1*x2 + y1*y2 + z1*z2    # Between [x1, y1, z1] and [x2, y2, z2]
	lenSq1 = x1*x1 + y1*y1 + z1*z1
	lenSq2 = x2*x2 + y2*y2 + z2*z2
	val = dot/np.sqrt(lenSq1 * lenSq2)
	print("val", val)
	return math.acos(val)

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
print('start polygon idx: ', startPoligonIdx)
print('start polygon: ', startPoligon)
print('start vertex idx: ', startVertexIdx)
startPoint = [startPoligon[startVertexIdx*3], startPoligon[startVertexIdx*3 + 1], startPoligon[startVertexIdx*3 + 2]]
print('Start point: ', startPoint[X], startPoint[Y], startPoint[Z])
print('Start polygon normal: ', cone.normals[startPoligonIdx])

# Create path
path = []
path.append(startPoint)

randomPoints = []
for idx, polygon in enumerate(cone.points):
	for n in range(1,5000):
		point = getRandomPoint(polygon)
		randomPoints.append(point)


#statprof.start()

currentPoint = startPoint
prevPoint = [startPoint[X] - START_DIRECTION[X], startPoint[Y] - START_DIRECTION[Y], startPoint[Z] - START_DIRECTION[Z]]
gravityVector = [0, -1, 0]
pointsLeft = np.array(randomPoints)

pointsByStep = []
iteration = 0
while len(pointsLeft) > 0:
	print("iteration: ", iteration)
	print("Points left: ", len(pointsLeft))
	(stepPoints, left) = getStepPoints(pointsLeft, currentPoint, PATH_STEP)
	
	step_size = PATH_STEP
	while len(stepPoints) == 0:
		print("No step points. Increase step")
		step_size = step_size * 1.1
		(stepPoints, left) = getStepPoints(pointsLeft, currentPoint, step_size)

	#pointsByStep.append(stepPoints) # for showing
	currentPoint = findBestPoint(stepPoints, prevPoint, gravityVector)
	надо дорабатывать findBestPoint(): вносить туда параметр угла от предыдущей точки. плюс понаблюдать за вычислением гравитации
	if len(path) > 1:
		angle = getAngle(prevPoint - path[-2], prevPoint - currentPoint)
		if angle < np.pi/2:
			print("angle: ", angle, "try increase step")
			while angle < np.pi/2 and step_size < PATH_STEP*2:
				step_size = step_size * 1.1
				(stepPoints, left1) = getStepPoints(pointsLeft, currentPoint, step_size)
				newCurrentPoint = findBestPoint(stepPoints, prevPoint, gravityVector)
				angle = getAngle(prevPoint - path[-2], prevPoint - newCurrentPoint)
				print("angle", angle)
			if angle > np.pi/2:
				print("Best value found, angle:", angle)
				currentPoint = newCurrentPoint


	# remove nearest points
	pointsLeft = left

	path.append(currentPoint)
	prevPoint = currentPoint
	iteration += 1


#statprof.stop()
#statprof.display()

show()


