"""
barcode generator
"""

"""
scripts for TDA course
"""
#import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3

from scipy import spatial as sptl
from scipy import special as spcl

import itertools as iter
import time

#need linux/macOS
import dionysus as d

"""
Everything to construct an alpha complex
"""
def check_if_subsimp(simp1,simp2):
	l = len(simp1)
	if l > len(simp2):
		return 0
	else:
		for i in range(len(simp2)):
			if (i+l)>len(simp2):
				test_simp = simp2[i:] + simp2[:(i+l)%len(simp2)]
			else:
				test_simp = simp2[i:i+l]
			if (test_simp == simp1) | (test_simp == simp1[::-1]):
				if len(simp2) == len(simp1):
					return 2
				return 1
		return 0

#check if a given simp is in a filtration
def check_if_simp_in_filtration(simp1,filtration):
	for i in range(len(filtration)):
		if check_if_subsimp(simp1,filtration[i])>1:
			return i
	return -1

#return subfaces of a simplex of a given dimension
def get_subfaces(simp,face_dim):
	return [list(s) for s in set(iter.combinations(simp, face_dim+1))]

#sort simps by length
def sort_simps_by_length(simp_list):
	simp_lengths = [len(simp) for simp in simp_list]

	return list(np.array(simp_list)[np.argsort(simp_lengths)])
#computes determinants for incircle business
#points = [(x1,y1,z1),(x2,y2,z2),..]
#or
#points = [(x1,y1),(x2,y2),...]
def Dxyz(points):
	if len(points[0]) == 2:
		a = np.linalg.det([[pt[0],pt[1],1] for pt in points])
		c = -np.linalg.det([[pt[0]**2+pt[1]**2, pt[0],pt[1]] for pt in points])
		Dx = -np.linalg.det([[pt[0]**2+pt[1]**2,pt[1],1] for pt in points])
		Dy = np.linalg.det([[pt[0]**2+pt[1]**2,pt[0],1] for pt in points])
		r = np.sqrt(Dx**2+Dy**2 - 4*a*c)/(2*np.abs(a))
		return np.array([-Dx/(2*a),-Dy/(2*a)]), r
	if len(points[0]) == 3:
		a = np.linalg.det([[pt[0],pt[1],pt[2],1] for pt in points])
		c = np.linalg.det([[pt[0]**2+pt[1]**2+pt[2]**2, pt[0],pt[1],pt[2]] for pt in points])
		Dx = np.linalg.det([[pt[0]**2+pt[1]**2+pt[2]**2,pt[1],pt[2],1] for pt in points])
		Dy = -np.linalg.det([[pt[0]**2+pt[1]**2+pt[2]**2,pt[0],pt[2],1] for pt in points])
		Dz = np.linalg.det([[pt[0]**2+pt[1]**2+pt[2]**2,pt[0],pt[1],1] for pt in points])
		r = np.sqrt(Dx**2+Dy**2+Dz**2 - 4*a*c)/(2*np.abs(a))
		return np.array([Dx/(2*a),Dy/(2*a),Dz/(2*a)]),r
	else:
		return np.array([0 for i in points[0]]), 0

#given points, compute the circumcenter and circumradius
def circumData(points):
	num_pts = len(points)

	#if one point, center is the point, radius 0
	if num_pts == 1:
		return points[0], 0
	#if 2 points, center is average, radius is 1/2 distance
	elif num_pts == 2:
		return (points[0]+points[1])/2, np.linalg.norm(points[0]-points[1])/2
	#otherwise, 3 points in 2d (maximal for 2d)
	elif len(points[0]) == 2:
		return Dxyz(points)
	#otherwise, 3 or 4 points in 3d
	#old conditional: if len(points[0]) == 3
	else:
		#if 3 points
		if len(points) == 3:
			#compute barycentric coordinates
			#origin at point[0]
			pt1 = np.array([0,0])
			#line from point[0] to point[1] is x-axis
			#orthogonal basis vectors
			x_basis = (1/np.linalg.norm(points[1]-points[0]))*(points[1]-points[0])
			y_basis_temp = (points[2] - points[0]) - np.dot(x_basis,points[2] - points[0])*x_basis
			y_basis = (1/np.linalg.norm(y_basis_temp))*y_basis_temp
			pt2 = np.array([np.linalg.norm(points[1]-points[0]),0])
			#pt3 is
			pt3 = np.array([np.dot(points[2]-points[0],x_basis), np.dot(points[2]-points[0],y_basis)])
			localDxyz = Dxyz([pt1, pt2, pt3])
			return points[0] + localDxyz[0][0]*x_basis + localDxyz[0][1]*y_basis, localDxyz[1]
		#otherwise maximal case
		else:
			return Dxyz(points)

#check if a given simplex is Gabriel, i.e. if the circumcircle contains any points of data
def is_Gabriel(data,data_kdtree,query_simp,query_circumData):
	query_result = data_kdtree.query(query_circumData[0])[1]
	if query_result in query_simp:
		return True
	else:
		return False

def construct_alpha_complex(data,tol = 0.0005):
	"""
	# input: data an n x d array of n points in dim d
	# output: simps the list of simplices in the alpha filtration sorted by bt, bt the list of birthtimes of all facets sorted
	#description: computes an alpha complex filtration by way of the Delaunay triangulation for points in R^d.
	"""

	num_pts = data.shape[0]
	dim_pts = data.shape[1]

	#initialize data structures
	birth_times = []
	simp_list = []

	#construct delaunay complex
	dlny = sptl.Delaunay(data)
	kdtree = sptl.KDTree(data)

	#filter top dimensional simplices into all possible faces

	#set up data structure
	all_simps = [[] for  i in range(dim_pts +1)]

	#get all top dimensional faces
	for simp in dlny.simplices:
		simp_circum = circumData([data[i] for i in simp])
		all_simps[-1].append([simp,simp_circum[0],simp_circum[1],True])
	#sort through top faces, get subfaces and determine attachedness
	for i in range(1,dim_pts+1)[::-1]:
		#get all simps in one level
		for simp_list in all_simps[i]:
			#get all subfaces
			for face in get_subfaces(simp_list[0],i-1):
				#print("face is: ")
				#print(face)
				#sort through next level
				#if nothing at the next level, add face and values
				if len(all_simps[i-1]) == 0:
					face_circum = circumData([data[k] for k in face])
					gabriel_bool = is_Gabriel(data, kdtree, face,face_circum)
					if gabriel_bool:
						all_simps[i-1].append([face,face_circum[0],np.min([face_circum[1],simp_list[2]]),gabriel_bool])
					else:
						all_simps[i-1].append([face,face_circum[0],simp_list[2],gabriel_bool])
				#otherwise there are faces already picked out
				else:
					#run through all faces already picked out
					for j in range(len(all_simps[i-1])):
						#print("checking against face:")
						#print(all_simps[i-1][j])
						#check if face matches what's there
						if check_if_subsimp(face,all_simps[i-1][j][0])>1:
							#if so, see if the whats-there is gabriel
							#if not gabriel, alpha value is min of the competing faces
							if not all_simps[i-1][j][-1]:
								all_simps[i-1][j][-2] = np.min([all_simps[i-1][j][-2],simp_list[2]])
							#then break
							break
						#else at the end of the list
						elif j == len(all_simps[i-1])-1:
							face_circum = circumData([data[k] for k in face])
							gabriel_bool = is_Gabriel(data, kdtree, face,face_circum)
							if gabriel_bool:
								all_simps[i-1].append([face,face_circum[0],np.min([face_circum[1],simp_list[2]]),gabriel_bool])
							else:
								all_simps[i-1].append([face,face_circum[0],simp_list[2],gabriel_bool])

	#extract simps, alpha values
	simps = []
	bt = []

	for i in range(dim_pts+1)[::-1]:
		for simp_list in all_simps[i]:
			simps.append([idx for idx in simp_list[0]])
			bt.append(simp_list[-2])

	simps = np.array(simps)
	bt = np.array(bt)
	#finally, sort
	sorted_idx = np.argsort(bt)

	simps = list(simps[sorted_idx])
	bt = bt[sorted_idx]

	#get distinct birthtimes
	distinct_entries = [bt[0]]
	for i in range(1,len(bt)):
		if (bt[i] - distinct_entries[-1])> tol:
			distinct_entries.append(bt[i])

	#sort each non-distinct run
	for entry in distinct_entries:
		#get first idx of occurence
		offender_locations = np.abs(bt - entry)<= tol
		first_offender_idx = list(offender_locations).index(True)
		offenders_list = np.array(simps)[offender_locations]
		num_offenders = len(offenders_list)
		sorted_offenders = sort_simps_by_length(offenders_list)
		simps[first_offender_idx:first_offender_idx+num_offenders] = sorted_offenders

	return simps, bt

def sample_spherical(npoints, sphere_dim=2):
	"""
	samples npoints on a sphere of dimension sphere_dim by sampling from a normal distribution and then projecting points onto a sphere
	"""

	vec = np.random.randn(sphere_dim+1, npoints)

	for i in range(npoints):
		vec[:,i] /= np.linalg.norm(vec[:,i], axis=0)
	return np.transpose(vec)

#sample points
circ_data = sample_spherical(50,sphere_dim=1)
circ_data = [p+0.2*(np.random.rand(2) - 0.5) for p in circ_data]
circ_data = np.array(circ_data)
circ_alpha1 = construct_alpha_complex(circ_data)

circ_data2 = np.array(circ_data)
circ_data2[-1] += 0.7*(np.random.rand(2)-0.5)
circ_alpha2 = construct_alpha_complex(circ_data2)

#draw circ 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.scatter(circ_data[:,0],circ_data[:,1],c="k")
ax.scatter(circ_data[-1,0],circ_data[-1,1],c="b")
ax.set_aspect("equal")
plt.show()

#draw circ 2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.scatter(circ_data2[:,0],circ_data2[:,1],c="k")
ax.scatter(circ_data2[-1,0],circ_data2[-1,1],c="r")
ax.set_aspect("equal")
plt.show()

#get each barcode
#alpha filtration TDA
test_comp = circ_alpha1[0]
test_bts = circ_alpha1[1]
test_f = d.Filtration()
for j in range(len(test_comp)):
    test_f.append(d.Simplex(test_comp[j],test_bts[j]))
p = d.homology_persistence(test_f)
dgms1 = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms1[0])
d.plot.plot_diagram(dgms1[1])
plt.show()

#alpha filtration TDA
test_comp = circ_alpha2[0]
test_bts = circ_alpha2[1]
test_f = d.Filtration()
for j in range(len(test_comp)):
    test_f.append(d.Simplex(test_comp[j],test_bts[j]))
p = d.homology_persistence(test_f)
dgms2 = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms2[0])
d.plot.plot_diagram(dgms2[1])
plt.show()

d.bottleneck_distance(dgms1[0], dgms2[0])

#get the nearest neighbors for the perturbed points
nn1 = np.sort([np.linalg.norm(pt - circ_data[-1]) for pt in circ_data])
nn2 = np.sort([np.linalg.norm(pt - circ_data2[-1]) for pt in circ_data2])

#plot the alpha complex for circ_data1
last_index = 1
pts = circ_data
bts = circ_alpha1[1][:-last_index]
current_simps = circ_alpha1[0][:-last_index]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)

ax.set_aspect("equal")

#for each simp, draw
for s in current_simps:
    if len(s) == 2:
        ax.plot(pts[s][:,0],pts[s][:,1],c = "b")
    elif len(s) == 3:
        if 49 in s:
            t1 = plt.Polygon(pts[s], color="g")
        else:
            t1 = plt.Polygon(pts[s], color="r")
        fig.gca().add_patch(t1)


ax.scatter(pts[:,0],pts[:,1], c = "k")
ax.scatter(pts[-1,0],pts[-1,1], c = "b")
plt.show()

#plot the alpha complex for circ_data1
last_index = 1
pts = circ_data2
bts = circ_alpha2[1][:-last_index]
current_simps = circ_alpha2[0][:-last_index]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)



ax.set_aspect("equal")

#for each simp, draw
for s in current_simps:
    if len(s) == 2:
        ax.plot(pts[s][:,0],pts[s][:,1],c = "b")
    elif len(s) == 3:
        if 49 in s:
            t1 = plt.Polygon(pts[s], color="g")
        else:
            t1 = plt.Polygon(pts[s], color="r")
        fig.gca().add_patch(t1)

ax.scatter(pts[:,0],pts[:,1], c = "k")
ax.scatter(pts[-1,0],pts[-1,1], c = "r")
plt.show()
