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


def VR_filter(symmat,max_dim = 2):
	"""
	construct a VR filter by hand (in a very naive way).
	inputs are:
	-symmat, a matrix containing the similarities/distances between points 
	-max_dim =2, the max dimension of simplices to out put
	"""
	#add all vertices
	simps = []
	bts = []

	num_pts = len(symmat)

	for i in range(num_pts):
		simps.append([i])
		bts.append(0.)

	#for each possible 1, 2, etc. simplex, figure out the birth time
	for d in range(1,max_dim+1):
		for t in iter.combinations(range(num_pts),d+1):
			#get the birth time of t
			bt = np.max(symmat[list(t)][:,list(t)])

			#if the birth time for a non-singleton is >0, add
			if bt > 0.:
				simps.append(list(t))
				bts.append(bt)

	#sort the final simplices
	##sort by simplex sizes
	simp_sizes = [len(t) for t in simps]
	sort_order = np.argsort(simp_sizes)

	simps = np.array(simps)[sort_order]
	bts = np.array(bts)[sort_order]

	##sort by birth times
	sort_order = np.argsort(bts)
	bts = bts[sort_order]
	simps = simps[sort_order]

	return simps, bts


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

def voxelizer(pts, dims = [],num_grid_pts = [],gauss_para = 1.):
	"""
	converts a point cloud to a voxel data set, using Gaussians (with sigma=gauss_para) at each sample point. inputs:
	-pts, the point cloud in R^n to voxelize
	"""
	#construct underlying lattice
	dim = len(pts[0])
	##if dims is empty, take boundaries to be componentwise maxs mins
	if len(dims) == 0:
		for i in range(dim):
			dims.append([np.min(pts[:,i]),np.max(pts[:,i])])

	## if grid points is empty, have uniform numbers of points (10 ?)
	if len(num_grid_pts) == 0:
		for i in range(dim):
			num_grid_pts.append(15)

	#construct the lattice
	grid = np.zeros(tuple(num_grid_pts))

	#for each lattice point, compute the light content
	if dim == 2:
		for t in iter.product(np.arange(0,num_grid_pts[0],1),np.arange(0,num_grid_pts[1],1)):
			c = np.array([dims[j][0] + (dims[j][1]-dims[j][0])*t[j]*(1./num_grid_pts[j]) for j in range(dim)])
			grid[t] = sum([np.exp(-sum((pt - c)**2)/gauss_para) for pt in pts])
	if dim == 3:
		for t in iter.product(np.arange(0,num_grid_pts[0],1),np.arange(0,num_grid_pts[1],1),np.arange(0,num_grid_pts[2],1)):
			c = np.array([dims[j][0] + (dims[j][1]-dims[j][0])*t[j]*(1./num_grid_pts[j]) for j in range(dim)])
			grid[t] = sum([np.exp(-sum((pt - c)**2)/gauss_para) for pt in pts])

	#sort centers of voxels by light content
	return grid


def filtration_animator(pts, simps, bts,save_panels = [],save_path="",save_title="",show_points = True,max_bt = 5.):
	"""
	animates a filtration given:
	-pts, the array of sample points
	-simps, the list of simplices 
	-bts, the list of birth times 
	"""
	#use data to get bounds on the plot
	dim = len(pts[0])

	dims = []
	for i in range(dim):
		dims.append([np.min(pts[:,i])-1.,np.max(pts[:,i])+1.])

	if dim ==2:
		#animate
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.set_xlim(dims[0][0],dims[0][1])
		ax.set_ylim(dims[1][0],dims[1][1])

		if show_points:
			ax.scatter(pts[:,0],pts[:,1])

		ax.set_aspect("equal")

		#wframe is an empty object to redraw the animation/plot over
		wframe = None

		#tframe is a text object to
		tframe = ax.text(dims[0][0],dims[1][1],"filtration at step %i, birth time %.3f"%(0,0.))

		#tstart
		tstart = time.time()

		#get the unique bts
		unique_bts = np.unique(bts)

		for i in range(len(unique_bts)):
			if unique_bts[i] <= max_bt:
				# If a line collection is already remove it before drawing.
				#if wframe:
				#	ax.collections.remove(wframe)

				#get the simplices <= current bt
				if i ==0:
					current_simps = simps[bts <= unique_bts[i]]
				else:
					current_simps = simps[(bts <= unique_bts[i])&(bts > unique_bts[i-1])]

				#for each simp, draw
				for s in current_simps:
					if len(s) == 2:
						ax.plot(pts[s][:,0],pts[s][:,1],c = "b")
					elif len(s) == 3:
						t1 = plt.Polygon(pts[s], color="r")
						fig.gca().add_patch(t1)


				t = unique_bts[i]

				tframe.set_text("filtration at step %i, birth time %.3f"%(i,t))
				#wframe = ax.plot_wireframe(xx, yy, vv, color = 'b')
				if i in save_panels:
					plt.savefig("%s%s_panel_%i"%(save_path,save_title,i))
				plt.pause(.01)

	elif dim == 3:
		#animate
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#ax.set_aspect('equal')
		ax.set_xlim(dims[0][0],dims[0][1])
		ax.set_ylim(dims[1][0],dims[1][1])
		ax.set_zlim(dims[2][0],dims[2][1])

		if show_points:
			ax.scatter(pts[:,0],pts[:,1],pts[:,2],alpha = 0.2)
		ax.set_aspect("equal")
		#wframe = None
		tframe = ax.text(dims[0][0],dims[1][1],dims[2][1],"filtration at step %i, birth time %.3f"%(0,0.))

		tstart = time.time()

		#get the unique bts
		unique_bts = np.unique(bts)

		for i in range(len(unique_bts)):
			if unique_bts[i] <= max_bt:
				# If a line collection is already remove it before drawing.
				#if wframe:
				#	ax.collections.remove(wframe)

				#get the simplices <= current bt
				if i ==0:
					current_simps = simps[bts <= unique_bts[i]]
				else:
					current_simps = simps[(bts <= unique_bts[i])&(bts > unique_bts[i-1])]

				#for each simp, draw
				for s in current_simps:
					if len(s) == 2:
						ax.plot(pts[s][:,0],pts[s][:,1],pts[s][:,2])
					elif len(s) == 3:
						tri = a3.art3d.Poly3DCollection([pts[s]])
						tri.set_color("r")
						tri.set_edgecolor('k')
						tri.set_alpha(0.2)
						ax.add_collection3d(tri)


				t = unique_bts[i]

				tframe.set_text("filtration at step %i, birth time %.3f"%(i,t))
				#wframe = ax.plot_wireframe(xx, yy, vv, color = 'b')
				if i in save_panels:
					plt.savefig("%s%s_panel_%i"%(save_path,save_title,i))
				plt.pause(.01)

	return


def filtration_animator_cubical(voxel_vals,save_panels = [],save_path="",save_title="",min_bt=0.0):
	"""
	animates a 2d cubical filtration given:
	-voxel_vals, a 2d array with function values for eacch pixel of the array
	"""
	#animate
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#wframe is an empty object to redraw the animation/plot over
	wframe = None

	#get the unique bts
	unique_bts = np.unique(voxel_vals)

	#tframe is a text object to
	plt.title("filtration at step %i, birth time %.3f"%(0,unique_bts[-1]))

	#tstart
	tstart = time.time()

	for i in range(len(unique_bts)):
		if unique_bts[-i] >= min_bt:
			# If a line collection is already remove it before drawing.
			#if wframe:
			#	ax.collections.remove(wframe)

			ax.clear()

			ax.imshow(voxel_vals > unique_bts[-i])

			t = unique_bts[-i]

			plt.title("filtration at step %i, birth time %.3f"%(i,t))
			#wframe = ax.plot_wireframe(xx, yy, vv, color = 'b')
			if i in save_panels:
				plt.savefig("%s%s_panel_%i"%(save_path,save_title,i))
			plt.pause(.001)
	return

def sample_spherical(npoints, sphere_dim=2):
	"""
	samples npoints on a sphere of dimension sphere_dim by sampling from a normal distribution and then projecting points onto a sphere
	"""
	
	vec = np.random.randn(sphere_dim+1, npoints)
	
	for i in range(npoints):
		vec[:,i] /= np.linalg.norm(vec[:,i], axis=0)
	return np.transpose(vec)

def spherical_harmonic(x,n=5,m=3):
	"""
	this is a rudimentary wrapper for scipy's spherical harmonic functionality
	"""
	
	#convert x's to theta, phi
	#x should be an array of triples
	f_val = []

	for p in x:
		phi = np.arctan(p[1]/p[0])
		theta = np.arccos(p[2])

		f_val.append(spcl.sph_harm(m,n,theta,phi))

	return np.array(f_val)

def f_eval_simplices(f_vals,simps,increasing = True):
	"""
	this function evaluates a function on a list of simplices. f_val is assumed to be the function values on verticces, and the goal is to extend the function to simplices.
	"""
	simp_vals = []
	for s in simps:
		if increasing:
			f = np.max(f_vals[s])
			simp_vals.append(f)
		else:
			f = np.min(f_vals[s])
			simp_vals.append(f)

	if increasing:
		sort_order = np.argsort(simp_vals)
	else:
		sort_order = np.argsort(simp_vals)[::-1]

	return np.array(simps)[sort_order], np.array(simp_vals)[sort_order]

def rotation_matrix(axis, theta):
	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.
	"""
	axis = np.asarray(axis)
	axis = axis / np.sqrt(np.dot(axis, axis))
	a = np.cos(theta / 2.0)
	b, c, d = -axis * np.sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def generate_TDA_data(dim=2,rand_rotate = True, noise_para = 0.1):
	"""
	(crudely) generates the TDA dataset we've been looking at, in 2d or 3d
	"""
	#if dim = 2, generate a 37 x 2 array of zeros
	#if dim = 3, generate a 37 x 3 array of zeros
	TDA_pts = np.zeros((37,dim))

	#manually writing out where voxels are to be found
	row1idx = [1,2,3,4,5,10,11,12,22,23,24,25,26]
	for i in row1idx:
		TDA_pts[i-1][1] = 6.
	row2idx = [6,13,14,27,28]
	for i in row2idx:
		TDA_pts[i-1][1] = 5.
	row3idx = [7,15,16,29,30,31,32,33]
	for i in row3idx:
		TDA_pts[i-1][1] = 4.
	row4idx = [8,17,18,34,35]
	for i in row4idx:
		TDA_pts[i-1][1] = 3.
	row5idx = [9,19,20,21,36,37]
	for i in row5idx:
		TDA_pts[i-1][1] = 2.
	col1idx = [1]
	for i in col1idx:
		TDA_pts[i-1][0] = 1.
	col2idx = [2]
	for i in col2idx:
		TDA_pts[i-1][0] = 2.
	col3idx = [3,6,7,8,9]
	for i in col3idx:
		TDA_pts[i-1][0] = 3.
	col4idx = [4]
	for i in col4idx:
		TDA_pts[i-1][0] = 4.
	col5idx = [5]
	for i in col5idx:
		TDA_pts[i-1][0] = 5.
	col8idx = [10,13,15,17,19]
	for i in col8idx:
		TDA_pts[i-1][0] = 9.
	col9idx = [11,20]
	for i in col9idx:
		TDA_pts[i-1][0] = 10.
	col10idx = [12,21]
	for i in col10idx:
		TDA_pts[i-1][0] = 11.
	col11idx = [14,16,18]
	for i in col11idx:
		TDA_pts[i-1][0] = 12.
	col14idx = [22,27,29,34,36]
	for i in col14idx:
		TDA_pts[i-1][0] = 16.
	col15idx = [23,30]
	for i in col15idx:
		TDA_pts[i-1][0] = 17.
	col16idx = [24,31]
	for i in col16idx:
		TDA_pts[i-1][0] = 18.
	col17idx = [25,32]
	for i in col17idx:
		TDA_pts[i-1][0] = 19.
	col18idx = [26,28,33,35,37]
	for i in col18idx:
		TDA_pts[i-1][0] = 20.

	#if dim = 3, rotate to some place (?)
	if dim == 3:
		if rand_rotate:
			axis = np.random.rand(3)-0.5
			axis = axis/np.linalg.norm(axis)
			theta = np.random.rand()*2.*np.pi
			rot = rotation_matrix(axis, theta)
			TDA_pts = np.array([np.matmul(rot,TDA_pts[i]) for i in range(len(TDA_pts))])

	if noise_para>0.:
		TDA_pts = [p + noise_para*(np.random.rand(dim)-0.5) for p in TDA_pts]

	return np.array(TDA_pts)

def sample_image(pattern_num):

	s = None

	#diag 1
	if pattern_num == 0:
		s = np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
	if pattern_num == 1:
		s = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
	if pattern_num == 2:
		s = np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])
	if pattern_num == 3:
		s = np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
	if pattern_num == 4:
		s = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])

	#diag 2
	if pattern_num == 5:
		s = np.array([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
	if pattern_num == 6:
		s = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
	if pattern_num == 7:
		s = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
	if pattern_num == 8:
		s = np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
	if pattern_num == 9:
		s = np.array([[0.,0.,1.],[0.,0.,0.],[0.,0.,0.]])

	#horizontal
	if pattern_num == 10:
		s = np.array([[1.,1.,1.],[0.,0.,0.],[0.,0.,0.]])
	if pattern_num == 11:
		s = np.array([[0.,0.,0.],[1.,1.,1.],[0.,0.,0.]])
	if pattern_num == 12:
		s = np.array([[0.,0.,0.],[0.,0.,0.],[1.,1.,1.]])

	#horizontal
	if pattern_num == 13:
		s = np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
	if pattern_num == 14:
		s = np.array([[0.,1.,0.],[0.,1.,0.],[0.,1.,0.]])
	if pattern_num == 15:
		s = np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]])

	return s

def generate_image_data_simple(noise_para = 0.1):
	samples = []
	for i in range(16):
		s = sample_image(i)

		s = s + noise_para*np.random.rand(3,3)
		s = s/np.max(s)

		samples.append(s)

	return np.array(samples)

def generate_image_data(num_samples,noise_para = 0.1,add_images = False):
	samples = []
	for i in range(num_samples):
		#decide which shape to draw
		pattern_num = np.random.randint(16)

		s = sample_image(pattern_num)

		if add_images:
			if np.random.randint(2):
				s = s+ sample_image(np.random.randint(16))
				s = (s>0.1).astype("float")

		s = s + noise_para*np.random.rand(3,3)
		s = s/np.max(s)

		samples.append(s)

	return np.array(samples)

def compute_distmat(pts):
	"""
	computes the nxn distance matrix using Euclidean distance for n points
	"""
	num_pts = len(pts)
	dm = np.zeros((num_pts,num_pts))

	for i in range(num_pts-1):
		for j in range(i+1,num_pts):
			dm[i,j] = np.linalg.norm(pts[i]-pts[j])

	dm = dm+ dm.T

	return dm


def graph_embed(A,dim = 2):
	"""
	computes embedding coordinates for an arbitrary graph by using laplace eigenvectors.
	"""
	#construct the graph laplacian
	L = np.diag([sum(A[i,:]) for i in range(len(A))]) - A

	evals, evecs = np.linalg.eig(L)

	sort_order = np.argsort(evals)
	evals = evals[sort_order]
	evecs = evecs[:,sort_order]

	return evecs[:,1:(dim+1)], evals,evecs
