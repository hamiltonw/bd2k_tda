"""
MAPPER 
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as iter

#find the pixel neighbors, i.e. neighboring integer lattice points + diagonals in the upper-right quadrant
def pixel_nbrs(pt):
	"""
	in: pt is an n-tuple of integers 
	out: list of adjacent (vertical, horizontal, diagonal) lattice points
	"""
	nbrs =[]
	
	#horizontals/verticals
	for i in range(len(pt)):
		for j in [-1,1]:
			if pt[i] + j >= 0:
				new_pt = np.array(pt)
				new_pt[i] += j
				nbrs.append(tuple(new_pt))
	#get the diagonals 
	for i in range(2**len(pt)):
		bi = "{0:b}".format(i + 2**len(pt))
		diag_pt = np.array([(-1.)**int(bi[k]) for k in range(1,len(bi))])
		if np.min(diag_pt+pt) >= 0:
			nbrs.append(tuple(diag_pt+pt))
	
	return nbrs

#given an adjacency matrix, find the neighboring points that are distance <= eps from a given point	
def geometric_nbrs(idx, dM,eps = 1.):
	"""
	in: -idx of pt under consideration
		-dM, distance matrix of point cloud 
		-eps, tolerance to choose neighbors 
	out: -idxs of nbrs 
	"""
	nbr_bool = dM[idx] <eps
	#don't want the same point as a neighbor.
	nbr_bool[idx] = False
	nbr_idx = np.arange(len(dM))[nbr_bool]
	
	return nbr_idx

#voxelized version
def fast_marching_CC(pts, pt_nbrs):
	"""
	in: -point list (should be indices)
		-point neighbors
	out: -list of connected components
	"""
	checked_array = np.zeros(len(pts))

	#preprocess pts, pt_nbrs
	#set up dictionary
	pt_dict = {}
	idx = 0
	for i in range(len(pts)):
		pt_dict[pts[i]] = idx
		idx += 1

	new_pt_nbrs = []
	#convert pt_nbrs to local idxs
	for nbrs in pt_nbrs:
		local_nbrs = []
		for nbr in nbrs:
			try:
				local_nbrs.append(pt_dict[nbr])
			except:
				pass
		new_pt_nbrs.append(local_nbrs)

	CCs = []
	while(sum(checked_array) < 2.*len(pts)):
		try:
			next_idx = np.arange(len(pts))[checked_array == 1.][0]
		except:
			next_idx = np.arange(len(pts))[checked_array == 0.][0]
			CCs.append([pts[next_idx]])


		#set the next_idx to have been checked
		checked_array[next_idx] = 2.

		#run through the neighbors of next_idx
		for nbr in new_pt_nbrs[next_idx]:
			if checked_array[nbr]<1.:
				CCs[-1].append(pts[nbr])
				checked_array[nbr] = 1.
	return CCs

def get_centroid(pt_list):
    cent = np.zeros(len(pt_list[0]))

    for pt in pt_list:
        for i in range(len(pt)):
            cent[i] += pt[i]
    cent /= len(pt_list)

    return cent
	
"""
pts = [0,1,2,3,4]

nbrs = [[1,2],[0,2],[0,1],[4],[3]]	

fast_marching_CC(pts,nbrs)
"""


def MAPPER(CCs):
	"""
	in: -list of connected components 
	out: -adjacency matrix of network
		- sizes of each CC
	"""
	adj = np.zeros((len(CCs),len(CCs)))
	sizes = np.zeros(len(CCs))
	
	for i in range(len(CCs)-1):
		sizes[i] = len(CCs[i])
		for j in range(i+1,len(CCs)):
			#if the two connected components intersect, add a 1 to the adjacency matrix
			if set(CCs[i]) & set(CCs[j]):
				adj[i,j] = 1.
	
	#don't forget the size of the last cc
	sizes[-1] = len(CCs[-1])
	
	#symmetrize the adjacency matrix 
	adj = adj + adj.T
	
	return adj, sizes
	
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
	

def sample_spherical(npoints, sphere_dim=2):
	"""
	samples npoints on a sphere of dimension sphere_dim by sampling from a normal distribution and then projecting points onto a sphere
	"""
	
	vec = np.random.randn(sphere_dim+1, npoints)
	
	for i in range(npoints):
		vec[:,i] /= np.linalg.norm(vec[:,i], axis=0)
	return np.transpose(vec)

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
	
#generate sample data
num_pts = 100
circle_sample = sample_spherical(num_pts,sphere_dim = 1)
noise_para = 0.2
pts = np.array([pt + noise_para*(np.random.rand(2)-0.5) for pt in circle_sample])


normalized_f_vals = (pts[:,1] - np.min(pts[:,1]))/np.max(pts[:,1] - np.min(pts[:,1]))
init_cols = plt.get_cmap("jet")(normalized_f_vals)
plt.scatter(pts[:,0],pts[:,1],c=init_cols); plt.show()

#get dist mat 
dM = np.zeros((num_pts,num_pts))
for i in range(num_pts-1):
	for j in range(i+1,num_pts):
		dM[i,j] = np.linalg.norm(pts[i]-pts[j])
dM = dM + dM.T 

#3 intervals 
dist_threshold = 0.7

pts1 = np.arange(num_pts)[pts[:,1]<=0.0]
CC1 = fast_marching_CC(list(pts1),[geometric_nbrs(idx, dM,eps = dist_threshold) for idx in pts1])

plt.scatter(pts[pts1,0],pts[pts1,1],c=init_cols[pts1])
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()


pts2 = np.arange(num_pts)[(pts[:,1]>=-0.5) & (pts[:,1] <= 0.5)]
CC2 = fast_marching_CC(list(pts2),[geometric_nbrs(idx, dM,eps = dist_threshold) for idx in pts2])

plt.scatter(pts[pts2,0],pts[pts2,1],c=init_cols[pts2])
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()

pts3 = np.arange(num_pts)[pts[:,1]>=0.0]
CC3 = fast_marching_CC(list(pts3),[geometric_nbrs(idx, dM,eps = dist_threshold) for idx in pts3])

plt.scatter(pts[pts3,0],pts[pts3,1],c=init_cols[pts3])
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()

#get unique connected components for each 
#also extract colours corresponding to average value for pts in said cluster
cluster_cols = []
CCs = []
for CCi in [CC1,CC2,CC3]:
	for cc in CCi:
		CCs.append(cc)
		avg_val = sum([normalized_f_vals[i] for i in cc])/len(cc)
		cluster_cols.append(plt.get_cmap("jet")(avg_val))

mapper = MAPPER(CCs)

A = mapper[0]
sizes = 10.*mapper[1]

#plot the graph 
coords = graph_embed(A)[0]

for i in range(len(A)-1):
	for j in range(i+1, len(A)):
		if A[i,j] >0.:
			plt.plot([coords[i,0],coords[j,0]],[-coords[i,1],-coords[j,1]],c="k")

plt.scatter(coords[:,0],-coords[:,1],s = sizes,c=cluster_cols)
			
plt.show()

#voxelized version 
def fast_marching_CC(pts, pt_nbrs,noisy = 0):
	"""
	in: -point list (should be indices)
		-point neighbors
	out: -list of connected components
	"""
	checked_array = np.zeros(len(pts))
	
	#preprocess pts, pt_nbrs 
	#set up dictionary
	pt_dict = {}
	idx = 0
	for i in range(len(pts)):
		pt_dict[pts[i]] = idx
		idx += 1
	
	new_pt_nbrs = []
	#convert pt_nbrs to local idxs 
	for nbrs in pt_nbrs:
		local_nbrs = []
		for nbr in nbrs:
			try:
				local_nbrs.append(pt_dict[nbr])
			except:
				pass
		new_pt_nbrs.append(local_nbrs)
	
	CCs = []
	while(sum(checked_array) < 2.*len(pts)):
		if noisy> 1:
			sum_check_time = time.time()
			print("checking the sum ({}) took {}".format(sum(checked_array),time.time()-sum_check_time))
		if noisy >= 1:
			global_time = time.time()
		try:
			if noisy>1:
				t1 = time.time()
				next_idx = np.arange(len(pts))[checked_array == 1.][0]
				t2 = time.time()
				#print("getting the index took: {}".format(t2-t1))
			else:
				next_idx = np.arange(len(pts))[checked_array == 1.][0]
		except:
			if noisy>1:
				t1 = time.time()
				next_idx = np.arange(len(pts))[checked_array == 0.][0]
				CCs.append([pts[next_idx]])
				t2 = time.time()
				#print("getting the index took: {}".format(t2-t1))
			else:
				next_idx = np.arange(len(pts))[checked_array == 0.][0]
				CCs.append([pts[next_idx]])
			
		
		#set the next_idx to have been checked 
		checked_array[next_idx] = 2.
		
		if noisy > 1:
			local_time = time.time()
			#run through the neighbors of next_idx 
			for nbr in new_pt_nbrs[next_idx]:
				if checked_array[nbr]<1.:
					CCs[-1].append(pts[nbr])
					checked_array[nbr] = 1.
			print("neighbor checking took {}".format(time.time()-local_time))
		else:
			#run through the neighbors of next_idx 
			for nbr in new_pt_nbrs[next_idx]:
				if checked_array[nbr]<1.:
					CCs[-1].append(pts[nbr])
					checked_array[nbr] = 1.
	if noisy >= 1:
		print("total time taken was {}".format(time.time()-global_time))
	return CCs

def get_centroid(pt_list):
    cent = np.zeros(len(pt_list[0]))

    for pt in pt_list:
        for i in range(len(pt)):
            cent[i] += pt[i]
    cent /= len(pt_list)

    return cent

voxeled = voxelizer(pts,gauss_para=0.1,num_grid_pts = [100,100])
voxeled = voxeled/np.max(voxeled)

plt.imshow(voxeled);plt.show()
plt.imshow(voxeled>0.27);plt.show()

time1 = time.time()
bits = (voxeled > 0.27).astype(int)

nonzeros = bits.nonzero()
v_pts = np.array([[nonzeros[0][i],nonzeros[1][i]] for i in range(len(nonzeros[0]))])

time2 = time.time()
print("time to isolate non-zeros was {}".format(time2-time1))

v_pts1 = v_pts[v_pts[:,0] <= 40]
v_pts1 = [tuple(vp) for vp in v_pts1]

time3 = time.time()
print("time to plot first non-zeros: {}".format(time3-time2))

v_CC1 = fast_marching_CC(v_pts1,[pixel_nbrs(idx) for idx in v_pts1],noisy = 0)

time4 = time.time()
print("time to get CCs: {}".format(time4-time3))
for cc in v_CC1:
	print("found a CC of size {}".format(len(cc)))

plot_mat = np.zeros((len(voxeled),len(voxeled)))
for vp in v_pts1:
	plot_mat[vp[0],vp[1]] = 1.
	

plt.imshow(plot_mat); plt.show()

v_pts2 = v_pts[(v_pts[:,0] >= 30)&(v_pts[:,0]<=70)]
v_pts2 = [tuple(vp) for vp in v_pts2]

time3 = time.time()
v_CC2 = fast_marching_CC(v_pts2,[pixel_nbrs(idx) for idx in v_pts2],noisy=0)

time4 = time.time()
print("time to get CCs: {}".format(time4-time3))
for cc in v_CC2:
	print("found a CC of size {}".format(len(cc)))

plot_mat = np.zeros((len(voxeled),len(voxeled)))
for vp in v_pts2:
	plot_mat[vp[0],vp[1]] = 1.
plt.imshow(plot_mat); plt.show()

v_pts3 = v_pts[v_pts[:,0] >= 60]
v_pts3 = [tuple(vp) for vp in v_pts3]
v_CC3 = fast_marching_CC(v_pts3,[pixel_nbrs(idx) for idx in v_pts3])

plot_mat = np.zeros((len(voxeled),len(voxeled)))
for vp in v_pts3:
	plot_mat[vp[0],vp[1]] = 1.
plt.imshow(plot_mat); plt.show()

v_CCs = []
for CCi in [v_CC1,v_CC2,v_CC3]:
	for cc in CCi:
		v_CCs.append(cc)

v_mapper = MAPPER(v_CCs)

A = v_mapper[0]
sizes = 10.*v_mapper[1]

#plot the graph 
coords = graph_embed(A)[0]

for i in range(len(A)-1):
	for j in range(i+1, len(A)):
		if A[i,j] >0.:
			plt.plot([coords[i,0],coords[j,0]],[-coords[i,1],-coords[j,1]],c="k")

plt.scatter(coords[:,0],-coords[:,1],s = sizes)
			
plt.show()

#test with suryas images 
fp = "C:/Users/hamilton.w/Downloads/TCGA-DH-A669-01Z-00-DX2-0752_seg68.png"

start_time = time.time()
#read image
im = Image.open(fp).convert("L")
arr = np.asarray(im)

unique_entries = np.unique(arr)

centroids = []

#get segments
for u in np.sort(unique_entries)[1:]:
	nonzeros = np.nonzero((arr==u).astype("int"))
	nonzeros = [tuple([nonzeros[0][i],nonzeros[1][i]]) for i in range(len(nonzeros[0]))]

	#print("starting fast marching for sample {}".format(index_of_computation))

	CCs = fast_marching_CC(nonzeros, [pixel_nbrs(pt) for pt in nonzeros])

	#print("there are {} cc's".format(len(CCs)))

	for cc in CCs:
		#print("this cc has {} pixels".format(len(cc)))
		centroids.append(get_centroid(cc))

centroids = np.array(centroids)

end_time = time.time()
print("total segmentation time: {}".format(end_time-start_time))