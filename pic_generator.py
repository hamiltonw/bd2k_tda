"""
2d TDA VR
"""

#generate TDA data 
pts = generate_TDA_data(dim=2,noise_para=0.2)
#plt.scatter(pts[:,0],pts[:,1]); plt.show()

#generate VR filtration
VR_filt = VR_filter(compute_distmat(pts))

#animate VR_filt
filtration_animator(pts,VR_filt[0],VR_filt[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "TDA_VR_2d",max_bt = 6.)

"""
2d noisy circle VR
"""

#generate noisy circle data 
pts = sample_spherical(30,sphere_dim=1)
#plt.scatter(pts[:,0],pts[:,1]); plt.show()

#add noise
pts = [p+0.1*(np.random.rand(2) - 0.5) for p in pts]
pts = np.array(pts)

#generate VR filtration
VR_filt = VR_filter(compute_distmat(pts))

#animate VR_filt
filtration_animator(pts,VR_filt[0],VR_filt[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "circle_VR_2d")

"""
voxel filtration 
"""
pts = generate_TDA_data(dim=2,noise_para=0.1)
vox_vals = voxelizer(pts, dims=[[0.,21.],[0.,7.]],num_grid_pts = [50,20],gauss_para = 1.0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(vox_vals)
plt.show()

filtration_animator_cubical(vox_vals,save_panels = [],save_path="C:/Users/hamilton.w/Pictures/",save_title="TDA_voxel_0p3",min_bt=0.1)

#varying gauss_para 
pts = generate_TDA_data(dim=2,noise_para=0.1)

for gp in [1.5,2.,5.,10.]:
	vox_vals = voxelizer(pts, dims=[[0.,21.],[0.,7.]],num_grid_pts = [50,20],gauss_para = gp)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(vox_vals)
	plt.show()

"""
sphere_with height function
"""
pts = sample_spherical(100)
cvxhull = sptl.ConvexHull(pts)
simps2 = cvxhull.simplices

simps = []
for i in range(len(pts)):
	simps.append(tuple([i]))

for s in simps2:
	simps.append(tuple(s))
	for e in iter.combinations(s,2):
		if not(e in simps):
			simps.append(e)

simps = [list(s) for s in simps]

#evaluate the function on simplices 
f_vals = pts[:,2]

filtered_complex = f_eval_simplices(f_vals,simps)
filtration_animator(pts,filtered_complex[0],filtered_complex[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "sphere_height_fn")

pts = sample_spherical(1000)
cvxhull = sptl.ConvexHull(pts)
simps2 = cvxhull.simplices

simps = []
for i in range(len(pts)):
	simps.append(tuple([i]))

for s in simps2:
	simps.append(tuple(s))
	for e in iter.combinations(s,2):
		if not(e in simps):
			simps.append(e)

simps = [list(s) for s in simps]

f_vals = spherical_harmonic(pts).real

filtered_complex = f_eval_simplices(f_vals,simps,increasing=True)
filtration_animator(pts,filtered_complex[0],filtered_complex[1],save_panels = [0,100,200,300,400],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "sphere_harmonics_top-down_fn",show_points = True)

"""
alpha shapes 
"""

#noisy circle 
pts = sample_spherical(30,sphere_dim=1)
#plt.scatter(pts[:,0],pts[:,1]); plt.show()

#add noise
pts = [p+0.1*(np.random.rand(2) - 0.5) for p in pts]
pts = np.array(pts)

#compute alpha filtration
alpha_filtration = construct_alpha_complex(pts)
filtration_animator(pts,np.array(alpha_filtration[0]),alpha_filtration[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "noisy_circle_alpha",show_points = True)


#noisy sphere  
pts = sample_spherical(50,sphere_dim=2)

#add noise
pts = [p+0.2*(np.random.rand(3) - 0.5) for p in pts]
pts = np.array(pts)

#compute alpha filtration
alpha_filtration = construct_alpha_complex(pts)
filtration_animator(pts,np.array(alpha_filtration[0]),alpha_filtration[1],save_panels = [0,100,250,400,500,600,640],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "noisy_sphere_alpha",show_points = True)

#TDA 2d
pts = generate_TDA_data(dim=2,noise_para=0.2)

alpha_filtration = construct_alpha_complex(pts)

"""
dim = 2
dims = []
for i in range(dim):
	dims.append([np.min(pts[:,i])-1.,np.max(pts[:,i])+1.])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(dims[0][0],dims[0][1])
ax.set_ylim(dims[1][0],dims[1][1])

ax.scatter(pts[:,0],pts[:,1])
ax.set_aspect("equal")

plt.show()
"""

#animate VR_filt
filtration_animator(pts,np.array(alpha_filtration[0]),alpha_filtration[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "TDA_2d_alpha",show_points = True,max_bt = 10.)


#TDA 3d
pts = generate_TDA_data(dim=3,noise_para=0.5)

alpha_filtration = construct_alpha_complex(pts)

#animate VR_filt
filtration_animator(pts,np.array(alpha_filtration[0]),alpha_filtration[1],save_panels = [],save_path = "C:/Users/hamilton.w/Pictures/",save_title = "TDA_3d_alpha",show_points = True)

