"""
barcode generator
"""

#need linux/macOS
import dionysus as d

#generate data
TDA_data = generate_TDA_data(dim=2,noise_para=0.2)
TDA_alpha = construct_alpha_complex(TDA_data)

#draw the data
dim = 2
dims = []
for i in range(dim):
	dims.append([np.min(TDA_data[:,i])-1.,np.max(TDA_data[:,i])+1.])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(dims[0][0],dims[0][1])
ax.set_ylim(dims[1][0],dims[1][1])
ax.scatter(TDA_data[:,0],TDA_data[:,1])
ax.set_aspect("equal")

plt.show()

circ_data = sample_spherical(50,sphere_dim=1)
circ_data = [p+0.2*(np.random.rand(2) - 0.5) for p in circ_data]
circ_data = np.array(circ_data)
circ_alpha = construct_alpha_complex(circ_data)

#draw the data
dim = 2
dims = []
for i in range(dim):
	dims.append([np.min(circ_data[:,i])-1.,np.max(circ_data[:,i])+1.])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(dims[0][0],dims[0][1])
ax.set_ylim(dims[1][0],dims[1][1])
ax.scatter(circ_data[:,0],circ_data[:,1])
ax.set_aspect("equal")

plt.show()


sphere_data = sample_spherical(100,sphere_dim=2)
sphere_data = [p+0.2*(np.random.rand(3) - 0.5) for p in sphere_data]
sphere_data = np.array(sphere_data)
sphere_alpha = construct_alpha_complex(sphere_data)

#draw the data
dim = 3
dims = []
for i in range(dim):
	dims.append([np.min(sphere_data[:,i])-1.,np.max(sphere_data[:,i])+1.])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(dims[0][0],dims[0][1])
ax.set_ylim(dims[1][0],dims[1][1])
ax.set_zlim(dims[2][0],dims[2][1])
ax.scatter(sphere_data[:,0],sphere_data[:,1],sphere_data[:,2])
ax.set_aspect("equal")

plt.show()


"""
computing homology
"""
#VR filtration TDA
test_f = d.fill_rips(TDA_data, 2, 10.)
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()


#alpha filtration TDA
test_comp = TDA_alpha[0]
test_bts = TDA_alpha[1]
test_f = d.Filtration()
for j in range(len(test_comp)):
    test_f.append(d.Simplex(test_comp[j],test_bts[j]))
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()

"""
circle data
"""
#VR filtration circle
test_f = d.fill_rips(circ_data, 2, 10.)
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()


#alpha filtration circle
test_comp = circ_alpha[0]
test_bts = circ_alpha[1]
test_f = d.Filtration()
for j in range(len(test_comp)):
    test_f.append(d.Simplex(test_comp[j],test_bts[j]))
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()


#VR filtration sphere
test_f = d.fill_rips(sphere_data, 3, 10.)
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
d.plot.plot_diagram(dgms[2])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()

d.plot.plot_bars(dgms[2])
plt.show()


#alpha filtration sphere
test_comp = sphere_alpha[0]
test_bts = sphere_alpha[1]
test_f = d.Filtration()
for j in range(len(test_comp)):
    test_f.append(d.Simplex(test_comp[j],test_bts[j]))
p = d.homology_persistence(test_f)
dgms = d.init_diagrams(p, test_f)

#scatters
d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])
d.plot.plot_diagram(dgms[2])
plt.show()

#bars
d.plot.plot_bars(dgms[0])
plt.show()

d.plot.plot_bars(dgms[1])
plt.show()

d.plot.plot_bars(dgms[2])
plt.show()
