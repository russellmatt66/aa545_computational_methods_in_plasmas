"""
Matt Russell
7/28/21
test_PW.py - unit tests for Particle-Weighting step in PIC algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import picmodule_1pt2 as pmod
from celluloid import Camera

# Parameters
Nx = 33 # number of grid points = 32 + 1
x_min = -np.pi
x_max = np.pi
L = x_max - x_min
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
dx = (x_max - x_min)/float(Nx)

# Normalization
omega_p = 1.0
tau_p = 2.0 * np.pi / omega_p
eps_0 = 1.0 # vacuum permittivity
q_over_m = -1.0 # charge to mass ratio of superparticle
# q_sp = (eps_0 * L / N) * (1.0 / q_over_m) # charge of a superparticle
# m_sp = (N / L) * q_sp**2
# Te = 1.0 # [eV]
# lambda_De = np.sqrt(Te * N * (q_over_m**2) / (eps_0 * L))
# v_th = omega_p * lambda_De

def PWtest(N,rho_j): # run through Nmin to Nmax and see output of charge-weighting
    x_i = np.zeros((N,1),dtype=float)
    a = (L - 2.0*dx)/(N-1)
    b = x_min + dx
    for pidx in np.arange(N):
        x_i[pidx] = a*pidx + b
    q_sp = (eps_0 * L / N) * (1.0 / q_over_m)
    m_sp = (N / L) * q_sp**2
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp)
    plt.plot(x_grid,rho_j,label='Charge Density')
    plt.scatter(x_i,np.ones(np.size(x_i)),label='Particles')
    plt.scatter(x_grid,np.zeros(np.size(x_grid)),label='Grid Points')
    plt.legend()
    plt.title('Initial Particle-Weighting N = %i' %N)
    return 0

def PWmove(x_i,v_i,dt,rho_j,flag,iter): # Watch a single particle move through the grid
    print('This is iteration %i' %iter)
    # flag = 0 is standard
    if flag == 1: # particle reached other side of grid
        return 0
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp) # WeightOrd=1
    print('x_i is %3.2f' %x_i[0])
    print('v_i is %3.2f' %v_i)
    print('dt is %3.2f' %dt)
    x_i[0] = x_i[0] + v_i*dt
    print('x_i is %3.2f' %x_i[0]) 
    if np.abs(x_i[0]) > x_max: # particle motion under control but this handles left
        flag = 1
    plt.scatter(x_grid,rho_j,label='Charge Density')
    plt.scatter(x_i,np.ones(np.size(x_i)),label='Particles')
    for gidx in np.arange(Nx):
        plt.axvline(x=x_grid[gidx],ls='--')
    plt.title('Charge density from a single particle moving through the grid')
    plt.legend()
    plt.show()
    iter = iter + 1
    return PWmove(x_i,v_i,dt,rho_j,flag,iter)

def PWmoveplusfind(x_i,v_i,dt,rho_j,flag,iter): # Watch a single particle move through the grid and overlay the indices: j and  j+1 s.t x_j < x_i < x_{j+1}
    print('This is iteration %i' %iter)
    # flag = 0 is standard
    if flag == 1: # particle reached other side of grid
        return 0
    plt.scatter(x_i,np.ones(np.size(x_i)),label='Particles')
    jfound = pmod.Findj(x_grid,x_i)
    jfoundp1 = (jfound + 1) % Nx
    plt.axvline(x=x_grid[jfound],ls='--')
    plt.axvline(x=x_grid[jfoundp1],ls='--')
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp) # WeightOrd=1
    plt.scatter(x_grid,rho_j,label='Charge Density')
    plt.title('Charge density from a single particle moving through the grid')
    plt.legend()
    plt.show()
    x_i[0] = x_i[0] + v_i*dt
    if np.abs(x_i[0]) > x_max: # particle motion under control but this handles left
        flag = 1
    iter = iter + 1
    return PWmoveplusfind(x_i,v_i,dt,rho_j,flag,iter)

""" Step through N \in [2,64] 
Nmin = 2
Nmax = 64

for nidx in np.linspace(Nmin,Nmax,num=(Nmax-Nmin)):
    rho_j = np.zeros((Nx,1),dtype=float)
    rvar = PWtest(int(nidx),rho_j)
    plt.show()
"""

""" Watch single particle move through grid
# Conclusion: my search algorithm is not working correctly
N = 1
x_i = np.zeros((N,1),dtype=float)
x_i[0] = x_min
Nt = 100 # number of time steps, i.e, frames
dt = 1.0
v_i = L/(float(Nt)*dt)
print('v_i is %3.2f' %v_i)
rho_j = np.zeros((Nx,1),dtype=float)
q_sp = (eps_0 * L / N) * (1.0 / q_over_m)
flag = 0
plt.figure()
rvar = PWmove(x_i,v_i,dt,rho_j,flag,0)
"""

""" Watch single particle move through grid and track search algorithm results
# Conclusion: search algorithm appears to be working correctly based on observations
x_i = -np.pi
Nt = 100
dt = 1.0
v_i = L/(float(Nt)*dt)

for tidx in np.arange(Nt):
    jfound = pmod.Findj(x_grid,x_i)
    jfoundp1 = (jfound + 1) % Nx
    plt.axvline(x=x_grid[jfound])
    plt.axvline(x=x_grid[jfoundp1])
    plt.scatter(x_i,np.ones(1),label='Particle')
    plt.scatter(x_grid,np.zeros(Nx),label='Grid Points')
    plt.title('Search algorithm output')
    plt.legend()
    plt.show()
    x_i = x_i + v_i*dt
"""

""" Watch single particle move through grid, track search algo results, and charge density weighting 
# Conclusion: Honestly, everything here seems to be hunky-dory
N = 1
x_i = np.zeros(N)
x_i[0] = -np.pi
Nt = 100
dt = 1.0
v_i = L/(float(Nt)*dt)
rho_j = np.zeros((Nx,1),dtype=float)
q_sp = (eps_0 * L / N) * (1.0 / q_over_m)
flag = 0
plt.figure()
rvar = PWmoveplusfind(x_i,v_i,dt,rho_j,flag,0)
"""

"""
Make a movie out of a single particle moving through grid, track search algo results, and particle-weighting
# Conclusion: movie successfully made - nothing anomalous detected
N = 1 # number of particles
Nt = 10 # number of frames
dt = 1.0
x_i = x_min
v_i = L/(float(Nt)*dt)
rho_j = np.zeros((Nx,1),dtype=float)
q_sp = (eps_0 * L / N) * (1.0 / q_over_m) # charge of a superparticle

ScatterFig = plt.figure()
camera = Camera(ScatterFig)
for i in np.arange(Nt):
    jfound = pmod.Findj(x_grid,x_i) # binary search
    jfoundp1 = jfound + 1 # '%' operator should not be necessary
    rho_j = pmod.ParticleWeighting(1,[x_i],x_grid,Nx,dx,L,rho_j,q_sp)
    l1 = plt.scatter(x_grid,np.zeros(Nx),color='black',label='grid')
    l2 = plt.scatter(x_i,1.0,color='red',label='Particle')
    l3 = plt.axvline(x=x_grid[jfound],label='j from search algo')
    l4 = plt.axvline(x=x_grid[jfoundp1])
    l5 = plt.scatter(x_grid,rho_j,color='blue')
    plt.legend([l1, l2, l3, l5],['grid', 'Particle', 'j and j+1 from search algo', 'rho'])
    camera.snap()
    x_i = x_i + v_i*dt # Particle Push

animation = camera.animate()
animation.save('/home/matt/python/movie_playground/movies/KP1pt2_part-grid-find_motion.gif', writer = 'imagemagick', fps=1)
"""

"""
Make a movie out of N = "powers-of-two" particle systems to demonstrate bug that appears from charge-weighting
# Conclusion: movie successfully made - anomaly demonstrated
#N_min = 2
#N_max = 64

N = 2 ** np.arange(7)

InitWeightFig = plt.figure()
camera = Camera(InitWeightFig)

rho_j = np.zeros((Nx,1),dtype=float)

for nidx in np.arange(np.size(N)):
    if nidx == 0:
        continue
    x_i = np.zeros((N[nidx],1),dtype=float)
    a = (L - 2.0*dx)/(N[nidx]-1)
    b = x_min + dx
    for pidx in np.arange(N[nidx]):
        x_i[pidx] = a*pidx + b
    q_sp = (eps_0 * L / N[nidx]) * (1.0 / q_over_m)
    m_sp = (N[nidx] / L) * q_sp**2
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp)
    l1 = plt.scatter(x_grid,np.zeros(Nx),color='black',label='grid')
    l2 = plt.scatter(x_i,np.ones((N[nidx],1),dtype=float),color='red',label='Particle')
    l3 = plt.scatter(x_grid,rho_j,color='blue')
    plt.title('Initial particle-weighting for N = %i particles' %N[nidx])
    plt.legend([l1,l2,l3],['Grid','Particles','Charge Density'])
    camera.snap()

animation = camera.animate()
animation.save('/home/matt/python/movie_playground/movies/KP1pt2_part-grid-weight.gif', writer = 'imagemagick', fps=1)
"""

"""
Make a movie out of the electric potential computation for  N = "powers-of-two" particle systems
"""
# Conclusion:
N = 2 ** np.arange(7)

Lmtx = pmod.LaplacianStencil(Nx,dx,eps_0)
PotentialFig = plt.figure()
camera = Camera(PotentialFig)

rho_j = np.zeros((Nx,1),dtype=float)
phi_j = np.zeros((Nx,1),dtype=float)

for nidx in np.arange(np.size(N)):
    if nidx == 0:
        continue
    x_i = np.zeros((N[nidx],1),dtype=float)
    a = (L - 2.0*dx)/(N[nidx]-1)
    b = x_min + dx
    for pidx in np.arange(N[nidx]):
        x_i[pidx] = a*pidx + b
    q_sp = (eps_0 * L / N[nidx]) * (1.0 / q_over_m)
    m_sp = (N[nidx] / L) * q_sp**2
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(Lmtx,phi_j,rho_j)
    l1 = plt.scatter(x_grid,np.zeros(Nx),color='black',label='grid')
    l2 = plt.scatter(x_i,np.ones((N[nidx],1),dtype=float),color='red',label='Particle')
    l3 = plt.scatter(x_grid,phi_j,color='blue')
    plt.title('Initial particle-weighting for N = %i particles' %N[nidx])
    plt.legend([l1,l2,l3],['Grid','Particles','Electric Potential'])
    camera.snap()

animation = camera.animate()
animation.save('/home/matt/python/movie_playground/movies/KP1pt2_potential.gif', writer = 'imagemagick', fps=1)
