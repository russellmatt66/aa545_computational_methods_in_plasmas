"""
Matt Russell
University of Washington
AA545: Computational Methods for Plasmas
Computer Project 1.2: Kinetic Modeling: Vlasov-Poisson PIC
picmain_1pt2.py
Main program for the given assignment to build a PIC simulation that can be used
to investigate Langmuir Oscillations and the Leap-Frog Instability
"""
import numpy as np
import matplotlib.pyplot as plt
import picmodule_1pt2 as pmod

""" Initialization Phase """
print("Opening Initialization Phase ...")

print("Calling pmod.Initialize()")
InitVector = pmod.Initialize()
N = int(InitVector[0]) # number of particles
Nx = int(InitVector[1]) + 1 # number of grid points = Ncells + 1
WeightingOrder = int(InitVector[2]) # 0th or 1st order weighting
InitialV = int(InitVector[3])

x_i = np.zeros((N,1),dtype=float) # array containing the particle's positions
v_i = np.zeros((N,1),dtype=float) # array containing the particle's velocities
E_i = np.zeros((N,1),dtype=float) # array containing the electric fields experienced by each particle

""" Grid Generation """
print("Generating grid")
x_min = -np.pi
x_max = np.pi
L = x_max - x_min
dx = L/float(Nx)
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)

print("Defining normalization")
omega_p = 1.0
tau_p = 2.0 * np.pi / omega_p
eps_0 = 1.0 # vacuum permittivity
q_over_m = -1.0 # charge to mass ratio of superparticle
q_sp = (eps_0 * L / N) * (1.0 / q_over_m) # charge of a superparticle
Te = 1.0 # [eV]
lambda_De = np.sqrt(Te * N * (q_over_m**2) / (eps_0 * L))
v_th = omega_p * lambda_De

E_j = np.zeros((Nx,1),dtype=float) # Grid Electric Field
phi_j = np.zeros((Nx,1),dtype=float) # Grid Electrostatic Potential
rho_j = np.zeros((Nx,1),dtype=float) # Grid Charge Density

Lmtx = pmod.LaplacianStencil(Nx,dx,eps_0) 
FDmtx = pmod.FirstDerivativeStencil(Nx,dx)

print("Closing Grid Generation Phase")

print("Initializaing Particle Distribution")
# 
if (N == 2 and InitialV == 0):
    vprime = 0.0 * v_th
    x_i[0] = -np.pi/4.0
    x_i[1] = np.pi/4.0
    v_i[0] = vprime
    v_i[1] = -vprime

if(N == 2 and InitialV == 1):
    vprime = 0.01 * v_th
    x_i[0] = -np.pi/2.0
    x_i[1] = np.pi/2.0
    v_i[0] = vprime
    v_i[1] = -vprime

if(N != 2):
    vprime = 0.0 * v_th
    for pidx in np.arange(N):
        x_i[pidx] = (x_min + dx) + float(pidx)*2.0*(x_max + dx)/float(N-1)
        v_i[pidx] = vprime*np.sin(2.0*np.pi/L * x_i[pidx])

""" Main Loop """
dt = 0.032 * tau_p
Nt = 100
for n in np.arange(Nt):
    print("Taking step %i" %n)
    rho_j = pmod.ParticleWeighting(WeightingOrder,x_i,N,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(rho_j,Lmtx)
    E_j = pmod.FieldSolveES(phi_j,FDmtx)
    E_i = pmod.ForceWeighting(WeightingOrder,x_i,E_i,x_grid,Nx,dx,E_j)
    x_i, v_i = pmod.LeapFrog(x_i,v_i,E_i,dt,q_over_m,n,x_min,x_max)
