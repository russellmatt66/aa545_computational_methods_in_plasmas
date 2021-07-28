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
m_sp = (N / L) * q_sp**2
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
a = (L - 2.0*dx)/(N-1) # linear coefficients for evenly distributing particles
b = x_min + dx 
if (N == 2 and InitialV == 0):
    x_i[0] = -np.pi/4.0
    x_i[1] = np.pi/4.0
    v_i[0] = 0.0
    v_i[1] = 0.0

if(N != 2 and InitialV == 0):
    for pidx in np.arange(N):
        x_i[pidx] = a*pidx + b
        v_i[pidx] = 0.0

if(N == 2 and InitialV == 1):
    vprime = 0.25 * v_th
    x_i[0] = -np.pi/2.0
    x_i[1] = np.pi/2.0
    v_i[0] = vprime
    v_i[1] = -vprime

if(N != 2 and InitialV == 1):
    vprime = 0.001 * v_th
    for pidx in np.arange(N):
        x_i[pidx] = a*pidx + b
        v_i[pidx] = vprime*np.sin(2.0*np.pi/L * x_i[pidx])

""" Initialize Diagnostics and Simulation Parameters """
EnergyFig = plt.figure()
PhaseSpaceFig = plt.figure()
FieldFig = plt.figure()
ChargeDensityFig = plt.figure()

dt = 0.032 * tau_p
Nt = 100

KineticEnergy = np.zeros(Nt)
ElectricFieldEnergy = np.zeros(Nt)
TotalEnergy = np.zeros(Nt)


""" Main Loop """
for n in np.arange(Nt):
    print("Taking step %i" %(n+1))
    KineticEnergy[n] = pmod.ComputeKineticEnergy(v_i,m_sp) # Compute before particle push
    # Plot particle locations in phase-space
    plt.figure(PhaseSpaceFig.number)
    plt.scatter(x_i,v_i)
    rho_j = pmod.ParticleWeighting(WeightingOrder,x_i,N,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(Lmtx,phi_j,rho_j)
    E_j = pmod.FieldSolveES(phi_j,FDmtx)
    E_i = pmod.ForceWeighting(WeightingOrder,x_i,E_i,x_grid,Nx,dx,E_j)
    x_i, v_i = pmod.LeapFrog(N,x_i,v_i,E_i,dt,q_over_m,n,x_min,x_max) # Particle push
    ElectricFieldEnergy[n] = pmod.ComputeElectricFieldEnergy(E_j,Nx,dx) # Compute after field is solved for
    TotalEnergy[n] = KineticEnergy[n] + ElectricFieldEnergy[n]
    if n == 0:
        plt.figure(ChargeDensityFig.number)
        plt.plot(x_grid,rho_j)
        plt.title('Initial grid charge density, N = %i particles' %N)
        plt.figure(FieldFig.number)
        plt.plot(x_grid,E_j)
        plt.title('Initial grid electric field, N = %i particles')
                  
# Plot Energy 
tvector = np.linspace(0.0,float((Nt-1)*dt),Nt)
plt.figure(EnergyFig.number)
plt.plot(tvector,KineticEnergy,label="Kinetic")
plt.plot(tvector,ElectricFieldEnergy,label="Electric")
plt.plot(tvector,TotalEnergy,label="Total")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('System Energy History')
# plt.title('System Energy History for %i-Order Weighting, $v^{\'}$ = %4.3f$v_{th}$, $\Delta t = %4.3f$, and $N_{steps} = %i$' %(WeightingOrder,vprime/v_th,dt,Nt))

plt.figure(PhaseSpaceFig.number)
plt.xlabel('x')
plt.ylabel('v (normalized to $v_{th}$)')
plt.xlim((x_min,x_max))
plt.title('Superparticle Trajectories')
# plt.title('Superparticle Trajectories for %i-Order Weighting with $v^{\'}$ = %4.3f$v_{th}$, dt = %4.3f and $N_{steps}$ = %i' %(WeightingOrder,vprime/v_th,dt,Nt))

plt.figure(ChargeDensityFig.number)
plt.xlabel('x')
plt.ylabel('$\\rho$')
plt.xlim((x_min,x_max))

plt.figure(FieldFig.number)
plt.xlabel('x')
plt.ylabel('E')
plt.xlim((x_min,x_max))

plt.show()
