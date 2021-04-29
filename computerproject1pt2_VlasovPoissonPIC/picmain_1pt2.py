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
# Don't know of another way to get rid of view
N = int(InitVector[0]) # number of particles
Nx = int(InitVector[1]) + 1 # number of grid points = Ncells + 1
WeightingOrder = int(InitVector[2]) # 0th or 1st order weighting

particlesPosition = np.empty((N,1),dtype=float)
particlesVelocity = np.empty((N,1),dtype=float)
particlesField = np.empty((N,1),dtype=float) # array of fields experienced by particles, E_i

# Initial Conditions for 1.
if (N == 2):
    particlesPosition[0] = -np.pi/4.0
    particlesPosition[1] = np.pi/4.0
    particlesVelocity[0] = 0.0
    particlesVelocity[1] = 0.0

# diagFig, (axKin, axE, axTot) = plt.subplots(nrows=3,ncols=1)
EnergyFig = plt.figure()
PhaseSpaceFig = plt.figure()

print("Closing Initialization Phase ...")
""" Grid Generation Phase """
print("Opening Grid Generation Phase ...")

x_min = -np.pi
x_max = np.pi
L = x_max - x_min
dx = (x_max - x_min)/float(Nx)
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
E_j = np.empty((Nx,1),dtype=float) # Grid Electric Field
phi_j = np.empty((Nx,1),dtype=float) # Grid Potential
rho_j = np.empty((Nx,1),dtype=float) # Grid Charge Density

Lmtx = pmod.LaplacianStencil(Nx,dx)
FDmtx = pmod.FirstDerivativeStencil(Nx,dx)

q_sp = -L/float(N) # charge associated with a superparticle given normalization - population assumed uniform
m_sp = -q_sp # mass associated with a particular superparticle given normalization - population assumed uniform
# q_background = -q_e # charge associated with the background
qm = -1.0 # charge to mass ratio of superparticle

print("Closing Grid Generation Phase")
""" PIC Phase """
print("Beginning PIC Simulation")
omega_p = 1.0
tau_plasma = 2.0*np.pi/omega_p
dt = 0.01*tau_plasma # [s]
Nt = 500 # number of steps to take
KineticEnergy = np.empty(Nt)
ElectricFieldEnergy = np.empty(Nt)
TotalEnergy = np.empty(Nt)

for n in np.arange(Nt):
    print("Taking step %i" %n)
    rho_j = pmod.ParticleWeighting(WeightingOrder,particlesPosition,N,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(rho_j,Lmtx,Nx)
    E_j = pmod.FieldSolveES(phi_j,FDmtx)
    particlesField = pmod.ForceWeighting(WeightingOrder,dx,particlesField,E_j,particlesPosition,x_grid)
    particlesPosition, particlesVelocity = pmod.LeapFrog(particlesPosition,particlesVelocity,particlesField,dt,qm,n)
    Efgrid,Ekin,Etotal = pmod.GridIntegrate(E_j,Nx,dx,particlesVelocity,m_sp) # Diagnostic
    KineticEnergy[n] = Ekin
    ElectricFieldEnergy[n] = Efgrid
    TotalEnergy[n] = Etotal
    plt.figure(PhaseSpaceFig.number)
    for i in np.arange(N):
        plt.scatter(particlesPosition[i],particlesVelocity[i])
    # axKin.scatter(n,Ekin)
    # axE.scatter(n,Efgrid)
    # axTot.scatter(n,Etotal)
    # pmod.Diagnostics(E_j,particlesVelocity,n,axes=[axKin,axE,axTot])

t = np.linspace(0.0,float((Nt-1)*dt),Nt)
plt.figure(EnergyFig.number)
plt.plot(t,KineticEnergy,t,ElectricFieldEnergy,t,TotalEnergy)
# axKin.plot(t,KineticEnergy)
# axE.plot(t,ElectricFieldEnergy)
# axTot.plot(t,TotalEnergy)

plt.show()
