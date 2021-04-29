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
ZeroInitialV = int(InitVector[3])

# Colors for the scatterplot later
particleColors = [None] * N
black = '0x000000'
white = '0xffffff'
hexIncrement = hex(int((int(white,16)-int(black,16))/N))
hexColor = '0x000000'

for cidx in np.arange(N):
    particleColors[cidx] = pmod.FormatHex(hexColor)
    hexColor = pmod.AddHex(hexColor,hexIncrement)

particlesPosition = np.zeros((N,1),dtype=float)
particlesVelocity = np.zeros((N,1),dtype=float)
particlesField = np.zeros((N,1),dtype=float) # array of fields experienced by particles, E_i

""" Grid Generation Phase """
print("Opening Grid Generation Phase ...")

x_min = -np.pi
x_max = np.pi
L = x_max - x_min
dx = (x_max - x_min)/float(Nx)
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
E_j = np.zeros((Nx,1),dtype=float) # Grid Electric Field
phi_j = np.zeros((Nx,1),dtype=float) # Grid Potential
rho_j = np.zeros((Nx,1),dtype=float) # Grid Charge Density

Lmtx = pmod.LaplacianStencil(Nx,dx)
FDmtx = pmod.FirstDerivativeStencil(Nx,dx)

print("Closing Grid Generation Phase")

print("Defining Normalization")
eps_0 = 1.0 # vacuum permittivity
qm = -1.0 # charge to mass ratio of superparticle
omega_p = 1.0 # plasma frequency
tau_plasma = 2.0*np.pi/omega_p
q_sp = eps_0 * omega_p**2 * (L/float(N)) / qm# charge associated with a superparticle given normalization - population assumed uniform
m_sp = qm*q_sp # mass associated with a particular superparticle given normalization - population assumed uniform
T = 1.0 # [eV]
lambda_D = np.sqrt((eps_0 * T * L)/(N * q_sp **2))
v_th = omega_p*lambda_D # thermal velocity

print("Initalizing Particle Distribution")
# Initial Conditions for 1.
if (N == 2 and ZeroInitialV == 0):
    particlesPosition[0] = -np.pi/4.0
    particlesPosition[1] = np.pi/4.0
    particlesVelocity[0] = 0.0
    particlesVelocity[1] = 0.0

if(N == 2 and ZeroInitialV == 1):
    vprime = 1.0*v_th
    particlesPosition[0] = -np.pi/2.0
    particlesPosition[1] = np.pi/2.0
    particlesVelocity[0] = vprime
    particlesVelocity[1] = -vprime

# diagFig, (axKin, axE, axTot) = plt.subplots(nrows=3,ncols=1)
EnergyFig = plt.figure()
PhaseSpaceFig = plt.figure()

print("Closing Initialization Phase ...")

""" PIC Phase """
print("Beginning PIC Simulation")
dt = 0.05*tau_plasma # [s]
Nt = 300 # number of steps to take
KineticEnergy = np.zeros(Nt)
ElectricFieldEnergy = np.zeros(Nt)
TotalEnergy = KineticEnergy + ElectricFieldEnergy

for n in np.arange(Nt):
    print("Taking step %i" %n)
    Efgrid,Ekin,Etotal = pmod.GridIntegrate(E_j,Nx,dx,particlesVelocity,m_sp) # Diagnostic
    KineticEnergy[n] = Ekin
    ElectricFieldEnergy[n] = Efgrid
    TotalEnergy[n] = KineticEnergy[n] + ElectricFieldEnergy[n]
    plt.figure(PhaseSpaceFig.number)
    for pidx in np.arange(N):
        plt.scatter(particlesPosition[pidx],particlesVelocity[pidx],c=particleColors[pidx])
    rho_j = pmod.ParticleWeighting(WeightingOrder,particlesPosition,N,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(rho_j,Lmtx,Nx)
    E_j = pmod.FieldSolveES(phi_j,FDmtx)
    particlesField = pmod.ForceWeighting(WeightingOrder,dx,particlesField,E_j,particlesPosition,x_grid)
    particlesPosition, particlesVelocity = pmod.LeapFrog(particlesPosition,particlesVelocity,particlesField,dt,qm,n)


    # axKin.scatter(n,Ekin)
    # axE.scatter(n,Efgrid)
    # axTot.scatter(n,Etotal)
    # pmod.Diagnostics(E_j,particlesVelocity,n,axes=[axKin,axE,axTot])

t = np.linspace(0.0,float((Nt-1)*dt),Nt)
plt.figure(EnergyFig.number)
plt.plot(t,KineticEnergy,label="KE")
plt.plot(t,ElectricFieldEnergy,label="FE")
plt.plot(t,TotalEnergy,label="TE")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Evolution of system energy')

plt.figure(PhaseSpaceFig.number)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Superparticle Trajectories')

plt.show()
