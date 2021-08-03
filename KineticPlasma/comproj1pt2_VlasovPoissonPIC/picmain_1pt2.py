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
WeightingOrder = int(InitVector[1]) # 0th or 1st order weighting
InitialV = int(InitVector[2])

x_i = np.zeros((N,1),dtype=float) # array containing the particle's positions
v_i = np.zeros((N,1),dtype=float) # array containing the particle's velocities
E_i = np.zeros((N,1),dtype=float) # array containing the electric fields experienced by each particle

print("Defining normalization")
q_sp = (pmod.EPS_0 * pmod.L / N) * (1.0 / pmod.Q_OVER_M) # charge of a superparticle
m_sp = (N / pmod.L) * q_sp**2
lambda_De = np.sqrt(pmod.T_ELCTRN * N * (pmod.Q_OVER_M**2) / (pmod.EPS_0 * pmod.L))
v_th = pmod.OMEGA_P * lambda_De

Lmtx = pmod.LaplacianStencil() 
FDmtx = pmod.FirstDerivativeStencil()

print("Closing Grid Generation Phase")

print("Initializaing Particle Distribution")
# linear coefficients for evenly distributing particles
a = (pmod.L - 2.0*pmod.DX)/(N-1) 
b = pmod.X_MIN + pmod.DX

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
        v_i[pidx] = vprime*np.sin(2.0*np.pi/pmod.L * x_i[pidx])

""" Initialize Diagnostics and Simulation Parameters """
EnergyFig = plt.figure()
PhaseSpaceFig = plt.figure()
FieldFig = plt.figure()
ChargeDensityFig = plt.figure()

dt = 0.032 * pmod.TAU_P
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
    pmod.ParticleWeighting(WeightingOrder,x_i,q_sp)
    pmod.PotentialSolveES(Lmtx)
    pmod.FieldSolveES(FDmtx)
    E_i = pmod.ForceWeighting(WeightingOrder,x_i,E_i)
    x_i, v_i = pmod.LeapFrog(N,x_i,v_i,E_i,dt,n) # Particle push
    ElectricFieldEnergy[n] = pmod.ComputeElectricFieldEnergy() # Compute after field is solved for
    TotalEnergy[n] = KineticEnergy[n] + ElectricFieldEnergy[n]
    """
    if n == 0:
        plt.figure(ChargeDensityFig.number)
        plt.plot(x_grid,rho_j)
        plt.title('Initial grid charge density, N = %i particles' %N)
        plt.figure(FieldFig.number)
        plt.plot(x_grid,E_j)
        plt.title('Initial grid electric field, N = %i particles')
    """           

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
plt.xlim((pmod.X_MIN,pmod.X_MAX))
plt.title('Superparticle Trajectories')
# plt.title('Superparticle Trajectories for %i-Order Weighting with $v^{\'}$ = %4.3f$v_{th}$, dt = %4.3f and $N_{steps}$ = %i' %(WeightingOrder,vprime/v_th,dt,Nt))

plt.figure(ChargeDensityFig.number)
plt.xlabel('x')
plt.ylabel('$\\rho$')
plt.xlim((pmod.X_MIN,pmod.X_MAX))

plt.figure(FieldFig.number)
plt.xlabel('x')
plt.ylabel('E')
plt.xlim((pmod.X_MIN,pmod.X_MAX))

plt.show()
