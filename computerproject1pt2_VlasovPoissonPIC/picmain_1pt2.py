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
white = '0xff00ff'
hexIncrement = hex(int((int(white,16)-int(black,16))/(N-1)))
hexColor = '0x000000'

for cidx in np.arange(N):
    particleColors[cidx] = pmod.FormatHex(hexColor)
    hexColor = pmod.AddHex(hexColor,hexIncrement)

particlesPosition = np.zeros((N,1),dtype=float)
particlesVelocity = np.zeros((N,1),dtype=float)
particlesField = np.zeros((N,1),dtype=float) # array of fields experienced by particles, E_i
v_n = np.empty(N,dtype=float) # use to compute oscillation frequency
v_np1 = np.empty(N,dtype=float) # use to compute oscillation frequency
v_0i = np.empty(N,dtype=float) # particle initial positions - use to compute oscillation frequency
ZeroCross_count = np.zeros(N,dtype=int) # use to compute oscillation frequency

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
    vprime = 0.0 # to make the presentation of results more uniform
    particlesPosition[0] = -np.pi/4.0
    particlesPosition[1] = np.pi/4.0
    particlesVelocity[0] = vprime
    particlesVelocity[1] = -vprime

if(N == 2 and ZeroInitialV == 1):
    vprime = 1.0*v_th
    particlesPosition[0] = -np.pi/2.0
    particlesPosition[1] = np.pi/2.0
    particlesVelocity[0] = vprime
    particlesVelocity[1] = -vprime

for idx in np.arange(N):
    v_0i[idx] = float(particlesVelocity[idx])

# diagFig, (axKin, axE, axTot) = plt.subplots(nrows=3,ncols=1)
EnergyFig = plt.figure()
PhaseSpaceFig = plt.figure()

print("Closing Initialization Phase ...")

""" PIC Phase """
print("Beginning PIC Simulation")
dt = 0.05*tau_plasma # [s]
Nt = 300 # number of steps to take
t0 = np.zeros((N,1),dtype=float) # for oscillation frequency computation
ExpectedNumberOfPeriods = (Nt*dt)/tau_plasma
SafetyFactor = 2 # Having more than twice the expected number of oscillations detected means period computation is junk anyways
tauOscillation = np.zeros((N,int(ExpectedNumberOfPeriods*SafetyFactor)),dtype=float) # store oscillation periods
tauTemp = np.zeros((N,1),dtype=float) # temporary container for tau_{osc} that attaches to end of tauOscillation w/np.hstack
numPeriod = np.zeros((N,1),dtype=int)
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
        v_n[pidx] = float(particlesVelocity[pidx]) # for oscillation frequency computation, float() is to break connection w/underlying array
    rho_j = pmod.ParticleWeighting(WeightingOrder,particlesPosition,N,x_grid,Nx,dx,L,rho_j,q_sp)
    phi_j = pmod.PotentialSolveES(rho_j,Lmtx,Nx)
    E_j = pmod.FieldSolveES(phi_j,FDmtx)
    particlesField = pmod.ForceWeighting(WeightingOrder,dx,particlesField,E_j,particlesPosition,x_grid)
    particlesPosition, particlesVelocity = pmod.LeapFrog(particlesPosition,particlesVelocity,particlesField,dt,qm,n) # Particle Push
    # Oscillation period calculation
    for pidx in np.arange(N):
        v_np1[pidx] = float(particlesVelocity[pidx]) # updated position
        if((v_n[pidx] - v_0i[pidx] > 0.0) and (v_np1[pidx] - v_0i[pidx] < 0.0)): # Check for Zero-Crossing
            print("Zero-Crossing detected")
            ZeroCross_count[pidx] += 1 # count it
        elif((v_n[pidx] - v_0i[pidx] < 0.0) and (v_np1[pidx] - v_0i[pidx] > 0.0)): # Check for Zero-Crossing
            print("Zero-Crossing detected")
            ZeroCross_count[pidx] += 1 # count it
        if(ZeroCross_count[pidx] == 2): # one oscillation period has elapsed!
            tauTemp[pidx,0] = float(n*dt - t0[pidx,0]) # calculate value and store in container
            numPeriod[pidx,0] += 1 # increment how many there have been for this particle
            t0[pidx,0] = n*dt
            ZeroCross_count[pidx] = 0
            tauOscillation[pidx,numPeriod[pidx,0]] = tauTemp[pidx,0]

print(tauOscillation)

omegaAverage = np.zeros((N,1),dtype=float)
omegaVariance = np.zeros((N,1),dtype=float)
NumberNonZeroEntries = np.zeros((N,1),dtype=int)
for pidx in np.arange(N):
    for oidx in np.arange(np.size(tauOscillation,axis=1)-1):
        if (tauOscillation[pidx,oidx] != 0.0):
            omegaAverage[pidx,0] = omegaAverage[pidx,0] + 2.0*np.pi/tauOscillation[pidx,oidx]
            NumberNonZeroEntries[pidx,0] += 1
    omegaAverage[pidx,0] = omegaAverage[pidx,0]/NumberNonZeroEntries[pidx,0] # something TypeError about conversion to Python scalar
    print("The average oscillation frequency for particle %i is %4.3f" %(pidx+1,omegaAverage[pidx,0]))

for pidx in np.arange(N):
    for oidx in np.arange(np.size(tauOscillation,axis=1)-1):
        if (tauOscillation[pidx,oidx] != 0.0):
            omegaVariance[pidx,0] = omegaVariance[pidx,0] + (2.0*np.pi/tauOscillation[pidx,oidx] - omegaAverage[pidx,0])**2
    omegaVariance[pidx,0] = omegaVariance[pidx,0]/NumberNonZeroEntries[pidx,0]
    print("With a variance of plus/minus %4.3f" %omegaVariance[pidx,0])

print("The theoretical oscillation frequency for all particles is %4.3f" %omega_p)



t = np.linspace(0.0,float((Nt-1)*dt),Nt)
plt.figure(EnergyFig.number)
plt.plot(t,KineticEnergy,label="Kinetic")
plt.plot(t,ElectricFieldEnergy,label="Electric")
plt.plot(t,TotalEnergy,label="Total")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('System Energy History for %i-Order Weighting, $\Delta t$ = %4.3f$\\tau_{p}$, and $N_{steps}$ = %i' %(WeightingOrder,dt/tau_plasma,Nt))

plt.figure(PhaseSpaceFig.number)
plt.xlabel('x')
plt.ylabel('v (normalized to $v_{th}$)')
plt.xlim((x_min,x_max))
plt.ylim((-6.0*v_th,6.0*v_th))
plt.title('Superparticle Trajectories for %i-Order Weighting with $v^{\'}$ = %4.3f$v_{th}$ and $N_{steps}$ = %i' %(WeightingOrder,vprime,Nt))

plt.show()
