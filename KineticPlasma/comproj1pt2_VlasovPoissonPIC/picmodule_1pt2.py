"""
Matt Russell
University of Washington
AA545: Computational Methods for Plasmas
Computer Project 1.2: Kinetic Modeling: Vlasov-Poisson PIC
picmodule_1pt2.py
Module that contains the functions needed to complete 1D1V PIC simulation
and obtain deliverables
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from numba import jit
from scipy import sparse as sp
from scipy.sparse import linalg as la

""" Basic Parameters """
PAUSE = 1.0 # for handling anomalies

NX = 33 # number of grid points - requirement
X_MIN = -np.pi
X_MAX = np.pi
L = X_MAX- X_MIN
DX = L/float(NX)
X_GRID = np.linspace(X_MIN,X_MAX,NX) # endpoint=True

OMEGA_P = 1.0
TAU_P = 2.0 * np.pi / OMEGA_P
EPS_0 = 1.0
Q_OVER_M = -1.0
T_ELCTRN = 1.0 # "[eV]", I don't think it really matters

""" Basic Objects """
RHO_j = np.zeros((NX,1),dtype=float) # Grid Charge Density
PHI_j = np.zeros((NX,1),dtype=float) # Grid Electrostatic Potential
E_j = np.zeros((NX,1),dtype=float) # Grid Electrostatic field

""" Functions """
def LaplacianStencil():
    """
    Finite difference approximation for Laplacian operator
    Output is provided to PotentialSolveES()
    B.Cs: Periodic
    Gauge: phi[0] = 0
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
    """
    global NX, DX, EPS_0
    v = np.ones(NX)
    diags = np.array([-1,0,1])
    vals = np.vstack((v,-2.0*v,v))
    Lmtx = sp.spdiags(vals,diags,NX,NX);
    Lmtx = sp.lil_matrix(Lmtx); # need to alter entries
    Lmtx[NX-1,:] = 0.0 # Part of gauge: phi_{0} = 0
    Lmtx[0,NX-1] = 1.0 # Periodic BCs
    Lmtx[NX-1,0] = 1.0 # Periodic BCs + Gauge
    LaplPreFactor = -DX**2 / EPS_0
    Lmtx /= LaplPreFactor
    Lmtx = sp.csr_matrix(Lmtx)
    return Lmtx

def FirstDerivativeStencil():
    """
    Finite (Central) Difference approximation of first derivative
    Output (sparse matrix) is used to solve for (irrotational) E_j given PHI_j
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        FDmtx - Nx x Nx matrix for calculating first derivative of potential
    """
    global NX
    v = np.ones(NX)
    diags = np.array([-1,0,1])
    vals = np.vstack((-v,0.0*v,v))
    FDmtx = sp.spdiags(vals,diags,NX,NX)
    FDmtx = sp.lil_matrix(FDmtx)
    FDmtx[1,0] = 0.0
    FDmtx[0,NX-1] = -1.0
    FDmtx /= 2.0*DX
    FDmtx = sp.csr_matrix(FDmtx)
    return FDmtx

def Initialize():
    """
    InitState - array
    """
    print("pmod.Initialize() executing ...")
    print("Please enter the number of particles to simulate")
    print("The number must be a power of 2 and cannot be larger than 64")

    N = int(input(''))

    if(N % 2 != 0):
        print("ERROR: %i is not a power of 2" %N)
        time.sleep(pause)
        AnomalyHandle()
    if(N > 64):
        print("ERROR: %i is larger than 64" %N)
        time.sleep(pause)
        AnomalyHandle()

    print("Do you want to initialize a velocity perturbation? 0 for no, 1 for yes")
    InitialV = int(input(''))
    if(InitialV != 0 and InitialV != 1):
        print("ERROR: value of %i is not 0 or 1" %InitialV)
        time.sleep(pause)
        AnomalyHandle()

    print("Please enter 0 for 0th-order charge/force weighting or 1 for 1st-order:")
    WeightingOrder = int(input(''))
    if(WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: The entry for charge/force weighting must be either 0 or 1")
        time.sleep(pause)
        AnomalyHandle()

    # I can't think of a better way to store this information
    InitState = [N, WeightingOrder, InitialV]
    print("pmod.Initialize() execution complete ...")
    return InitState

def Findj(x_i):
    """
    Binary search to find j for 1st-order weighting s.t x_j[j] < x_i < x_j[j+1]
    Inputs:
        x_i - Scalar (float64) variable representing superparticle position
        guess - Initial guess for jfound
        low - Initial lower bound for jfound
        high - Initial upper bound for jfound
    Outputs:
        jfound - the index that fulfills the condition described in fncn summary
    """
    global X_GRID
    i = 0 # counter
    low = 0
    high = np.size(X_GRID) - 1
    guess = int(np.floor((low+high)/2))
    while((x_i > X_GRID[guess] and x_i < X_GRID[guess + 1]) == False):
        # print("cycle count = %i" %i)
        if(X_GRID[guess] < x_i):
            low = guess
            guess = int(np.floor((low+high)/2))
        elif(X_GRID[guess] > x_i):
            high = guess
            guess = int(np.floor((low+high)/2))
        i += 1 # brakes
        if(i > int(np.sqrt(np.size(X_GRID)))):
            break
    return guess

def ParticleWeighting(WeightingOrder,x_i,q_sp):
    """
    Function to weight the particles to grid. Step #1 in PIC computational cycle
    Electrostatic -> no current at the moment.
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        x_i - N x 1 array containing the particle locations, i.e, particlesPositio
        q_sp - charge associated with the population of superparticles
    Outputs:
        RHO_j - Nx x 1 array containing the charge density at each grid point
    Return Code:
        0 = Hunky-dory
        -1 = FUBAR
    """
    global RHO_j, X_GRID, DX
    if (WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: Input weighting order appears to be out of bounds")
        print("LOCATION: ParticleWeighting() function ")
        return -1

    if WeightingOrder == 0:
        count = np.zeros((np.size(X_GRID),1),dtype=int)
        for j in np.arange(np.size(X_GRID)):
            for i in np.arange(np.size(x_i)):
                if (np.abs(X_GRID[j] - x_i[i]) < DX/2.0):
                    count[j] += 1
                RHO_j[j] = q_sp*float(count[j])/DX
    
    if WeightingOrder == 1:
        RHO_j[:] = 0.0
        for i in np.arange(np.size(x_i)): # Find j s.t. x_{j} < x_{i} < x_{j+1}
            jfound = Findj(x_i[i]) # binary search
            jfoundp1 = (jfound + 1) % NX
            RHO_j[jfound] = q_sp*(X_GRID[jfoundp1] - x_i[i])/DX
            RHO_j[jfoundp1] = q_sp*(x_i[i] - X_GRID[jfound])/DX

    # Add contribution of static, uniform ion background s.t plasma is quasineutra
    RHO_bg = -(DX/L)*(np.sum(RHO_j[1:(NX-2)]) + (RHO_j[0] + RHO_j[NX-1]))
    RHO_j = RHO_j + RHO_bg
    return 0

def PotentialSolveES(Lmtx):
    """
    Function to solve for the electric potential on the grid, phi_j
    Inputs:
        rho_j - Nx x 1 array containing the charge density at each grid point
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
        Nx - number of grid points
    Outputs:
        phi_j - Nx x 1 array containing the electric potential at each grid 
    """
    global RHO_j, PHI_j
    RHO_j[np.size(RHO_j)-1] = 0.0 # Part of gauge: phi_{0} = 0
    PHI_j = la.spsolve(Lmtx,RHO_j) # Field Solve
    PHI_j[np.size(PHI_j)-1] = PHI_j[0] # Periodic BCs
    return 0

def FieldSolveES(FDmtx):
    """
    Function to solve for the electric field on the grid, E_j.
    Inputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
        FDmtx - Nx x Nx matrix for calculating first derivative on the grid
    Outputs:
        E_j - Nx x 1 array containing the value of the  electric field at each grid point
    """
    global E_j, PHI_j
    E_j = FDmtx @ PHI_j
    E_j = -1.0*E_j # Don't forget the negative sign
    return 0

def ForceWeighting(WeightingOrder,x_i,E_i):
    """
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        E_i - N x 1 array containing the electric fields experienced by the particles
        x_i - N x 1 array containing the positions of the particles
    Outputs:
        E_i
    """
    global X_GRID, E_j, NX, DX
    if (WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: Input weighting order appears to be out of bounds")
        print("LOCATION: ForceWeighting() function ")
        return -1

    if WeightingOrder == 0:
        for i in np.arange(np.size(x_i,axis=0)):
            for j in np.arange(np.size(X_GRID,axis=0)):
                if (np.abs(X_GRID[j] - x_i[i]) < DX/2.0):
                    E_i[i] = E_j[j]
                    
    if WeightingOrder == 1:
        for i in np.arange(np.size(x_i)): # Find j s.t. x_{j} < x_{i} < x_{j+1}
            jfound = Findj(x_i[i])
            jfoundp1 = (jfound + 1) % NX # handles jfound == Nx-1 and returns jfound + 1 for all else
            E_i[i] = ((X_GRID[jfoundp1] - x_i[i])/DX)*E_j[jfound] + \
                ((x_i[i] - X_GRID[jfound])/DX)*E_j[jfoundp1]
    return E_i

"""
Legacy
def EulerStep(dt,E_i,v_i,qm):
    Euler half-step to start Leap-Frog method
    Inputs:
        dt - Time step, Advance is Forward or Backward depending on sgn(dt)
        E_i - N x 1 array containing the values of the electric field experienced
            by each of the different particles
        v_i - N x 1 array containing the velocity of the different particles at t = 0 = t_{0}
        qm - charge-to-mass ratio of the superparticle
    Outputs:
        v_i - N x 1 array containing the velocity of the different particles at t = dt/2 = t_{1/2}
    v_i = v_i + qm*dt*E_i
    return v_i
"""

def LeapFrog(N,x_i,v_i,E_i,dt,n):
    """
    Function to compute the particle advance using the Leapfrog algorithm
    Inputs:
    N - number of particles
        x_i - x_i(t_{n-1}), N x 1 array of particle positions at t = t_{n-1} = (n-1)*dt
        v_i - v_i(t_{n-1/2}), N x 1 array of particle velocities in 1D
        dt - Time step
        qm - charge-to-mass ratio of the superparticles
        n - Time level
        X_min - x_grid[0], for periodic
        X_max - x_grid[Nx-1], for periodic
    Outputs:
        x_i - x_i(t_{n}), N x 1 array of particles positions at t = t_{n} = n*dt
        v_i - v_i(t_{n+1/2})
    """
    global Q_OVER_M
    for pidx in np.arange(N):
        if n == 0: # Initial step requires velocity half-step backwards
            v_i[pidx] = v_i[pidx] - 0.5*dt*Q_OVER_M*E_i[pidx]
        else:
            v_i[pidx] = v_i[pidx] + dt*Q_OVER_M*E_i[pidx]
        x_i[pidx] = x_i[pidx] + dt*v_i[pidx]
    """
    Legacy
    if n == 0: # First step requires an initial velocity half-step to begin
        for pidx in np.arange(N):
            v_i[pidx] = EulerStep(-dt/2.0,E_i[pidx],v_i[pidx],qm)# half-step backwards gives v_i(t_{-1/2}) to begin
    v_i = v_i + qm*dt*E_i # v_i(t_{n+1/2}) = v_i(t_{n-1/2}) + (q/m)*dt*E_i
    x_i = x_i + dt*v_i # x_i(t_{n+1}) = x_i(t_{n}) + dt*v_i(t_{n+1/2})
    for pidx in np.arange(np.size(x_i)):
        if(x_i[pidx] > X_max): # particle exited grid via right boundary
            Dx = x_i[pidx] - X_max
            x_i[pidx] = X_min + Dx
        if(x_i[pidx] < X_min):# particle exited grid via left boundary
            Dx = x_i[pidx] - X_min
            x_i[pidx] = X_max + Dx
    """
    return x_i, v_i

""" Diagnostics """
def ComputeElectricFieldEnergy():
    """
    Compute the grid-integrated electric field energy
    """
    global DX, NX, E_j
    E_j_sq = np.square(E_j)
    PEgrid = DX * 0.5 * (0.5*(E_j[0]**2 + E_j[NX-1]**2)+np.sum(E_j_sq[1:(NX-2)]))
    return PEgrid

def ComputeKineticEnergy(v_i,m_sp):
    """ 
    Compute the system kinetic energy
    """
    KE = 0.5*m_sp*np.sum(np.square(v_i))
    return KE
                            
""" Helper Functions """

def AddHex(hex1,hex2):
    """
    hex1 - str representing first hexadecimal number
    hex2 - str representing second hexadecimal number
    Assumes hexadecimal in format: hexnumber = "0x******"
    """
    if(hex1[0] != '0' and hex1[1] != 'x'):
        print("First input to AddHex() is incorrectly formatted")
        AnomalyHandle()
    if(hex2[0] != '0' and hex2[1] != 'x'):
        print("Second input to AddHex() is incorrectly formatted")
        AnomalyHandle()
    int1 = int(hex1,16)
    int2 = int(hex2,16)
    sum = int1 + int2
    return(hex(sum))

def FormatHex(hex2f):
    """
    Format str representing hexadecimal number for usage as argument in pyplot.scatter()
    Take "0x******" to "#******"
    """
    if(hex2f[0] != '0' and hex2f[1] != 'x'):
        print("Input to FormatHex() is incorrectly formatted")
        AnomalyHandle()
    hex2f = "#" + hex2f[2:]
    if(len(hex2f) == 6):
        hex2f = hex2f + "0"
    if(len(hex2f) == 8):
        hex2f = hex2f[0:7]
    return hex2f

def UnFormatHex(hex2unf):
    """
    """
    hex2unf = "0x" + hex2unf[2:]
    return hex2unf

def AnomalyHandle():
    print("Please rerun the program")
    time.sleep(pause)
    sys.exit("Exiting ...")
