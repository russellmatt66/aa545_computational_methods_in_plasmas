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
pause = 1.0

def Initialize():
    """
    InitState - 4x1 column vector containing:
        N = InitState[0], the number of particles to be used in the simulation
        Ncells = InitState[1], the number of grid cells " " " " " "
        WeightingOrder = InitState[2], \textit{a} \in {0,1} representing the choice of
                                        0th- or 1st-order weighting for charge and forces.
        InitialV = InitState[3], flag for whether to initialize a velocity perturbation on the particle distribution
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

    Ncells = 32 # Number of grid cells. NumGridPoints: Nx = Nc + 1

    print("Please enter 0 for 0th-order charge/force weighting or 1 for 1st-order:")
    WeightingOrder = int(input(''))
    if(WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: The entry for charge/force weighting must be either 0 or 1")
        time.sleep(pause)
        AnomalyHandle()

    # I can't think of a better way to store this information
    InitState = np.empty((4,1),dtype=int)
    InitState[0] = N
    InitState[1] = Ncells
    InitState[2] = WeightingOrder
    InitState[3] = InitialV
    print("pmod.Initialize() execution complete ...")
    return InitState

def Findj(x_j,x_i):
    """
    Binary search to find j for 1st-order weighting s.t x_j[j] < x_i < x_j[j+1]
    Inputs:
        x_j - Nx x 1 ndarray representing spatial grid
        x_i - Scalar (float64) variable representing superparticle position
        guess - Initial guess for jfound
        low - Initial lower bound for jfound
        high - Initial upper bound for jfound
    Outputs:
        jfound - the index that fulfills the condition described in fncn summary
    """
    i = 0 # counter
    low = 0
    high = np.size(x_j) - 1
    guess = int(np.floor((low+high)/2))
    while((x_i > x_j[guess] and x_i < x_j[guess + 1]) == False):
        # print("cycle count = %i" %i)
        if(x_j[guess] < x_i):
            low = guess
            guess = int(np.floor((low+high)/2))
        elif(x_j[guess] > x_i):
            high = guess
            guess = int(np.floor((low+high)/2))
        i += 1 # brakes
        if(i > int(np.sqrt(np.size(x_j)))):
            break
    # print("Search complete, index found is %i in %i iterations" %(guess,i))
    # time.sleep(pause)
    return guess

def ParticleWeighting(WeightingOrder,x_i,N,x_j,Nx,dx,L,rho_j,q_sp):
    """
    Function to weight the particles to grid. Step #1 in PIC general procedure.
    Electrostatic -> no current at the moment.
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        x_i - N x 1 array containing the particle locations, i.e, particlesPosition
        N - Number of Particles
        x_j - Nx x 1 array representing the spatial grid
        Nx - Number of grid points
        dx - grid spacing
        L - length of grid
        rho_j - Nx x 1 array containing the charge density at each grid point
        q_sp - charge associated with the population of superparticles
    Outputs:
        rho_j - Nx x 1 array containing the charge density at each grid point
    Return Code:
        -1 - FUBAR
    """
    if (WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: Input weighting order appears to be out of bounds")
        print("LOCATION: ParticleWeighting() function ")
        return -1

    # dx = (x_j[np.size(x_j)-1] - x_j[0])/float(np.size(x_j))

    if WeightingOrder == 0:
        count = np.zeros((np.size(x_j),1),dtype=int)
        for j in np.arange(np.size(x_j)):
            for i in np.arange(np.size(x_i)):
                if (np.abs(x_j[j] - x_i[i]) < dx/2.0):
                    count[j] += 1
                rho_j[j] = q_sp*float(count[j])/dx

    if WeightingOrder == 1:
        rho_j[:] = 0.0
        for i in np.arange(np.size(x_i)): # Find j s.t. x_{j} < x_{i} < x_{j+1}
            jfound = Findj(x_j,x_i[i]) # binary search
            jfoundp1 = (jfound + 1) % Nx
            rho_j[jfound] = q_sp*(x_j[jfoundp1] - x_i[i])/dx
            rho_j[jfoundp1] = q_sp*(x_i[i] - x_j[jfound])/dx

    # Add contribution of static, uniform ion background s.t plasma is quasineutra
    rho_background = -(dx/L)*(np.sum(rho_j[1:(Nx-2)]) + (rho_j[0] + rho_j[Nx-1]))
    rho_j = rho_j + rho_background
    return rho_j

def LaplacianStencil(Nx,dx,eps_0):
    """
    Output is used as an argument for PotentialSolve()
    B.Cs: Periodic
    Gauge: phi[0] = 0
    Discretization: Finite Difference
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
    Governing Equations:
        (1) Gauss' Law -> Poisson's Equation
    """
    v = np.ones(Nx)
    diags = np.array([-1,0,1])
    vals = np.vstack((v,-2.0*v,v))
    Lmtx = sp.spdiags(vals,diags,Nx,Nx);
    Lmtx = sp.lil_matrix(Lmtx); # need to alter entries
    Lmtx[0,:] = 0.0
    Lmtx[0,0] = -1.0 # Gauge of phi_{0] = 0 asserted in PotentialSolve()
    Lmtx[Nx-1,0] = 1.0
    LaplPreFactor = dx**2 / eps_0
    Lmtx /= LaplPreFactor
    Lmtx = sp.csr_matrix(Lmtx)
    return Lmtx

def PotentialSolveES(Lmtx,phi_j,rho_j):
    """
    Function to solve for the electric potential on the grid, phi_j
    Inputs:
        rho_j - Nx x 1 array containing the charge density at each grid point
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
        Nx - number of grid points
    Outputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
    """
    # phi_j[0] = 0.0 # phi_{0} = 0
    phi_j = la.spsolve(Lmtx,rho_j)
    phi_j = -1.0*phi_j # don't forget the negative sign
    return phi_j

def FirstDerivativeStencil(Nx,dx):
    """
    Output is used as an argument for FieldSolve()
    Governing Equations:
        (1) E = -grad(phi) => E = -d(phi)/dx
    Discretization: Central Difference
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        FDmtx - Nx x Nx matrix for calculating first derivative of potential
    """
    v = np.ones(Nx)
    diags = np.array([-1,0,1])
    vals = np.vstack((-v,0.0*v,v))
    FDmtx = sp.spdiags(vals,diags,Nx,Nx)
    FDmtx = sp.lil_matrix(FDmtx)
    FDmtx[1,0] = 0.0
    FDmtx[0,Nx-1] = -1.0
    FDmtx /= 2.0*dx
    FDmtx = sp.csr_matrix(FDmtx)
    return FDmtx

def FieldSolveES(phi_j,FDmtx):
    """
    Function to solve for the electric field on the grid, E_j.
    Inputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
        FDmtx - Nx x Nx matrix for calculating first derivative on the grid
    Outputs:
        E_j - Nx x 1 array containing the value of the  electric field at each grid point
    """
    E_j = FDmtx @ phi_j
    E_j = -1.0*E_j # Don't forget the negative sign
    return E_j

def ForceWeighting(WeightingOrder,x_i,E_i,x_j,Nx,dx,E_j):
    """
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        E_i - N x 1 array containing the electric fields experienced by the particles
        dx - Grid Spacing
        E_j - Nx x 1 array containing the values of the electric field on the grid
        x_i - N x 1 array containing the positions of the particles
        x_j - Nx x 1 array containing the grid points
        Nx - Number of grid points
    Outputs:
        E_i
    """
    if (WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: Input weighting order appears to be out of bounds")
        print("LOCATION: ForceWeighting() function ")
        return -1

    if WeightingOrder == 0:
        for i in np.arange(np.size(x_i,axis=0)):
            for j in np.arange(np.size(x_j,axis=0)):
                if (np.abs(x_j[j] - x_i[i]) < dx/2.0):
                    E_i[i] = E_j[j]

    if WeightingOrder == 1:
        for i in np.arange(np.size(x_i)): # Find j s.t. x_{j} < x_{i} < x_{j+1}
            jfound = Findj(x_j,x_i[i])
            jfoundp1 = (jfound + 1) % Nx # handles jfound == Nx-1 and returns jfound + 1 for all else
            E_i[i] = ((x_j[jfoundp1] - x_i[i])/dx)*E_j[jfound] + ((x_i[i] - x_j[jfound])/dx)*E_j[jfoundp1]
            # for j in np.arange(np.size(x_j)): # Search algorithm here could be better
            #     if (x_j[j] < x_i[i] and x_i[i] < x_j[j+1]):

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

def LeapFrog(N,x_i,v_i,E_i,dt,q_over_m,n,X_min,X_max):
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
    for pidx in np.arange(N):
        if n == 0: # Initial step requires velocity half-step backwards
            v_i[pidx] = v_i[pidx] - 0.5*dt*q_over_m*E_i[pidx]
        else:
            v_i[pidx] = v_i[pidx] + dt*q_over_m*E_i[pidx]
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
def ComputeElectricFieldEnergy(E_j,Nx,dx):
    """
    Compute the grid-integrated electric field energy
    """
    E_j_sq = np.square(E_j)
    PEgrid = dx*0.5*(0.5*(E_j[0]**2 + E_j[Nx-1]**2)+np.sum(E_j_sq[1:(Nx-2)]))
    return PEgrid

def ComputeKineticEnergy(v_i,m_sp):
    """ 
    Compute the system kinetic energy
    """
    KE = 0.5*m_sp*np.sum(np.square(v_i))
    return KE
                            
def GridIntegrate(E_j,Nx,dx,v_i,m_sp):
    """
    Compute the grid-integrated electric field energy, kinetic energy, and their
    sum
    """
    # Efgrid = 0.0
    # KineticEnergy = 0.0
    # ETotal = 0.0
    # for j in np.arange(np.size(E_j)):
    #     Efgrid = Efgrid + E_j[j]*E_j[j]/2.0
    # print("Grid-integrated Electric field energy is %f" %Efgrid)
    # for i in np.arange(np.size(v_i)):
    #     KineticEnergy = KineticEnergy + 0.5*v_i[i]*v_i[i]
    # print("System kinetic energy is %f" %KineticEnergy)
    """ Grid-Integrated """
    E_j_squared = np.square(E_j)
    Efgrid = dx*0.5*(0.5*(E_j[0]**2 + E_j[Nx-1]**2) + np.sum(E_j_squared[1:(Nx-2)]))
    KineticEnergy = 0.5*m_sp*np.sum(np.square(v_i))
    ETotal = Efgrid + KineticEnergy
    return Efgrid,KineticEnergy,ETotal

# def ComputeOscillationFrequency(x_n,x_np1):
#     """
#     """
#     if(x_n)

# def Diagnostics(E_j,v_i,n,**ax):
#     """
#     Function to implement appropriate diagnostic. Don't know how to pass axes
#     object to function as an argument.
#     """
#     Egrid = 0.0
#     KineticEnergy = 0.0
#     # if n == 0:
#         # diagFig, (axKin, axE, axTot) = plt.subplots(nrows=3,ncols=1)
#     for j in np.arange(np.size(E_j)):
#         Egrid = Egrid + E_j[j]*E_j[j]/2.0
#     ax[0].scatter(n,Egrid)
#     for i in np.arange(np.size(v_i)):
#         KineticEnergy = KineticEnergy + 0.5*v_i[i]*v_i[i]
#     ax[1].scatter(n,KineticEnergy)
#     ax[2].scatter(n,Egrid+KineticEnergy)
#     return 0

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
