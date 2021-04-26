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
from scipy import sparse as sp
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt
import time
import sys

""" Basic Parameters """
pause = 1.0
e = 1.0 # normalized fundamental unit of electric charge
q_e = -e # electron charge
q_i = e # ion charge

def Initialize():
    """
    InitState - 3x1 column vector containing:
        N = InitState[0], the number of particles to be used in the simulation
        Nx = InitState[1], the number of grid points " " " " " "
        WeightingOrder = InitState[2], Z \in {0,1} representing 0th- or 1st-
                                        order weighting for charge and forces.
    """
    print("pmod.Initialize() executing ...")
    # I can't think of a better way to store this information

    InitState = np.empty((3,1),dtype=int)
    print("Please enter the number of particles to simulate, must be a power of 2:")
    InitState[0] = int(input(''))
    if(InitState[0] % 2 != 0):
        print("ERROR: %i is not a power of 2" %InitState[0])
        time.sleep(pause)
        AnomalyHandle()

    print("Please enter the desired number of grid CELLS for the mesh (must be a power of 2).")
    print("Note that this will be one less than the number of grid POINTS:")
    InitState[1] = int(input(''))
    if(InitState[1] % 2 != 0):
        print("ERROR: %i is not a power of 2" %InitState[1])
        time.sleep(pause)
        AnomalyHandle()

    print("Please enter 0 for 0th-order charge/force weighting or 1 for 1st-order:")
    InitState[2] = int(input(''))
    if(InitState[2] != 0 and InitState[2] != 1):
        print("ERROR: The entry for charge/force weighting must be either 0 or 1")
        time.sleep(pause)
        AnomalyHandle()

    print("pmod.Initialize() execution complete ...")
    print(InitState)
    return InitState

"""
Grid Generation: Left out for moment due to simplicity
"""
# def GridGeneration():

def ParticleWeighting(WeightingOrder,x_i,x_j,rho_j,dx,N):
    """
    Function to weight the particles to grid. Step #1 in PIC general procedure.
    Electrostatic -> no velocity or current at the moment.
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        x_i - N x 1 array containing the particle locations, i.e, particlesPosition
        x_j - Nx x 1 array representing the spatial grid
        rho_j - Nx x 1 array containing the charge density at each grid point
        dx - grid spacing
        N - Number of Particles
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
        count = np.empty((np.size(x_j),1),dtype=int)
        for j in np.arange(np.size(x_j)):
            for i in np.arange(np.size(x_i)):
                if (np.abs(x_j[j] - x_i[i]) < dx/2.0):
                    count[j] += 1
            rho_j[j] = q_e*float(count[j])/dx

    if WeightingOrder == 1:
        for i in np.arange(np.size(x_i)): # Find j s.t. x_{j} < x_{i} < x_{j+1}
            for j in np.arange(np.size(x_j)): # Search algorithm here could be better
                if x_j[j] < x_i[i] and x_i[i] < x_j[j+1]:
                    break
            rho_j[j] = q_e*(x_j[j+1] - x_i[i])/dx
            rho_j[j+1] = q_e*(x_i[i] - x_j[j])/dx

    # Add contribution of static ion background
    rho_j = rho_j + q_i*float(N)/float(x_j[np.size(x_j)-1]-x_j[0])
    return rho_j

def PotentialSolveES(rho_j,Lmtx,Nx):
    """
    Function to solve for the electric potential on the grid, phi_j
    Inputs:
        rho_j - Nx x 1 array containing the charge density at each grid point
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
        Nx - number of grid points
    Outputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
    """
    phi_j = la.spsolve(Lmtx,rho_j)
    return phi_j

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
    return E_j

def LaplacianStencil(Nx,dx):
    """
    Output is used as an argument for PotentialSolve()
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        Lmtx - Nx x Nx matrix for calculating Laplacian on the grid
    Governing Equations:
        (1) Gauss' Law -> Poisson's Equation
    B.Cs: Periodic
    Gauge: phi[0] = 0
    Discretization: Finite Difference
    """
    v = np.ones(Nx)
    diags = np.array([-1,0,1])
    vals = np.vstack((v,-2.0*v,v))
    Lmtx = sp.spdiags(vals,diags,Nx,Nx);
    Lmtx = sp.lil_matrix(Lmtx); # need to alter entries
    Lmtx[0,0] = 0.0
    Lmtx[1,0] = 0.0
    Lmtx[Nx-1,0] = 0.0
    Lmtx /= dx**2
    Lmtx = sp.csr_matrix(Lmtx)
    return Lmtx

def FirstDerivativeStencil(Nx,dx):
    """
    Output is used as an argument for FieldSolve()
    Inputs:
        Nx - Number of grid points
        dx - Grid spacing
    Outputs:
        FDmtx - Nx x Nx matrix for calculating first derivative of potential
    Governing Equations:
        (1) E = -grad(phi) => E = -d(phi)/dx
    Discretization: Central Difference
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

def ForceWeighting(WeightingOrder,dx,E_i,E_j,x_i,x_j):
    """
    Inputs:
        WeightingOrder - {0,1}, information the program uses to determine whether
                        to use 0th or 1st order weighting
        E_i - N x 1 array containing the electric fields experienced by the particles
        dx - Grid Spacing
        E_j - Nx x 1 array containing the values of the electric field on the grid
        x_i - N x 1 array containing the positions of the particles
        x_j - Nx x 1 array containing the grid points
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
            for j in np.arange(np.size(x_j)): # Search algorithm here could be better
                if x_j[j] < x_i[i] and x_i[i] < x_j[j+1]:
                    break
            E_i[i] = ((x_j[j+1] - x_i[i])/dx)*E_j[j] + ((x_i[i] - x_j[j])/dx)*E_j[j+1]

    return E_i

def EulerStep(dt,E_i,v_i,qm):
    """
    Function used for half-step to start Leap-Frog method
    Inputs:
        dt - Time step
        E_i - N x 1 array containing the values of the electric field experienced
            by each of the different particles
        v_i - N x 1 array containing the velocity of the different particles at t = 0 = t_{0}
        qm - charge-to-mass ratio of the superparticle
    Outputs:
        v_i - N x 1 array containing the velocity of the different particles at t = dt/2 = t_{1/2}
    """
    v_i = v_i + qm*dt*E_i
    return v_i

def LeapFrog(x_i,v_i,E_i,dt,qm,n):
    """
    Function to compute the particle advance using the Leapfrog algorithm
    Inputs:
        x_i - x_i(t_{n-1}), N x 1 array of particle positions at t = t_{n-1} = (n-1)*dt
        v_i - v_i(t_{n-1/2}), N x 1 array of particle velocities in 1D
        dt - Time step
        qm - charge-to-mass ratio of the superparticles
        n - Time level
    Outputs:
        x_i - x_i(t_{n}), N x 1 array of particles positions at t = t_{n} = n*dt
        v_i - v_i(t_{n+1/2})
    """
    if n == 0: # First step requires a half-step for the velocity
        v_i = EulerStep(dt/2.0,E_i,v_i,qm)# dt -> dt/2.0 so that it's a half-step
        x_i = x_i + dt*v_i
    v_i = v_i + qm*dt*E_i # v_i(t_{n+1/2}) = v_i(t_{n-1/2}) + (q/m)*dt*E_i
    x_i = x_i + dt*v_i # x_i(t_{n+1}) = x_i(t_{n}) + dt*v_i(t_{n+1/2})
    return x_i, v_i

def Diagnostics():
    return 0

def AnomalyHandle():
    print("Please rerun the program")
    time.sleep(pause)
    sys.exit("Exiting")
