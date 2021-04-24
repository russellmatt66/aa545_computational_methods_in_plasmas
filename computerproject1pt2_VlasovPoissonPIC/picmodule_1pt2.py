"""
Matt Russell
University of Washington
AA545: Computational Methods for Plasmas
Computer Project 1.2: Kinetic Modeling: Vlasov-Poisson PIC
picmodule.py
Module that contains the functions needed to complete 1D1V PIC simulation
and obtain deliverables
"""
import numpy as np
import scipy as sp
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

    print("Please enter the desired number of grid cells for the mesh, must be a power of 2:")
    InitState[1] = int(input(''))
    if(InitState[1] % 2 != 0):
        print("ERROR: %i is not a power of 2" %InitState[1])
        time.sleep(pause)
        AnomalyHandle()

    print("Please enter 0 for 0th-order charge/force weighting or 1 for 1st-order:")
    InitState[2] = int(input(''))
    if(InitState[2] != 0 or InitState[2] != 1):
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

def ParticleWeighting(WeightingOrder,dx,x_i,x_j,rho_j):
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

def PotentialSolve(rho_j,Nx):
    """
    Function to solve for the electric potential on the grid, phi_j
    Inputs:
        rho_j - Nx x 1 array containing the charge density at each grid point
        Nx - number of grid points
    Outputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
    Governing Equation: Gauss' Law
    B.Cs: Periodic
    Gauge: phi[0] = 0
    Discretization: Finite Difference
    """
    # Not good that I end up reassining these each time function is called
    # but I need to know Nx to create them in the first place
    v = np.ones(Nx)
    diags = np.array([-1,0,1])
    vals = np.vstack((v,-2.0*v,v))



def FieldSolve(phi_j):
    """
    Function to solve for the electric field on the grid, E_j.
    Inputs:
        phi_j - Nx x 1 array containing the electric potential at each grid point
    Outputs:
        E_j - Nx x 1 array containing the value of the  electric field at each grid point
    """


def Diagnostics():
    return 0

def AnomalyHandle():
    print("Please rerun the program")
    time.sleep(pause)
    sys.exit("Exiting")
