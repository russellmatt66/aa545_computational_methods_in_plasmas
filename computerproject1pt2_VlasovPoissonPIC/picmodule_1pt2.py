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
import matplotlib.pyplot as plt
import time
import sys

pause = 1.0
q = 1.0 # normalized fundamental unit of electric charge

def Initialize():
    """
    InitState - 3x1 column vector containing:
        N = InitState[0], the number of particles to be used in the simulation
        Nx = InitState[1], the number of grid points " " " " " "
        WeightingOrder = InitState[2], Z \in {0,1} representing 0th- or 1st-
                                        order weighting for charge and forces.
    """
    print("Opening Initialization Phase ...")
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

    print("Closing Initialization Phase ...")
    print(InitState)
    return InitState

"""
Grid Generation: Left out for moment due to simplicity
"""
# def GridGeneration():

def ParticleWeighting(WeightingOrder,x_i,x_j,rho_j):
    """
    Function to weight the particles to grid. Step #1 in PIC general procedure.
    Electrostatic -> no velocity or current at the moment.
    Inputs:
        WeightingOrder - {0,1}, information program uses to determine whether
                        to use 0th or 1st order weighting
        x_i - N x 1 array containing the particle locations
        x_j - Nx x 1 array representing the spatial grid
        rho_j - Nx x 1 array containing the charge density at each grid point
    Return Values:
        1 = Hunky Dory
        -1 = FUBAR
    """
    if (WeightingOrder != 0 and WeightingOrder != 1):
        print("ERROR: Input weighting order appears to be out of bounds\n")
        print("LOCATION: ParticleWeighting() function ")
        return -1

    if WeightingOrder == 0:
        dx = (x_j[np.size(x_j)-1] - x_j[0])/float(np.size(x_j))
        count = np.empty((np.size(x_j),1),dtype=int)
        for j in np.arange(np.size(x_j)):
            for i in np.arange(np.size(x_i)):
                if (np.abs(x_j[j] - x_i[i]) < dx/2.0):
                    count[j] += 1
            rho_j[j] = -q*float(count[j])/dx

    # Add contribution of static ion background
    rho_j = rho_j + q*float(Nx)/float(x_j[np.size(x_j)-1]-x_j[0])
    return 1

def Diagnostics():
    return 0

def AnomalyHandle():
    print("Please rerun the program")
    time.sleep(pause)
    sys.exit("Exiting")
