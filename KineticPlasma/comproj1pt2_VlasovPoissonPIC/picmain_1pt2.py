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

""" Grid Generation """
print("Generating grid")
