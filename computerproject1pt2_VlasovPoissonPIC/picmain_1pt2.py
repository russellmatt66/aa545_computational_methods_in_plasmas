"""
Matt Russell
University of Washington
AA545: Computational Methods for Plasmas
Computer Project 1.2: Kinetic Modeling: Vlasov-Poisson PIC
main.py
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
particleField = np.empty((N,1),dtype=float) # array of fields experienced by particles, E_i

# Initial Conditions for 1.
if (N == 2):
    particlesPosition[0] = -np.pi/4.0
    particlesPosition[1] = np.pi/4.0
    particlesVelocity[0] = 0.0
    particlesVelocity[1] = 0.0

print("Closing Initialization Phase ...")
""" Grid Generation Phase """
print("Opening Grid Generation Phase ...")

x_min = -np.pi
x_max = np.pi
dx = (x_max - x_min)/float(Nx)
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
E_j = np.empty((Nx,1),dtype=float) # Grid Electric Field
phi_j = np.empty((Nx,1),dtype=float) # Grid Potential
rho_j = np.empty((Nx,1),dtype=float) # Grid Charge Density

print("Closing Grid Generation Phase")
""" PIC Phase """
print("Beginning PIC Simulation")
