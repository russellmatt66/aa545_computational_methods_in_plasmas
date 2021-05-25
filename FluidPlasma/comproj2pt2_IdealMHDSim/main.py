"""
Matt Russell
University of Washington Aero & Astro
Computer Project 2.2
5/19/21
Main program file for project to simulate screw-pinch with 3D, time-dependent
MHD code
"""
import numpy as np
import matplotlib.pyplot as plt
import fluxfcndefn as flux

""" Grid Generation """
spacing = 1.0 # Grid spacing - relative to radius of the plasma
dx = spacing
dy = spacing
dz = spacing
num_Nodes = 100
Nx = Nodes
Ny = Nodes
Nz = Nodes

num_ConservedQuantities = 8

SimulationData = np.zeros((Nx,Ny,Nz,num_ConservedQuantities))

""" """
# L = float(spacing*num_Nodes)
a = 1.0 # isotropic advection speed for information propagation
dt = 0.1 # CFL condition should be satisfied

""" Initial Conditions """
