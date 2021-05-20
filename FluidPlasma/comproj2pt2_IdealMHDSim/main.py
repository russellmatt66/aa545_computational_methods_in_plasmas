"""
Matt Russell
University of Washington Aero & Astro
Computer Project 2.2
5/19/21
Main program file for project.
"""
import numpy as np
import matplotlib.pyplot as plt
import fluxfcndefn as flux

""" Grid Generation """
r_plasma = 1.0
spacing = 0.01*r_plasma # Grid spacing - relative to radius of the plasma
dx = spacing
dy = spacing
dz = spacing
Nodes = int(3.0*(2.0*r_plasma/spacing)) # Factor out front should give the number of plasma diameters to fit into the grid
Nx = Nodes
Ny = Nodes
Nz = Nodes 
