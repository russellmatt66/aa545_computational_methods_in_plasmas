"""
Matt Russell
University of Washington Aero & Astro
Computer Project 2.2
5/19/21
File containing the various unit tests, visualizations, and analysis
done for the project
"""
import numpy as np
import matplotlib.pyplot as plt

""" Conversion Factors """
amutokg = 1.67e-27 # "1 amu = 'amutokg' kilograms"
eVtoJoules = 1.6e-19 # "1 eV = 'eVtoJoules' Joules"

""" Constants """
k_b = 1.38e-23 # [J/K] - Boltzmann Constant
m_H = 1.008*amutokg # [kg]<-[amu] - Mass of Hydrogen Atom (ion)

""" Simulation Parameters """
spacing = 1.0
numNodes = 100
L = int(spacing*numNodes)

""" Visualize Axial Current Distribution """
# JFig = plt.figure()
# R = 1.0
# Nr = 1000
# r = np.linspace(-R,R,Nr)
# J0 = 1.0 # Magnitude of Current Density
#
# J1 = J0*(1.0 - (r/R)**2)
# J2 = J0*(1.0 - r/R)**2
#
# plt.plot(r,J1,label='J1')
# plt.plot(r,J2,label='J2')
# plt.title('Candidate Current Distributions')
# plt.xlabel('x')
# plt.ylabel('Current Density')
# plt.legend()

""" Visualize scale of pressure-density relationship """
# pressrhoFig = plt.figure()
# NTemp = 1000
# Temp = np.logspace(0,5,NTemp) # [eV]
#
# pressure_over_rho = (2.0*k_b/m_H)*eVtoJoules*Temp
#
# plt.plot(Temp,pressure_over_rho,label='Ideal Gas')
# plt.xlabel('T [eV]')
# plt.ylabel('pressure_over_rho')

""" Visualize J0sq-B0sq relationship """
B0sqvJ0sqFig = plt.figure()
numTest = 100
J0sq = np.logspace(-1,1)
R = int(np.floor(L/8))
B0sq = (1.0 - (4.0/9.0)*(R**2/6.0))*J0sq

plt.plot(J0sq,B0sq)

plt.show()
