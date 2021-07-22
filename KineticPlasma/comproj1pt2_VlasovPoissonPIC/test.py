"""
Matt Russell
AA545: Computational Methods for Plasmas
Computer Project 1.2: Kinetic Modeling: Vlasov-Poisson PIC
test.py
Script to test code
"""
from scipy import sparse as sp
import picmodule_1pt2 as pmod
import numpy as np
import matplotlib.pyplot as plt

Nx = 33
x_min = -np.pi
x_max = np.pi
L = x_max - x_min
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
dx = (x_max - x_min)/float(Nx)
N = 4
x_i = np.zeros((N,1),dtype=float)
v_i = np.zeros((N,1),dtype=float)

# Normalization
omega_p = 1.0
tau_p = 2.0 * np.pi / omega_p
eps_0 = 1.0 # vacuum permittivity
q_over_m = -1.0 # charge to mass ratio of superparticle
q_sp = (eps_0 * L / N) * (1.0 / q_over_m) # charge of a superparticle
m_sp = (N / L) * q_sp**2
Te = 1.0 # [eV]
lambda_De = np.sqrt(Te * N * (q_over_m**2) / (eps_0 * L))
v_th = omega_p * lambda_De

""" Test Findj/B.Cs """
# Conclusion: Findj works correctly for the most part
# Exception: When x_i is located exactly on a grid point
# xfind = x_max - 0.5*dx
x_to_find = np.zeros((Nx,1),dtype=float)
for xidx in np.arange(np.size(x_to_find)):
    x_to_find[xidx] = x_min + (0.5 + float(xidx))*dx
    jfound = pmod.Findj(x_grid,x_to_find[xidx])
    jfoundp1 = (jfound +1) % Nx
    print("For x_i = %f, jfound is %i and jfoundp1 is %i" %(x_to_find[xidx],jfound,jfoundp1))

""" Test particle distribution Initialization for N = 64 """
"""
vprime = 0.01*v_th
for pidx in np.arange(N):
    particlesPosition[pidx] = x_min/2.0 + float(pidx)*x_max/float(N-1)
    particlesVelocity[pidx] = vprime*np.sin(2.0*np.pi/L * particlesPosition[pidx])

PhaseSpaceFig = plt.figure()
plt.figure(PhaseSpaceFig.number)
plt.scatter(particlesPosition,particlesVelocity)
plt.xlabel('x')
plt.ylabel('v (normalized to $v_{th}$)')
plt.xlim((x_min,x_max))
plt.ylim((-2.0,2.0))
"""

""" Test Hexadecimal functions for N = 64 """
# particleColors = [None] * N
# black = '0x000000'
# white = '0xffffff'
# hexIncrement = hex(int((int(white,16)-int(black,16))/(N-1)))
# if(len(hexIncrement) != 8): # LSB = 0 was dropped and will cause expection later
#     hexIncrement = hexIncrement + "0" # so add it back
# print(hexIncrement)
# hexColor = '0x000000'
#
# for cidx in np.arange(N):
#     particleColors[cidx] = pmod.FormatHex(hexColor)
#     hexColor = pmod.AddHex(hexColor,hexIncrement)
#
# print(particleColors)
""" Test Zero-Crossing """
# testTrajectory = np.sin(2.0*np.pi*x_grid)
# x_n = 0.0
# x_np1 = 0.0
# for idx in np.arange(np.size(x_grid)-1):
#     x_n = float(testTrajectory[idx])
#     x_np1 = float(testTrajectory[idx+1])
#     if(x_n > 0.0 and x_np1 < 0.0): # Check for Zero-Crossing
#         print("Zero-Crossing detected")
#     elif(x_n < 0.0 and x_np1 > 0.0): # Check for Zero-Crossing
#         print("Zero-Crossing detected")

""" Test appending (for oscillation frequency calculation) """
# Ntests_append = 10
# tauOscillation = np.zeros((N,1)) # axis=1 will be the jth oscillation period
# tauTemp = np.ones((N,1))
# print(tauOscillation)
# a = np.hstack((tauOscillation,tauTemp))
# print(a)
# for n in np.arange(Ntests_append):
#     tauOscillation = np.hstack((tauOscillation,tauTemp))
#     print(tauOscillation)

""" Test Findj() """
# Ntests_Findj = 10
# x_i_test = (x_max - x_min)*np.random.rand(Ntests_Findj) + x_min
# # low = 0
# # high = np.size(x_grid) - 1 # looking for index
# # guess = int(np.floor((low + high)/2))
# # print("x_j[j] is %f" %x_grid[guess])
# # print("x_j[j+1] is %f" %x_grid[guess+1])
# for i in np.arange(Ntests_Findj):
#     print("Test %i beginning" %(i+1))
#     print("x_i is %f" %x_i_test[i])
#     # print(x_i_test[i] < x_grid[guess])
#     # print(x_i_test[i] < x_grid[guess+1])
#     jfound_test = pmod.Findj(x_grid,x_i_test[i])

""" Test AddHex() and FormatHex()"""
# hexA = '0x100d00'
# hexB = '0x020f00'
# print(pmod.AddHex(hexA,hexB))
# print(pmod.FormatHex(hexA))
# print(hexA)

""" Test particleColors[:] creation """
# N = 2
# particleColors = [None] * N
# black = '0x000000'
# white = '0xffffff'
# hexIncrement = hex(int((int(white,16)-int(black,16))/(N-1)))
# print(hexIncrement)
# print(type(hexIncrement))
# hexColor = '0x000000'
#
# for cidx in np.arange(N):
#     print(hexColor)
#     particleColors[cidx] = hexColor
#     print(particleColors[cidx])
#     hexColor = pmod.AddHex(hexColor,hexIncrement)
#
# print(particleColors)

""" Trying to observe Lmtx structure """
# Lmtx is exactly singular on first iteration 4 some reason
# Lmtx = pmod.LaplacianStencil(Nx,dx)
# Lmtx = sp.lil_matrix(Lmtx)
# print(Lmtx.shape)
# fig = plt.figure()
# ax = plt.axes()
# ax.pcolor(Lmtx.astype(np.ndarray))

plt.show()
