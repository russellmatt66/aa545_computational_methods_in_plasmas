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
N = 64
particlesPosition = np.zeros((N,1),dtype=float)
particlesVelocity = np.zeros((N,1),dtype=float)

# Normalization
eps_0 = 1.0 # vacuum permittivity
qm = -1.0 # charge to mass ratio of superparticle
omega_p = 1.0 # plasma frequency
tau_plasma = 2.0*np.pi/omega_p
q_sp = eps_0 * omega_p**2 * (L/float(N)) / qm# charge associated with a superparticle given normalization - population assumed uniform
m_sp = qm*q_sp # mass associated with a particular superparticle given normalization - population assumed uniform
T = 1.0 # [eV]
lambda_D = np.sqrt((eps_0 * T * L)/(N * q_sp **2))
v_th = omega_p*lambda_D # thermal velocity
print(v_th)
print(1.0e-32*v_th)

""" Test particle distribution Initialization for N = 64 """
# vprime = 0.01*v_th
# for pidx in np.arange(N):
#     particlesPosition[pidx] = x_min + float(pidx)*L/float(N-1)
#     particlesVelocity[pidx] = vprime*np.sin(2.0*np.pi/L * particlesPosition[pidx])
#
# PhaseSpaceFig = plt.figure()
# plt.figure(PhaseSpaceFig.number)
# plt.scatter(particlesPosition,particlesVelocity)
# plt.xlabel('x')
# plt.ylabel('v (normalized to $v_{th}$)')
# plt.xlim((x_min,x_max))
# plt.ylim((-2.0,2.0))
#
# plt.show()

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
# plt.show()
