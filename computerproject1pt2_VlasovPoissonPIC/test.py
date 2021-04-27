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
x_grid = np.linspace(x_min,x_max,Nx,dtype=float)
dx = (x_max - x_min)/float(Nx)

""" Test Findj() """
Ntests_Findj = 10
x_i_test = (x_max - x_min)*np.random.rand(Ntests_Findj) + x_min
low = 0
high = np.size(x_grid) - 1 # looking for index
guess = int(np.floor((low + high)/2))
# print("x_j[j] is %f" %x_grid[guess])
# print("x_j[j+1] is %f" %x_grid[guess+1])
for i in np.arange(Ntests_Findj):
    print("Test %i beginning" %(i+1))
    print("x_i is %f" %x_i_test[i])
    # print(x_i_test[i] < x_grid[guess])
    # print(x_i_test[i] < x_grid[guess+1])
    jfound_test = pmod.Findj(x_grid,x_i_test[i],guess,low,high)

""" Trying to observe Lmtx structure """
# Lmtx is exactly singular on first iteration 4 some reason
# Lmtx = pmod.LaplacianStencil(Nx,dx)
# Lmtx = sp.lil_matrix(Lmtx)
# print(Lmtx.shape)
# fig = plt.figure()
# ax = plt.axes()
# ax.pcolor(Lmtx.astype(np.ndarray))
# plt.show()
