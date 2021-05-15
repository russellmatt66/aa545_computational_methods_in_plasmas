"""
Matt Russell
AA545: Computational Methods for Plasmas
5/14/21
Linear Wave Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import baseconvert as bc
import module as mod

""" Basic Parameters """
a = 1.0 # wave speed
dt = 0.5 # size of time-step
Nx = 201 # Number of grid points
xf = 200.0

""" Grid Generation """
x_grid = np.linspace(0.0,xf,num=Nx,retstep=True) # ([grid], dx)
dx = x_grid[1] # dx
x_grid = x_grid[0] # [grid] = {x_{0},x_{1},x_{2},...,x_{199},x_{200}}

""" Assignment based on exploring the output of several different numerical schemes """
Schemes = ['FTCS','Lax','Lax-Wendroff','Upwind']
NumberofSchemes = len(Schemes)
FinalStateFig, FinalStateAx = plt.subplots(NumberofSchemes,1)
u_collection = np.zeros((x_grid.shape[0],NumberofSchemes)) # Ties script together
for nsidx in np.arange(NumberofSchemes): # I.Cs
    u_collection[10:20,nsidx] = 1.0
    FinalStateAx[nsidx].plot(x_grid,u_collection[:,nsidx],'--',label='IC')
    # plt.legend()

""" Numerical Stage """
SchemeNumber = 0 # see description of 'flag' in mod.Advance()
limit = 1000
eps = 1.0e-2
while(SchemeNumber < NumberofSchemes):
    print("Simulating scheme %i" %SchemeNumber)
    stepcount = 0
    while(u_collection[Nx-2,SchemeNumber] < eps): # Need to stop each process once it hits boundary
        if(stepcount > limit):
            print("Limiting number of steps taken, breaking control flow")
            break
        u_collection[:,SchemeNumber] = mod.Advance(u_collection[:,SchemeNumber],a,dx,dt,SchemeNumber)
        stepcount += 1
    print("Scheme %i halted in %i steps" %(SchemeNumber,stepcount))
    SchemeNumber += 1

plt.figure(FinalStateFig.number)
for axidx in np.arange(NumberofSchemes):
    FinalStateAx[axidx].plot(x_grid,u_collection[:,axidx],'-',label=Schemes[axidx])

FinalStateFig.suptitle('Numerical Simulation of Square Wave Advection')
plt.show()
