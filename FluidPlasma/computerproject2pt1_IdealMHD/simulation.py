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

# print("Please indicate if you would like to add diffusion to stabilize the FTCS method, 1 = 'Yes', 0 = 'No':")
# DiffusionFlag = int(input(''))
# if(DiffusionFlag != 0 and DiffusionFlag != 1):
#     print("ERROR: Program cannot parse input of %i" %DiffusionFlag)
#     print("Only values of 1 or 0 are acceptable input for this step")
#     AnomalyHandle()

""" Grid Generation """
x_grid = np.linspace(0.0,xf,num=Nx,retstep=True) # ([grid], dx)
dx = x_grid[1] # dx
x_grid = x_grid[0] # [grid] = {x_{0},x_{1},x_{2},...,x_{199},x_{200}}

""" Finalization of Initialization Process """
NumberofSchemes = 4 # {FTCS,Lax,Lax-Wendroff,Upwind}
u_collection = np.zeros((x_grid.shape[0],NumberofSchemes)) # Ties script together
for nsidx in np.arange(NumberofSchemes): # I.Cs
    u_collection[10:20,nsidx] = 1.0

# u_ftcs = np.zeros(x_grid.shape[0])
# u_lax = np.zeros(x_grid.shape[0])
# u_laxwendroff = np.zeros(x_grid.shape[0])
# u_upwind = np.zeros(x_grid.shape[0])
# u_ftcs[10:20] = 1.0 # I.Cs
# u_upwind[10:20] = 1.0

""" Numerical Stage """
SchemeNumber = 0 # see description of 'flag' in mod.Advance()
limit = 1000
eps = 1.0e-2
while(SchemeNumber < NumberofSchemes):
    print("Simulating scheme %i" %SchemeNumber)
    stepcount = 1
    while(u_collection[Nx-2,SchemeNumber] < eps): # Need to stop each process once it hits boundary
        if(stepcount > limit):
            print("Limiting number of steps taken, breaking control flow")
            break
        print("Taking step %i" %stepcount)
        u_collection[:,SchemeNumber] = mod.Advance(u_collection[:,SchemeNumber],a,dx,dt,SchemeNumber)
        stepcount += 1
    SchemeNumber += 1

FinalStateFig, FinalStateAx = plt.subplots(4,1)
for axidx in np.arange(NumberofSchemes):
    FinalStateAx[axidx].plot(x_grid,u_collection[:,axidx]) # bc.base(Number,BaseFrom,BaseTo)

plt.show()
