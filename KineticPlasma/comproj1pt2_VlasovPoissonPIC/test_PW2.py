"""
Matt Russell
8/2/21
test_PW2.py 
"""
import numpy as np
import matplotlib.pyplot as plt
import picmodule_1pt2 as pmod

from matplotlib.animation import FuncAnimation

N_min = 16
N_max = 64
Np_vec = np.array([Nprtcl for Nprtcl in range(N_min,N_max+1)])

ChrgFig = plt.figure()
ChrgAx = plt.axes()
ChrgPlot, = ChrgAx.plot([], [], lw=3)

def init():
    ChrgPlot.set_data([], [])
    return ChrgPlot,

def animate(N):
    global rho_j, x_i, x_grid
    
    rho_j = pmod.ParticleWeighting(1,x_i,x_grid,Nx,dx,L,rho_j,q_sp)
    ChrgPlot.set_data(x_grid,rho_j)
    ChrgPlot.set_data(x_i,np.ones(len(x_i)))
    return ChrgPlot,

PrtclWgtAnimation = FuncAnimation(fig=ChrgFig, func=animate, frames=Np_vec, interval=250, blit=True, repeat=True)

PrtclWgtAnimation.save('rho-ster_scan.gif', writer='imagemagick',fps=4)
