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

#rho_j = pmod.RHO_j
#x_grid = pmod.X_GRID

#ChrgFig = plt.figure()
#ChrgAx = plt.axes()
#plt.xlim(pmod.X_MIN,pmod.X_MAX)
ChrgFig, ChrgAx = plt.subplots(1,2)

sc_rho = ChrgAx[0].scatter([],[])
ChrgAx[0].set_xlim(pmod.X_MIN,pmod.X_MAX)

sc_prtcl = ChrgAx[1].scatter([],[])
ChrgAx[1].set_xlim(pmod.X_MIN,pmod.X_MAX)

"""
def init():
    ChrgPlot.set_data([], [])
    ChrgAx.set_xlim(pmod.X_MIN,pmod.X_MAX)
    ChrgAx.set_ylim(-3.0,3.0)
    return ChrgPlot,
"""

def animate(N):
    # I need each frame to show the charge density and each particle for that particular N with NO superposition, i.e, for N = 20 I only want to see the data for 20 particles. I do not want data from N = 16, 17, etc. to also be on the figure. 
    x_i = np.zeros((N,1),dtype=float) # particle positions
    q_sp = (pmod.EPS_0 * pmod.L / N) * (1.0 / pmod.Q_OVER_M)
    # linear coefficients for evenly distributing particles
    a = (pmod.L - 2.0*pmod.DX)/(N-1)
    b = pmod.X_MIN + pmod.DX
    for pidx in np.arange(N):
        x_i[pidx] = a*pidx + b
    pmod.ParticleWeighting(1,x_i,q_sp) # updates rho_j
    sc_rho.set_offsets(np.c_[pmod.X_GRID,pmod.RHO_j])
    rhomax = np.amax(pmod.RHO_j)
    ChrgAx[0].set_ylim(-rhomax,rhomax)
    sc_prtcl.set_offsets(np.c_[x_i,np.ones(N)])
    ChrgAx[1].set_ylim(0.0,2.0)
    
PrtclWgtAnimation = FuncAnimation(fig=ChrgFig, func=animate, frames=Np_vec, interval=1000, repeat=False)

PrtclWgtAnimation.save('rho-ster_scan.gif', writer='imagemagick')
