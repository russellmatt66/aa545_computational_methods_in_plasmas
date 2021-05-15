import numpy as np
import sys

def Advance(u,a,dx,dt,flag):
    """
    u - quantity being advected through spacetime
    flag - {0,1,2,3}
        0 = FTCS
        1 = Lax
        2 = Lax-Wendroff
        3 = Upwind Differencing
    """
    for j in np.arange(1,u.shape[0]-1):
        if flag == 0: # FTCS
            u[j] = u[j] - (a*dt/(2.0*dx))*(u[j+1] - u[j-1])
        if flag == 1: # Lax
            u[j] = u[j] - (a*dt/(2.0*dx))*(u[j+1] - u[j-1]) \
                + (1.0/2.0)*(u[j+1] - 2.0*u[j] + u[j-1])
        if flag == 2: # Lax-Wendroff
            u[j] = u[j] - ((a*dt)/(2.0*dx))*(u[j+1] - u[j-1]) \
                + ((a**2 * dt**2)/(2.0*dx**2))*(u[j+1] - 2.0*u[j] + u[j-1])
        if flag == 3: # Upwind Differencing
            if a >= 0.0:
                u[j] = u[j] - ((a*dt)/dx)*(u[j] - u[j-1])
            elif a < 0.0: # for completeness
                u[j] = u[j] - ((a*dt)/dx)*(u[j+1] - u[j])
    return u



def AnomalyHandle():
    print("Please rerun the program")
    sys.exit("Exiting ...")
