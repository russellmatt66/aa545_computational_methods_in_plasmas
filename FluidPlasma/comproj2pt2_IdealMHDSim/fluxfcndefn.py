"""
Matt Russell
University of Washington Aero & Astro
Computer Project 2.2
5/19/21
File containing the definitions of the functions relating Q, the vector of
conserved quantities, to F, G, and H, the vectors containing the directed fluxes.
"""
import numpy as np
import sys

mu0 = 1.0 # permeability of free space
gamma = 5.0/3.0 # polytropic index

def FofQ(F,Q):
    """
    F - 8x1 ndarray
        Terms represent the flux of various conserved quantities through the
        x spatial dimension. Form is determined from Conservative formulation
        of Ideal MHD equations.
    Q - 8x1 ndarray
        Contains the conserved fluid variables:
        Q[0] = \rho, MHD fluid density
        Q[1] = \rho u, MHD fluid x-momentum
        Q[2] = \rho v, MHD fluid y-momentum
        Q[3] = \rho w, MHD fluid z-momentum
        Q[4] = B_{x}, Magnetic field x-component
        Q[5] = B_{y}, Magnetic field y-component
        Q[6] = B_{z}, Magnetic field z-component
        Q[7] = e, MHD fluid energy
    """
    if(F.shape[0]!= 8):
        print("F, the x-directed flux vector, is shaped wrong")
        AnomalyHandle()
    if(Q.shape[0]!= 8):
        print("Q, the vector of conserved Ideal MHD variables, is shaped wrong")
        AnomalyHandle()
    Bsq = Q[4]**2 + Q[5]**2 + Q[6]**2
    p = (gamma - 1.0)*(Q[7] - 0.5*(Q[1]**2)/(Q[0]) - Bsq/(2.0*mu0)) # pressure
    B_dotprod_uflow = (1.0/Q[0])*(Q[1]*Q[4] + Q[2]*Q[5] + Q[3]*Q[6])
    F[0] = Q[1]
    F[1] = (Q[1]**2)/Q[0] - (Q[4]**2)/mu0 + p + Bsq/(2.0*mu0)
    F[2] = (Q[1]*Q[2])/Q[0] - (Q[4]*Q[5])/mu0
    F[3] = (Q[1]*Q[3])/Q[0] - (Q[4]*Q[6])/mu0
    F[4] = 0.0
    F[5] = (Q[1]/Q[0])*Q[5] - (Q[2]/Q[0])*Q[4]
    F[6] = (Q[1]/Q[0])*Q[6] - (Q[3]/Q[0])*Q[4]
    F[7] = (Q[7] + p + Bsq/(2.0*mu0))*(Q[1]/Q[0]) - (Q[4]/mu0)*B_dotprod_uflow

def GofQ(G,Q):
    """
    G - 8x1 ndarray
        Terms represent the flux of various conserved quantities through the
        y spatial dimension. Form is determined from Conservative formulation
        of Ideal MHD equations.
    Q - 8x1 ndarray
        Contains the conserved fluid variables:
        Q[0] = \rho, MHD fluid density
        Q[1] = \rho u, MHD fluid x-momentum
        Q[2] = \rho v, MHD fluid y-momentum
        Q[3] = \rho w, MHD fluid z-momentum
        Q[4] = B_{x}, Magnetic field x-component
        Q[5] = B_{y}, Magnetic field y-component
        Q[6] = B_{z}, Magnetic field z-component
        Q[7] = e, MHD fluid energy
    """
    if(G.shape[0]!= 8):
        print("F, the x-directed flux vector, is shaped wrong")
        AnomalyHandle()
    if(Q.shape[0]!= 8):
        print("Q, the vector of conserved Ideal MHD variables, is shaped wrong")
        AnomalyHandle()
    Bsq = Q[4]**2 + Q[5]**2 + Q[6]**2
    p = (gamma - 1.0)*(Q[7] - 0.5*(Q[1]**2)/(Q[0]) - Bsq/(2.0*mu0)) # pressure
    B_dotprod_uflow = (1.0/Q[0])*(Q[1]*Q[4] + Q[2]*Q[5] + Q[3]*Q[6])
    G[0] = Q[2]
    G[1] = (Q[1]*Q[2])/Q[0] - (Q[4]*Q[5])/mu0
    G[2] = (Q[2]**2)/Q[0] - (Q[5]**2)/mu0 + p + Bsq/(2.0*mu0)
    G[3] = (Q[2]*Q[3])/Q[0] - (Q[5]*Q[6])/mu0
    G[4] = -(Q[1]/Q[0])*Q[5] + (Q[2]/Q[0])*Q[4]
    G[5] = 0.0
    G[6] = (Q[2]/Q[0])*Q[6] - (Q[3]/Q[0])*Q[5]
    G[7] = (Q[7] + p + Bsq/(2.0*mu0))*(Q[2]/Q[0]) - (Q[5]/mu0)*B_dotprod_uflow

def HofQ(G,Q):
    """
    H - 8x1 ndarray
        Terms represent the flux of various conserved quantities through the
        z spatial dimension. Form is determined from Conservative formulation
        of Ideal MHD equations.
    Q - 8x1 ndarray
        Contains the conserved fluid variables:
        Q[0] = \rho, MHD fluid density
        Q[1] = \rho u, MHD fluid x-momentum
        Q[2] = \rho v, MHD fluid y-momentum
        Q[3] = \rho w, MHD fluid z-momentum
        Q[4] = B_{x}, Magnetic field x-component
        Q[5] = B_{y}, Magnetic field y-component
        Q[6] = B_{z}, Magnetic field z-component
        Q[7] = e, MHD fluid energy
    """
    if(H.shape[0]!= 8):
        print("F, the x-directed flux vector, is shaped wrong")
        AnomalyHandle()
    if(Q.shape[0]!= 8):
        print("Q, the vector of conserved Ideal MHD variables, is shaped wrong")
        AnomalyHandle()
    Bsq = Q[4]**2 + Q[5]**2 + Q[6]**2
    p = (gamma - 1.0)*(Q[7] - 0.5*(Q[1]**2)/(Q[0]) - Bsq/(2.0*mu0)) # pressure
    B_dotprod_uflow = (1.0/Q[0])*(Q[1]*Q[4] + Q[2]*Q[5] + Q[3]*Q[6])
    H[0] = Q[3]
    H[1] = (Q[1]*Q[3])/Q[0] - (Q[4]*Q[6])/mu0
    H[2] = (Q[2]*Q[3])/Q[0] - (Q[5]*Q[6])/mu0
    H[3] = (Q[3]**2)/Q[0] - (Q[6]**2)/mu0 + p + Bsq/(2.0*mu0)
    H[4] = (Q[3]/Q[0])*Q[4] - (Q[1]/Q[0])*Q[6]
    H[5] = (Q[3]/Q[0])*Q[5] - (Q[2]/Q[0])*Q[6]
    H[6] = 0.0
    H[7] = (Q[7] + p + Bsq/(2.0*mu0))*(Q[3]/Q[0]) - (Q[6]/mu0)*B_dotprod_uflow

def AnomalyHandle():
    print("Please rerun the program")
    time.sleep(pause)
    sys.exit("Exiting ...")
