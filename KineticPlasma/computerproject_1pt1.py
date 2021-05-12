"""
Matt Russell
AA545: Computational Plasma Physics
4/7/21
Script for the assignment:
"Computer Project 1.1:
Preliminary Code - Kinetic Modeling: Vlasov-Poisson PIC"
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

""" Function Declarations """
def firstorderfd_x(x_n,v,dt):
    return float(x_n + dt*v)

""" Declare Boundaries of the Phase-Space """
vx_lb = -5.0
vx_ub = 5.0

x_lb = -2.0*np.pi
x_ub = 2.0*np.pi

""" Obtain Number of Particles to simulate and check if the value is valid """
N = int(input('Number of Particles (must be 128, 512, or 2048): '))

if (N != 128 and N != 512 and N != 2048):
    print("N = %i is invalid" %N)
    sys.exit("exiting execution...")
else:
    print("N = %i is valid!\nContinuing program execution ..." %N)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Initialize State of the Distibution of Particles """
print("Beginning Initialization stage")
particle_positions = 2.0*np.pi*(2.0*np.random.rand(N,1) - 1.0)#Uniform over span
particle_velocities = np.empty(N)#Initialize as Maxwellian w/FWHM = 2
vx_fwhm = 2.0 #Full Width (at) Half Max
vx_sigma = vx_fwhm/(2.0*np.sqrt(2.0*np.log(2.0))) #standard deviation from fwhm
vx_mean = 0.0

for i_init in np.arange(N):
    particle_velocities[i_init] = np.random.normal(vx_mean,vx_sigma)

print("Initialization stage completed successfully, continuing flow ...")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Evolution of Particle Motion """
print("Beginning Particle Evolution stage")
# No Force or Collisions -> velocities remain constant
ti = 0.0
tf = 8.0*np.pi
num_tsteps = int(input('How many steps in time do you want to take (must be an integer)?: '))
dt = float((tf - ti)/num_tsteps)
print("The size of the time step is %f" %dt)

kineticenergy_fig = plt.figure()

if N == 512:
    positions512_fig, positions512_ax = plt.subplots(ncols=3)
if N == 128:
    trajectory128_fig = plt.figure()

while ti <= tf:
    print("Beginning evolution for t = %f" %ti)
    for i_evol in np.arange(np.size(particle_positions)):
        x_ni = float(particle_positions[i_evol]) # x_ni is already a float: this
        # is done so that it doesn't change after timestep and ruin Periodic B.Cs.
        vx_ni = particle_velocities[i_evol]
        particle_positions[i_evol]=firstorderfd_x(x_ni,vx_ni,dt)
        # Periodic B.Cs
        if x_ni < x_ub and particle_positions[i_evol] > x_ub:
            print("Asserting Periodic B.C. on the Right")
            particle_positions[i_evol] = x_lb + particle_positions[i_evol] - x_ub
        if x_ni > x_lb and particle_positions[i_evol] < x_lb:
            print("Asserting Periodic B.C. on the Left")
            particle_positions[i_evol] = x_ub + particle_positions[i_evol] - x_lb
    # Grab necessary data for the report
    if N == 128 and ti < 2.0*np.pi:
        plt.figure(trajectory128_fig.number)
        plt.scatter(particle_positions,particle_velocities)
    if N == 512 and ti == 0.0:
        plt.figure(positions512_fig.number)
        positions512_ax[0].scatter(particle_positions,particle_velocities)
    if N == 512 and ti < 2.0*np.pi and ti+dt > 2.0*np.pi:
        plt.figure(positions512_fig.number)
        positions512_ax[1].scatter(particle_positions,particle_velocities)
    if N == 512 and ti <= 8.0*np.pi and ti+dt > 8.0*np.pi:
        plt.figure(positions512_fig.number)
        positions512_ax[2].scatter(particle_positions,particle_velocities)
    # Plot history of kinetic energy for system
    plt.figure(kineticenergy_fig.number)
    kineticenergy_snapshot = 0.5*np.dot(particle_velocities,particle_velocities)
    plt.scatter(ti,kineticenergy_snapshot)
    ti += dt

print("Particle Evolution stage completed successfully, continuing flow ...")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Diagnostics """
hist_fig = plt.figure()
bin_size = 0.25
num_bins = int((vx_ub - vx_lb)/bin_size)
plt.hist(particle_velocities,num_bins)
plt.xlim((vx_lb,vx_ub))
plt.title('Velocity distribution for N = %i particles' %N)
plt.xlabel('$v_{x}$')
plt.ylabel('Number')

plt.figure(kineticenergy_fig.number)
plt.title('Kinetic Energy as a function of time for N = %i particles and dt=%4.3f' %(N,dt))
plt.xlabel('time')
plt.ylabel('KE')
plt.xlim((0.0,tf))
plt.ylim((0.0,np.dot(particle_velocities,particle_velocities)))

if N == 128:
    plt.figure(trajectory128_fig.number)
    plt.title('Trajectories of N = %i particles in phase space for t $\in$ [0,$2\pi$] with dt = %4.3f' %(N,dt))
    plt.xlabel('x')
    plt.ylabel('vx')
    plt.xlim((x_lb,x_ub))
    plt.ylim((vx_lb,vx_ub))

if N == 512:
    plt.figure(positions512_fig.number)
    positions512_fig.suptitle('State of N = %i particles at different moments with dt = %4.3f' %(N,dt))
    positions512_ax[0].title.set_text('t = 0')
    positions512_ax[0].set_xlim([x_lb,x_ub])
    positions512_ax[0].set_ylim([vx_lb,vx_ub])
    positions512_ax[1].title.set_text('t = $2\pi$')
    positions512_ax[1].set_xlim([x_lb,x_ub])
    positions512_ax[1].set_ylim([vx_lb,vx_ub])
    positions512_ax[2].title.set_text('t = $8\pi$')
    positions512_ax[2].set_xlim([x_lb,x_ub])
    positions512_ax[2].set_ylim([vx_lb,vx_ub])

plt.show()
