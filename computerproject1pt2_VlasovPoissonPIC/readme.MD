Project: Electrostatic PIC Investigation of Langmuir Oscillations and Leapfrog
         Instability

Project Status: [INCOMPLETE]
  - Initialization
    - Mostly Written
      - Lacks possibility of velocity perturbation
    - Untested
  - Particle-Weighting
    - Written
    - Untested
  - Field-Solver
    - Written
    - Untested
  - Force-Weighting
    - Written
    - Untested
  - Particle Push
    - Unwritten

Project Structure:
  picmain_1pt2.py - Main simulation file that handles the flow of data and calling
                    of functions
  picmodule_1pt2.py - Module that contains definitions for the various functions
                      used in the simulation

Project Requirements:
  - Diagnostics [INCOMPLETE]
      - Time history of system kinetic energy
      - Time history of electric field energy
  - Langmuir Oscillation Frequency Measurement [INCOMPLETE]
  - Leap-Frog Instability [INCOMPLETE]
