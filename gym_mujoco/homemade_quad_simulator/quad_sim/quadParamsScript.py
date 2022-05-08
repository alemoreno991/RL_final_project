import numpy as np

# Loads quadrotor parameters into the structure quadParams
quadParams = {}

# kF(i) is the rotor thrust constant for the ith rotor, in N/(rad/s)^2
quadParams['kF'] = 6.11e-8*(0.104719755)**-2 * np.ones(4)

# kN(i) is the rotor counter-torque constant for the ith rotor, in N-m/(rad/s)^2
quadParams['kN'] = 1.5e-9*(0.104719755)**-2 * np.ones(4)

# omegaRdir(i) indicates the ith rotor spin direction: 1 for a rotor angular
# rate vector aligned with the body z-axis, -1 for the opposite direction.
# Note that the torque vector caused by the ith rotor's twisting against the
# air is *opposite* the spin direction: Ni = [0;0;-kN*omegaRdir(i)*omegaVec(i)^2], 
# where omegaVec is the 4x1 vector of rotor angular rates.
quadParams['omegaRdir'] = np.array( [1, -1,  1, -1] )

# rotor_loc(:,i) holds the 3x1 coordinates of the ith rotor in the body frame,
# in meters
quadParams['rotor_loc']  =  0.21 * np.array( [[ 1,  1, -1, -1], 
                                              [-1,  1,  1, -1],
                                              [ 0,  0,  0,  0]] )

# Mass of the quad, in kg
quadParams['m'] = 0.78

# The quad's moment of inertia, expressed in the body frame, in kg-m^2
quadParams['Jq'] = 1e-9*np.diag( [1756500, 3572300, 4713400] )

# The circular-disk-equivalent area of the quad's body, in m^2
quadParams['Ad'] = 0.01

# The quad's coefficient of drag (unitless)
quadParams['Cd'] = 0.3

# taum(i) is the time constant of the ith rotor, in seconds. This governs how
# quickly the rotor responds to input voltage.
quadParams['taum'] = (1/20)*np.ones(4)

# cm(i) is the factor used to convert motor voltage to motor angular rate
# in steady state for the ith motor, with units of rad/sec/volt
quadParams['cm'] = 200*np.ones(4)

# Maximum voltage that can be applied to any motor, in volts
quadParams['eamax'] = 12

# r_rotor(i) is the radius of the ith rotor, in meters
quadParams['r_rotor'] = 0.06*np.ones(4)

#-----------------------------------------------------------------------------
# The parameters below are for a more detailed model of the quad that includes
# a detailed aerodynamics model and handling of blade flexure.  You do not
# need to use these parameters for the Aerial Robotics course unless you'd
# like to make an especially high-fidelity simulator.
#
# The more detailed model is described in 
#
# G. M. Hoffmann, H. Huang, S. L. Waslander, and C. J. Tomlin, "Quadrotor
# helicopter flight dynamics and control: Theory and experiment," in Proc.of
# the AIAA Guidance, Navigation, and Control Conference, vol.  2, p. 4, 2007.
# Jm(i) is the moment of inertia about the main axis of rotation for the ith
# rotor, in kg-m^2.
quadParams['Jm'] = 0.0124*np.ones(4)

# Lock number, nondimensional
quadParams['gamma'] = 1e-3*quadParams['r_rotor']**4/quadParams['Jm']

# Offset of "hinge" from rotor hub, as a fraction of full rotor length.  See
# reference [2] of Lab Assignment 1.
quadParams['efConst'] = 1e-3*np.ones(4)

# Ratio of kB (parameter modeling rotor stiffness) to the rotor's moment of
# inertia.  Units of (rad/s)^2
quadParams['kBIb'] = 1e-4*np.ones(4)

# Average induced velocity ratio, unitless. See reference [1].
quadParams['lambdaI'] = 1.3772e-4*np.ones(4)

# Average pitch angle of the blade, radians
quadParams['thetaAvg'] = np.pi/12*np.ones(4)

# Linearized lift coefficients of the body, expressed in the body frame.
# Units of N/(m/s).
quadParams['Clmat'] = np.array( [[0   ,    0, 0],
                                 [0   ,    0, 0], 
                                 [.001, .001, 0]] )

# Linearized drag coefficients of the body, expressed in the body frame.
# Units of N/(m/s).
quadParams['Cdmat'] = np.diag([0.007, 0.002, 0.01])

# Second-order drag coefficients of the body, expressed in the body frame.
# Units of N/(m/s)^2.
quadParams['Cd2mat'] = np.diag([0.0001, 0.0001, 0.0005])

# Linearized coefficients of body roll/pitch/yaw.  Units of N-m/(rad/s).
quadParams['Cpqr']=np.zeros((3,3))