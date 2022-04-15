# Top-level script for calling simulateQuadrotorDynamics
from constantsScript import constants
from quadParamsScript import quadParams
from simulateQuadrotorDynamics import simulateQuadrotorDynamics
from simulateQuadrotorDynamicsHF import simulateQuadrotorDynamicsHF
from visualizeQuad import visualizeQuad

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine


# Total simulation time, in seconds
Tsim = 5

# Update interval, in seconds.  This value should be small relative to the
# shortest time constant of your system.
delt = 0.005

# Time vector, in seconds 
N = int( np.floor(Tsim/delt) )

S = {}
S['tVec'] = np.arange(0,N).reshape(N,1)*delt

# Matrix of disturbance forces acting on the body, in Newtons, expressed in I
S['distMat'] = np.zeros((N,3))


omega = 2400
# Rotor speeds at each time, in rad/s
S['eaMat'] = (omega/200)*np.ones((N,4))

# Rotor speeds at each time, in rad/s
S['omegaMat'] = omega*np.ones((N,4))

S['state0'] = {} 

# Initial position in m
S['state0']['r'] = np.array([0, 0, 0])

# Initial attitude expressed as Euler angles, in radians
S['state0']['e'] = np.array([0, 0, 0])

# Initial velocity of body with respect to I, expressed in I, in m/s
S['state0']['v'] = np.array([0, 0, 0])

# Initial angular rate of body with respect to I, expressed in B, in rad/s
S['state0']['omegaB'] = np.array([0, 0, 0])

# Oversampling factor
S['oversampFact'] = 10

# Quadrotor parameters and constants
S['P'] = {}
S['P']['quadParams'] = quadParams
S['P']['constants']  = constants

P_simple = simulateQuadrotorDynamics(S)
P_hf     = simulateQuadrotorDynamicsHF(S)

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
plt.plot( P_simple['tVec'], P_simple['state']['rMat'][:,2], label='rudimentary')
plt.plot( P_hf['tVec'], P_hf['state']['rMat'][:,2], label='high fidelity')
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Z (m)')
plt.title('Vertical position of CM')
plt.legend()

plt.subplot(1,2,2)
plt.plot( P_simple['tVec'], P_simple['state']['vMat'][:,2], label='rudimentary')
plt.plot( P_hf['tVec'], P_hf['state']['vMat'][:,2], label='high fidelity')
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Z (m/s)')
plt.legend()
plt.title('Vertical velocity of CM')

plt.savefig('../doc/figs/comparation_3.png')
plt.show()
