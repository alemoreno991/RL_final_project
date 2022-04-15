# Top-level script for calling simulateQuadrotorDynamics
from constantsScript import constants
from quadParamsScript import quadParams
from simulateQuadrotorDynamicsHF import SimulateQuadrotorDynamicsHF
from visualizeQuad import visualizeQuad

import numpy as np
import matplotlib.pyplot as plt


# Total simulation time, in seconds
Tsim = 5

# Update interval, in seconds. (This value should be small relative to the
# shortest time constant of your system)
delt = 0.005

# Number of time steps to finish the simulation 
N = int( np.floor(Tsim/delt) )

S = {}
S['tVec'] = np.arange(0,N).reshape(N,1)*delt

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


Q = np.eye(12)
R = np.eye(4)
quad_dynamics = SimulateQuadrotorDynamicsHF(S, Q, R)

actions = (590/200)*np.ones(4)  # Voltages to each motor 
disturbances = np.zeros(3)      # Disturbance forces acting on the body, in Newtons, expressed in I

while True:
    s, r, done = quad_dynamics.step(actions, disturbances)
    if done:
        break

Q = quad_dynamics.getResult()

S2 = {}
S2['tVec'] = Q['tVec']
S2['rMat'] = Q['state']['rMat']
S2['eMat'] = Q['state']['eMat']
S2['plotFrequency'] = 20
S2['makeGifFlag'] = False
S2['gifFileName'] = 'testGif.gif'
S2['bounds'] = 1*np.array( [-5, 5, -5, 5, -1, 1] )

visualizeQuad( S2 )

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
plt.plot( Q['tVec'], Q['state']['rMat'][:,2], label='high fidelity')
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Z (m)')
plt.title('Vertical position of CM')
plt.legend()

plt.subplot(1,2,2)
plt.plot( Q['tVec'], Q['state']['vMat'][:,2], label='high fidelity')
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Z (m/s)')
plt.legend()
plt.title('Vertical velocity of CM')

plt.show()
