# Top-level script for calling simulateQuadrotorDynamics
from constantsScript import constants
from quadParamsScript import quadParams
from simulateQuadrotorDynamicsHF import SimulateQuadrotorDynamicsHF
from visualizeQuad import visualizeQuad

import numpy as np
import matplotlib.pyplot as plt

Q = np.eye(12)
R = np.eye(4)
quad_dynamics = SimulateQuadrotorDynamicsHF(Q, R)

actions = (590/200)*np.ones(4)  # Voltages to each motor 
disturbances = np.zeros(3)      # Disturbance forces acting on the body, in Newtons, expressed in I

_ = quad_dynamics.reset()
for _ in range(20000):
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
