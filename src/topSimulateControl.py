# Top-level script for calling simulateQuadrotorDynamics
from auxiliar import e1
from constantsScript import constants
from quadParamsScript import quadParams
from euler2dcm import euler2dcm
from simulateQuadrotorControl import simulateQuadrotorControl
from visualizeQuad import visualizeQuad

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

# Total simulation time, in seconds
Tsim = 10

# Update interval, in seconds.  This value should be small relative to the
# shortest time constant of your system.
delt = 0.005

# Time vector, in seconds 
N = int( np.floor(Tsim/delt) )
t = np.arange(0,N)*delt


w = (2*np.pi)/Tsim 
radius = 2.0
pitch = -np.arctan( radius*w**2/constants['g'] )

R = {}
R['tVec'] = t
R['rIstark'] =  radius      * np.stack((  np.cos(w*t), np.sin(w*t), np.zeros_like(t) )).T
R['vIstark'] =  radius*w    * np.stack(( -np.sin(w*t), np.cos(w*t), np.zeros_like(t) )).T
R['aIstark'] = -radius*w**2 * np.stack((  np.cos(w*t), np.sin(w*t), np.zeros_like(t) )).T
R['xIstark'] = -np.stack((  np.cos(w*t), np.sin(w*t), np.zeros_like(t) )).T  


S = {}
S['oversampFact'] = 10

# S['state0'] = {}
# S['state0']['r'] = np.array( [radius, 0, 0] )
# S['state0']['e'] = np.array( [0, pitch, 0] )
# S['state0']['v'] = np.array( [0, radius*w, 0] )
# S['state0']['omegaB'] = euler2dcm(S['state0']['e'])@(w*e3)

S['state0'] = {}
S['state0']['r'] = np.array( [radius, 0, 0] )
S['state0']['e'] = np.array( [0, 0, np.rad2deg(90)] )
S['state0']['v'] = np.array( [0, 0, 0] )
S['state0']['omegaB'] = np.array( [0, 0, 0] )

S['distMat']  = np.zeros( (len(R['tVec']), 3) )

# Quadrotor parameters and constants
P = {}
P['quadParams'] = quadParams
P['constants']  = constants

Q = simulateQuadrotorControl(R,S,P)

S2 = {}
S2['tVec'] = Q['tVec']
S2['rMat'] = Q['state']['rMat']
S2['eMat'] = Q['state']['eMat']
S2['plotFrequency'] = 20
S2['makeGifFlag'] = False
S2['gifFileName'] = 'testGif.gif'
S2['bounds'] = 1*np.array( [-5, 5, -5, 5, -1, 1] )

visualizeQuad( S2 )

P=Q
plt.figure(figsize=(20,15))
plt.plot( P['tVec'], P['state']['rMat'][:,2], color='black', linewidth=2, label="P" )
#plt.plot( R['tVec'], R['rIstark'][:,2], linewidth=2, label="Pexp" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Vertical (m)')
plt.title('Vertical position of CM')
plt.legend()
plt.show()

nth=50
for k, e in enumerate(S2['eMat']):
    if k == 0:
        bodyX = euler2dcm(e).T @ e1
    else:
        bodyX = np.vstack( (bodyX, euler2dcm(e).T @e1) )


plt.figure(figsize=(20,15))
plt.plot( P['state']['rMat'][:,0], P['state']['rMat'][:,1], linewidth=2, label="P"); 
#plt.plot( R['rIstark'][:,0], R['rIstark'][:,1], linewidth=2, label="Pexp" )
#plt.quiver( R['rIstark'][0::nth,0], R['rIstark'][0::nth,1], R['xIstark'][0::nth,0], R['xIstark'][0::nth,1], color='black' )
plt.quiver( P['state']['rMat'][0::nth,0], P['state']['rMat'][0::nth,1], bodyX[0::nth,0], bodyX[0::nth,1], color='green' )
plt.grid( True )
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Horizontal position of CM')
plt.legend()
plt.show()

def isin_with_tolerance(vtr, test_vtr):
    idxs = []
    i = 0
    for k, a in enumerate(vtr):
        if np.abs(a - test_vtr[i]) < 0.0001:
            idxs.append(k)
            i += 1

    return idxs

idxs = isin_with_tolerance(P['tVec'], R['tVec'])
error_h = np.linalg.norm(P['state']['rMat'][idxs][:,0:2] - R['rIstark'][:,0:2], axis=1)
error_z = np.abs(P['state']['rMat'][idxs][:,2] - R['rIstark'][:,2])

plt.figure(figsize=(20,15))
plt.subplot(2,1,1)
plt.plot( P['tVec'][idxs], 1000*error_z, linewidth=2)
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Error norm (mm)')
plt.title('Vertical error of CM')
plt.legend()

plt.subplot(2,1,2)
plt.plot( P['tVec'][idxs], 1000*error_h, linewidth=2)
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Error norm (mm)')
plt.title('Horizontal error of CM')
plt.legend()
plt.show()

plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot( P['tVec'], P['state']['vMat'][:,2], linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Vertical (m)')
plt.title('Vertical velocity of CM')
plt.legend()

plt.subplot(3,1,2)
plt.plot( P['tVec'], P['state']['vMat'][:,1], linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Y (m/s)')
plt.title('Y velocity of CM')
plt.legend()

plt.subplot(3,1,3)
plt.plot( P['tVec'], P['state']['vMat'][:,0], linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('X (m/s)')
plt.title('X velocity of CM')
plt.legend()

plt.show()



plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot( P['tVec'], np.rad2deg(P['state']['eMat'][:,2]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Yaw (deg)')
plt.title('Yaw')
plt.legend()

plt.subplot(3,1,2)
plt.plot( P['tVec'], np.rad2deg(P['state']['eMat'][:,1]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Pitch (deg)')
plt.title('Pitch')
plt.legend()

plt.subplot(3,1,3)
plt.plot( P['tVec'], np.rad2deg(P['state']['eMat'][:,0]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Roll (deg)')
plt.title('Pitch')
plt.legend()

plt.show()


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot( P['tVec'], np.rad2deg(P['state']['omegaBMat'][:,2]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Yaw_dot (deg/s)')
plt.title('Yaw_dot')
plt.legend()

plt.subplot(3,1,2)
plt.plot( P['tVec'], np.rad2deg(P['state']['omegaBMat'][:,1]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Pitch_dot (deg/s)')
plt.title('Pitch_dot')
plt.legend()

plt.subplot(3,1,3)
plt.plot( P['tVec'], np.rad2deg(P['state']['omegaBMat'][:,0]), linewidth=2, label="P" )
plt.grid( True )
plt.xlabel('Time (sec)')
plt.ylabel('Roll_dot (rad/s)')
plt.title('Pitch_dot')
plt.legend()

plt.show()


