import numpy as np
from scipy.integrate import ode

from quadOdeFunction import quadOdeFunction
from euler2dcm import euler2dcm
from dcm2euler import dcm2euler
from auxiliar import stateVtrFormat2ToFormat1, stateVtrFormat1ToFormat2, stateVtrFormat2, parseXFormat2

def simulateQuadrotorDynamics(S):
# simulateQuadrotorDynamics : Simulates the dynamics of a quadrotor aircraft.
#
#
# INPUTS
#
# S ---------- Structure with the following elements:
#
#          tVec = Nx1 vector of uniformly-sampled time offsets from the
#                 initial time, in seconds, with tVec(1) = 0.
#
#  oversampFact = Oversampling factor. Let dtIn = tVec(2) - tVec(1). Then the
#                 output sample interval will be dtOut =
#                 dtIn/oversampFact. Must satisfy oversampFact >= 1.   
#
#  
#      omegaMat = (N-1)x4 matrix of rotor speed inputs.  omegaMat(k,j) is the
#                 constant (zero-order-hold) rotor speed setpoint for the jth rotor
#                 over the interval from tVec(k) to tVec(k+1).
#
#        state0 = State of the quad at tVec(1) = 0, expressed as a structure
#                 with the following elements:
#                   
#                   r = 3x1 position in the world frame, in meters
# 
#                   e = 3x1 vector of Euler angles, in radians, indicating the
#                       attitude
#
#                   v = 3x1 velocity with respect to the world frame and
#                       expressed in the world frame, in meters per second.
#                 
#              omegaB = 3x1 angular rate vector expressed in the body frame,
#                       in radians per second.
#
#       distMat = (N-1)x3 matrix of disturbance forces acting on the quad's
#                 center of mass, expressed in Newtons in the world frame.
#                 distMat(k,:)' is the constant (zero-order-hold) 3x1
#                 disturbance vector acting on the quad from tVec(k) to
#                 tVec(k+1).
#
#    quadParams = Structure containing all relevant parameters for the
#                 quad, as defined in quadParamsScript.m 
#
#     constants = Structure containing constants used in simulation and
#                 control, as defined in constantsScript.m 
#
#
# OUTPUTS
#
# P ---------- Structure with the following elements:
#
#         tVec = Mx1 vector of output sample time points, in seconds, where
#                 P.tVec(1) = S.tVec(1), P.tVec(M) = S.tVec(N), and M =
#                 (N-1)*oversampFact + 1.
#                  
#  
#         state = State of the quad at times in tVec, expressed as a structure
#                 with the following elements:
#                   
#                rMat = Mx3 matrix composed such that rMat(k,:)' is the 3x1
#                       position at tVec(k) in the world frame, in meters.
# 
#                eMat = Mx3 matrix composed such that eMat(k,:)' is the 3x1
#                       vector of Euler angles at tVec(k), in radians,
#                       indicating the attitude.
#
#                vMat = Mx3 matrix composed such that vMat(k,:)' is the 3x1
#                       velocity at tVec(k) with respect to the world frame
#                       and expressed in the world frame, in meters per
#                       second.
#                 
#           omegaBMat = Mx3 matrix composed such that omegaBMat(k,:)' is the
#                       3x1 angular rate vector expressed in the body frame in
#                       radians, that applies at tVec(k).
#
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  Alejandro Moreno (ale.moreno991@utexas.edu)
#+==============================================================================+
    # Set config params
    params = {}
    
    params['g']  = S['P']['constants']['g']
    params['m']  = S['P']['quadParams']['m']
    params['J']  = S['P']['quadParams']['Jq']
    params['kF'] = S['P']['quadParams']['kF']
    params['kN'] = S['P']['quadParams']['kN']
    params['rotor_loc'] = S['P']['quadParams']['rotor_loc']

    # Initial value of vector X
    X0 = stateVtrFormat2( S['state0']['r'], \
                          S['state0']['v'], \
                          S['state0']['e'], \
                          S['state0']['omegaB'] )

    # Vector of simulation segment start times
    tVecIn = S['tVec']
    dtIn   = tVecIn[2] - tVecIn[1]
    N      = len(tVecIn)

    # Oversampling causes the ODE solver to produce output at a finer time
    # resolution than dtIn. oversampFact is the oversampling factor.
    oversampFact = S['oversampFact']
    dtOut = dtIn/oversampFact


    # Set initial state 
    Xk = stateVtrFormat2ToFormat1(X0)

    # output structure
    P = {}
    P['tVec'] = tVecIn[0]
    P['state'] = {}
    P['state']['rMat'] = S['state0']['r']
    P['state']['vMat'] = S['state0']['v']
    P['state']['eMat'] = S['state0']['e']
    P['state']['omegaBMat'] = S['state0']['omegaB']

    # In this example, we run ode45 for N-1 segments of dtIn seconds each.  Note
    # that the value of parameters.D gets updated at the end of each iteration.
    # This is just as an example of something that changes from one iteration to
    # the next.
    for k in np.arange(0, N-1):  
        # Build the time vector for kth segment.  We oversample by a factor
        # oversampFact relative to the coarse timing of each segment because we
        # may be interested in dynamical behavior that is short compared to the
        # segment length.
        ti = tVecIn[k]
        tf = tVecIn[k+1]
        dt = dtOut

        # Run ODE solver for segment
        solver = ode(quadOdeFunction).set_integrator( 'dopri5' )
        solver.set_initial_value(Xk, ti).set_f_params(S['omegaMat'][k,:], S['distMat'][k,:], params)
        while solver.successful() and solver.t < tf-dt/3: # "-dt/3" is just to make sure that no numeric errors alter the intention
            t = solver.t+dt 
            Xk = solver.integrate(t)

            rI, vI, e, omegaB = parseXFormat2( stateVtrFormat1ToFormat2(Xk) )
            # Add the data from the kth segment to your storage vector
            P['tVec']              = np.append( P['tVec']             , t  )
            P['state']['rMat']     = np.vstack( (P['state']['rMat']    , rI ))
            P['state']['eMat']     = np.vstack( (P['state']['eMat']    , e  ))
            P['state']['vMat']     = np.vstack( (P['state']['vMat']    , vI ))                        
            P['state']['omegaBMat']= np.vstack( (P['state']['omegaBMat'], omegaB ))

    return P