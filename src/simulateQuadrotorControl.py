import numpy as np
from scipy.integrate import ode

from euler2dcm import euler2dcm
from dcm2euler import dcm2euler
from quadOdeFunctionHF import quadOdeFunctionHF
from trajectoryController import trajectoryController
from attitudeController import attitudeController
from voltageConverter import voltageConverter

def simulateQuadrotorControl(R,S,P):
# simulateQuadrotorControl : Simulates closed-loop control of a quadrotor
#                            aircraft.
#
#
# INPUTS
#
# R ---------- Structure with the following elements:
#
#          tVec = Nx1 vector of uniformly-sampled time offsets from the
#                 initial time, in seconds, with tVec(1) = 0.
#
#        rIstar = Nx3 matrix of desired CM positions in the I frame, in
#                 meters.  rIstar(k,:)' is the 3x1 position at time tk =
#                 tVec(k).
#
#        vIstar = Nx3 matrix of desired CM velocities with respect to the I
#                 frame and expressed in the I frame, in meters/sec.
#                 vIstar(k,:)' is the 3x1 velocity at time tk = tVec(k).
#
#        aIstar = Nx3 matrix of desired CM accelerations with respect to the I
#                 frame and expressed in the I frame, in meters/sec^2.
#                 aIstar(k,:)' is the 3x1 acceleration at time tk =
#                 tVec(k).
#
#        xIstar = Nx3 matrix of desired body x-axis direction, expressed as a
#                 unit vector in the I frame. xIstar(k,:)' is the 3x1
#                 direction at time tk = tVec(k).
#  
# S ---------- Structure with the following elements:
#
#  oversampFact = Oversampling factor. Let dtIn = R.tVec(2) - R.tVec(1). Then
#                 the output sample interval will be dtOut =
#                 dtIn/oversampFact. Must satisfy oversampFact >= 1.
#
#        state0 = State of the quad at R.tVec(1) = 0, expressed as a structure
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
#                 disturbance vector acting on the quad from R.tVec(k) to
#                 R.tVec(k+1).
#
# P ---------- Structure with the following elements:
#
#    quadParams = Structure containing all relevant parameters for the
#                 quad, as defined in quadParamsScript.m 
#
#     constants = Structure containing constants used in simulation and
#                 control, as defined in constantsScript.m 
#
#  sensorParams = Structure containing sensor parameters, as defined in
#                 sensorParamsScript.m
#
#
# OUTPUTS
#
# Q ---------- Structure with the following elements:
#
#          tVec = Mx1 vector of output sample time points, in seconds, where
#                 Q.tVec(1) = R.tVec(1), Q.tVec(M) = R.tVec(N), and M =
#                 (N-1)*oversampFact + 1.
#  
#         state = State of the quad at times in tVec, expressed as a
#                 structure with the following elements:
#                   
#                rMat = Mx3 matrix composed such that rMat(k,:)' is the 3x1
#                       position at tVec(k) in the I frame, in meters.
# 
#                eMat = Mx3 matrix composed such that eMat(k,:)' is the 3x1
#                       vector of Euler angles at tVec(k), in radians,
#                       indicating the attitude.
#
#                vMat = Mx3 matrix composed such that vMat(k,:)' is the 3x1
#                       velocity at tVec(k) with respect to the I frame
#                       and expressed in the I frame, in meters per
#                       second.
#                 
#           omegaBMat = Mx3 matrix composed such that omegaBMat(k,:)' is the
#                       3x1 angular rate vector expressed in the body frame in
#                       radians, that applies at tVec(k).
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  
#+==============================================================================+  
    # Vector of simulation segment start times
    tVecIn = R['tVec']
    dtIn   = tVecIn[2] - tVecIn[1]
    N      = len(tVecIn)

    # Oversampling causes the ODE solver to produce output at a finer time
    # resolution than dtIn. oversampFact is the oversampling factor.
    oversampFact = S['oversampFact']
    dtOut = dtIn/oversampFact

    # Set initial state  
    Xk = np.concatenate(
        (
            S['state0']['r'], 
            S['state0']['v'], 
            (euler2dcm(S['state0']['e'])).flatten(),
            S['state0']['omegaB'], 
            585*np.ones(4)
        )
    )

    # output structure
    Q = {}
    Q['tVec'] = tVecIn[0]
    Q['state'] = {}
    Q['state']['rMat'] = S['state0']['r']
    Q['state']['vMat'] = S['state0']['v']
    Q['state']['eMat'] = S['state0']['e']
    Q['state']['omegaBMat'] = S['state0']['omegaB']

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
        
        Saux = {}
        Saux['statek'] = {}
        Saux['statek']['rI']     = Xk[0:3]
        Saux['statek']['vI']     = Xk[3:6]
        Saux['statek']['RBI']    = Xk[6:15].reshape(3,3)
        Saux['statek']['omegaB'] = Xk[15:18]

        Raux = {}
        Raux['rIstark'] = R['rIstark'][k,:]
        Raux['vIstark'] = R['vIstark'][k,:]
        Raux['aIstark'] = R['aIstark'][k,:]
        Raux['xIstark'] = R['xIstark'][k,:]

        Fk, Raux['zIstark'] = trajectoryController(Raux,Saux,P)
        NBk = attitudeController(Raux,Saux,P)
        eak = voltageConverter( Fk, NBk, P )

        # Run ODE solver for segment
        solver = ode(quadOdeFunctionHF).set_integrator( 'dopri5' )
        solver.set_initial_value(Xk, ti).set_f_params(eak, S['distMat'][k,:], P)
        while solver.successful() and solver.t < tf-dt/3: # "-dt/3" is just to make sure that no numeric errors alter the intention
            t = solver.t+dt 
            Xk = solver.integrate(t)

            rI = Xk[0:3]
            vI = Xk[3:6]
            RBI = Xk[6:15].reshape(3,3)
            omegaB = Xk[15:18]
            
            # Add the data from the kth segment to your storage vector
            Q['tVec']              = np.append(  Q['tVec'], t )
            Q['state']['rMat']     = np.vstack( (Q['state']['rMat'], rI) )
            Q['state']['vMat']     = np.vstack( (Q['state']['vMat'], vI) )                        
            Q['state']['eMat']     = np.vstack( (Q['state']['eMat'], dcm2euler(RBI)) )
            Q['state']['omegaBMat']= np.vstack( (Q['state']['omegaBMat'], omegaB) )

    return Q