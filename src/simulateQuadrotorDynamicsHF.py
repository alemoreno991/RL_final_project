import numpy as np
from scipy.integrate import ode

from euler2dcm import euler2dcm
from dcm2euler import dcm2euler
from quadOdeFunctionHF import quadOdeFunctionHF

class SimulateQuadrotorDynamicsHF:
    
    def __init__(self, S):
    # simulateQuadrotorDynamicsHF : Simulates the dynamics of a quadrotor
    #                               aircraft (high-fidelity version).
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
    #         eaMat = (N-1)x4 matrix of motor voltage inputs.  eaMat(k,j) is the
    #                 constant (zero-order-hold) voltage for the jth motor over
    #                 the interval from tVec(k) to tVec(k+1).
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
    #+------------------------------------------------------------------------------+
    # References:
    #
    #
    # Author:  
    #+==============================================================================+  
        # Vector of simulation segment start times
        tVecIn = S['tVec']
        dtIn   = tVecIn[2] - tVecIn[1]
        N      = len(tVecIn)

        # Oversampling causes the ODE solver to produce output at a finer time
        # resolution than dtIn. oversampFact is the oversampling factor.
        oversampFact = S['oversampFact']
        dtOut = dtIn/oversampFact

        self.t = tVecIn[0]
        self.tf = tVecIn[-1]
        self.dt = dtOut

        # Set initial state  
        self.Xk = np.concatenate(
            (
                S['state0']['r'], 
                S['state0']['v'], 
                (euler2dcm(S['state0']['e'])).flatten(), 
                S['state0']['omegaB'], 
                585*np.ones(4)
            )
        )

        self.Parameters = S['P']
        # Run ODE solver for segment
        self.solver = ode(quadOdeFunctionHF).set_integrator( 'dopri5' )
        
        
        # output structure
        self.P = {}
        self.P['tVec'] = tVecIn[0]
        self.P['state'] = {}
        self.P['state']['rMat'] = S['state0']['r']
        self.P['state']['vMat'] = S['state0']['v']
        self.P['state']['eMat'] = S['state0']['e']
        self.P['state']['omegaBMat'] = S['state0']['omegaB']


    def step(self, action, disturbance):
        done = False
        self.solver.set_initial_value(self.Xk, self.t).set_f_params(action, disturbance, self.Parameters)
            
        if self.solver.successful() and self.solver.t < self.tf-self.dt/3: # "-dt/3" is just to make sure that no numeric errors alter the intention
            self.t = self.solver.t + self.dt 
            self.Xk = self.solver.integrate(self.t)

            rI     = self.Xk[0:3]
            vI     = self.Xk[3:6]
            RBI    = self.Xk[6:15].reshape(3,3)
            omegaB = self.Xk[15:18]
            
            # Add the data from the kth segment to your storage vector
            self.P['tVec']              = np.append(  self.P['tVec'], self.t )
            self.P['state']['rMat']     = np.vstack( (self.P['state']['rMat'], rI) )
            self.P['state']['vMat']     = np.vstack( (self.P['state']['vMat'], vI) )                        
            self.P['state']['eMat']     = np.vstack( (self.P['state']['eMat'], dcm2euler(RBI)) )
            self.P['state']['omegaBMat']= np.vstack( (self.P['state']['omegaBMat'], omegaB) )
        else:
            done = True

        return self.Xk, self.t, done

    def getResult(self):
        return self.P