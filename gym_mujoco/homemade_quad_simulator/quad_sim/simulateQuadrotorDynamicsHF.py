import numpy as np
from scipy.integrate import ode

from euler2dcm import euler2dcm
from dcm2euler import dcm2euler
from quadOdeFunctionHF import quadOdeFunctionHF
from constantsScript import constants
from quadParamsScript import quadParams

class SimulateQuadrotorDynamicsHF:
    
    def __init__(self, Q, R):
    # simulateQuadrotorDynamicsHF : Simulates the dynamics of a quadrotor
    #                               aircraft (high-fidelity version).
    #
    #
    # INPUTS
    #
    # Q -------------- Matrix that weights the state costs
    #
    # R -------------- Matrix that weights the state costs
    #
    #+------------------------------------------------------------------------------+
    # References:
    #
    #
    # Author:  
    #+==============================================================================+  
        # Vector of simulation segment start times
        self.t0 = 0

        # Oversampling causes the ODE solver to produce output at a finer time
        # resolution than dtIn. oversampFact is the oversampling factor.
        self.dt = 5e-4

        # Weight matrices
        self.Q = Q
        self.R = R

        # Quadrotor parameters and constants
        self.Parameters = {}
        self.Parameters['quadParams'] = quadParams
        self.Parameters['constants']  = constants


        # Run ODE solver for segment
        self.solver = ode(quadOdeFunctionHF).set_integrator( 'dopri5' )

    def reset(self):
    # reset : Resets the environment and the simulation
    #
    #
    #+------------------------------------------------------------------------------+
    # References:
    #
    #
    # Author:  Alejandro Moreno (ale.moreno991@utexas.edu)
    #+==============================================================================+  
        # Reset the time
        self.t = self.t0

        # Reset the cost function
        self.reward = 0

        # Initialize the state vector
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
        rI     = np.zeros(3)
        vI     = np.zeros(3)
        eK     = np.zeros(3)
        omegaB = np.zeros(3)

        # Set initial state  
        self.Xk = np.concatenate(
            (
                rI, 
                vI, 
                euler2dcm(eK).flatten(), 
                omegaB, 
                585*np.ones(4)
            )
        )

        # output structure
        self.P = {}
        self.P['tVec'] = self.t0
        self.P['state'] = {}
        self.P['state']['rMat'] = rI
        self.P['state']['vMat'] = vI
        self.P['state']['eMat'] = eK
        self.P['state']['omegaBMat'] = omegaB

        return np.concatenate( (rI, vI, eK, omegaB) )


    def step(self, action, disturbance):
    ###########################################################################
    # INPUTS: 
    #
    #       actions = 1x4 vector of motor voltage inputs.
    #
    #   disturbance = 1x3 matrix of disturbance forces acting on the quad's
    #                 center of mass, expressed in Newtons in the world frame.
    #                 
    ###########################################################################
        done = False
        self.reward = 0
        stateVtr = np.empty(12)
        self.solver.set_initial_value(self.Xk, self.t).set_f_params(action, disturbance, self.Parameters)  

        if self.solver.successful():
            self.t = self.solver.t + self.dt 
            self.Xk = self.solver.integrate(self.t)

            rI     = self.Xk[0:3]
            vI     = self.Xk[3:6]
            RBI    = self.Xk[6:15].reshape(3,3)
            eK     = dcm2euler(RBI)
            omegaB = self.Xk[15:18]

            stateVtr = np.concatenate( (rI, vI, eK, omegaB) )
            self.reward -= 1e-3 * ((stateVtr @ self.Q @ stateVtr) + (action @ self.R @ action))
            self.reward += 1

             # End the episode if the roll or pitch is greater than a certain threshold
            if np.abs(eK[1]) > np.deg2rad(60) or np.abs(eK[2]) > np.deg2rad(60):
                self.reward -= 1000
                done = True
    
            # Add the data from the kth segment to your storage vector
            self.P['tVec']              = np.append(  self.P['tVec'], self.t )
            self.P['state']['rMat']     = np.vstack( (self.P['state']['rMat'], rI) )
            self.P['state']['vMat']     = np.vstack( (self.P['state']['vMat'], vI) )                        
            self.P['state']['eMat']     = np.vstack( (self.P['state']['eMat'], eK ) )
            self.P['state']['omegaBMat']= np.vstack( (self.P['state']['omegaBMat'], omegaB) )
        else:
            print("Error with the ODE solver")
            exit()

        return stateVtr, self.reward, done

    def getResult(self):
        return self.P