import numpy as np
from crossProductEquivalent import crossProductEquivalent

def attitudeController(R,S,P):
# attitudeController : Controls quadcopter toward a reference attitude
#
#
# INPUTS
#
# R ---------- Structure with the following elements:
#
#       zIstark = 3x1 desired body z-axis direction at time tk, expressed as a
#                 unit vector in the I frame.
#
#       xIstark = 3x1 desired body x-axis direction, expressed as a
#                 unit vector in the I frame.
#
# S ---------- Structure with the following elements:
#
#        statek = State of the quad at tk, expressed as a structure with the
#                 following elements:
#                   
#                  rI = 3x1 position in the I frame, in meters
# 
#                 RBI = 3x3 direction cosine matrix indicating the
#                       attitude
#
#                  vI = 3x1 velocity with respect to the I frame and
#                       expressed in the I frame, in meters per second.
#                 
#              omegaB = 3x1 angular rate vector expressed in the body frame,
#                       in radians per second.
#
# P ---------- Structure with the following elements:
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
# NBk -------- Commanded 3x1 torque expressed in the body frame at time tk, in
#              N-m.
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  
#+==============================================================================+  
    Jq = P['quadParams']['Jq']

    RBIstar = getRBIstar( R['zIstark'], R['xIstark'] )
    RE = RBIstar @ S['statek']['RBI'].T
    eE = np.array( [RE[1,2]-RE[2,1], RE[2,0]-RE[0,2], RE[0,1] - RE[1,0]] )
    eE_dot = -S['statek']['omegaB'] # this is an approximation valid when 
                                    # omegaBstar = 0

    K  = np.diag( [0.35, 0.7, 0.8] )
    Kd = np.diag( [0.1 ,0.2 , 0.25] )

    NBk = K@eE + Kd@eE_dot
    + crossProductEquivalent(S['statek']['omegaB']) @ Jq @ S['statek']['omegaB']

    return NBk

def getRBIstar( zIstar, xIstar ):
    b = crossProductEquivalent(zIstar) @ xIstar
    b /= np.linalg.norm(b)
    a = crossProductEquivalent(b) @ zIstar

    return np.stack( (a, b, zIstar) )
