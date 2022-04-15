import numpy as np
from crossProductEquivalent import crossProductEquivalent
from auxiliar import e1, e2, e3, parseXFormat1, stateVtrFormat1, parseXFormat1, parseParams

def quadOdeFunction( t, X, omegaVec, distVec, P ):
# quadOdeFunction : Ordinary differential equation function that models
#                   quadrotor dynamics.  For use with one of Matlab's ODE
#                   solvers (e.g., ode45).
#
#
# INPUTS
#
# t ---------- Scalar time input, as required by Matlab's ODE function
#              format.
#
# X ---------- Nx-by-1 quad state, arranged as 
#
#              X = [rI',vI',RBI(1,1),RBI(2,1),...,RBI(2,3),RBI(3,3),omegaB']'
#
#              rI = 3x1 position vector in I in meters
#              vI = 3x1 velocity vector wrt I and in I, in meters/sec
#             RBI = 3x3 attitude matrix from I to B frame
#          omegaB = 3x1 angular rate vector of body wrt I, expressed in B
#                   in rad/sec
#
# omegaVec --- 4x1 vector of rotor angular rates, in rad/sec.  omegaVec(i)
#              is the constant rotor speed setpoint for the ith rotor.
#
# distVec --- 3x1 vector of constant disturbance forces acting on the quad's
#              center of mass, expressed in Newtons in I.
#
# P ---------- Structure with the following elements:
#
#    quadParams = Structure containing all relevant parameters for the
#                 quad, as defined in quadParamsScript.m 
#
#     constants = Structure containing constants used in simulation and
#                 control, as defined in constantsScript.m 
#
# OUTPUTS
#
# Xdot ------- Nx-by-1 time derivative of the input vector X
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  Alejandro Moreno (ale.moreno991@utexas.edu)
#+==============================================================================+
    rI, vI, RBI, omegaB = parseXFormat1(X)
    m, g, J = parseParams( P )

    rI_dot     = vI 
    vI_dot     = -g*e3 + (RBI.T @ FB(omegaVec,P))/m + distVec/m
    RBI_dot    = -crossProductEquivalent(omegaB) @ RBI
    omegaB_dot = np.linalg.inv(J) @ ( NB(omegaVec,P) - crossProductEquivalent(omegaB) @ J @ omegaB )
    
    return stateVtrFormat1(rI_dot, vI_dot, RBI_dot, omegaB_dot)

def FB(omegaVec, P):
    FB = np.zeros(3)

    for i in range(4):
        FB += P['kF'][i] * omegaVec[i]**2 * e3
    
    return FB

def NB(omegaVec, P):
    s = np.array([-1, 1, -1, 1])
    NB = np.zeros(3)
    
    for i in range(4):
        ri = P['rotor_loc'][:,i]
        Fi = P['kF'].item(i)*omegaVec.item(i)**2 * e3
        Ni = s.item(i)*P['kN'].item(i)*omegaVec.item(i)**2 * e3
        NB +=  Ni + crossProductEquivalent(ri) @ Fi
    
    return NB