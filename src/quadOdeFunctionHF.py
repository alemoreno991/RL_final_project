import numpy as np
from crossProductEquivalent import crossProductEquivalent
from auxiliar import e3

def quadOdeFunctionHF( t, X, eaVec, distVec, P ):
# quadOdeFunctionHF : Ordinary differential equation function that models
#                     quadrotor dynamics -- high-fidelity version.  For use
#                     with one of Matlab's ODE solvers (e.g., ode45).
#
#
# INPUTS
#
# t ---------- Scalar time input, as required by Matlab's ODE function
#              format.
#
# X ---------- Nx-by-1 quad state, arranged as 
#
#              X = [rI',vI',RBI(1,1),RBI(2,1),...,RBI(2,3),RBI(3,3),...
#                   omegaB',omegaVec']'
#
#              rI = 3x1 position vector in I in meters
#              vI = 3x1 velocity vector wrt I and in I, in meters/sec
#             RBI = 3x3 attitude matrix from I to B frame
#          omegaB = 3x1 angular rate vector of body wrt I, expressed in B
#                   in rad/sec
#        omegaVec = 4x1 vector of rotor angular rates, in rad/sec.
#                   omegaVec(i) is the angular rate of the ith rotor.
#
#    eaVec --- 4x1 vector of voltages applied to motors, in volts.  eaVec(i)
#              is the constant voltage setpoint for the ith rotor.
#
#  distVec --- 3x1 vector of constant disturbance forces acting on the quad's
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
# Author:  
#+==============================================================================+
    # Unpack constants and parameters needed
    m = P['quadParams']['m']
    Jq = P['quadParams']['Jq']
    g = P['constants']['g']
    cm = P['quadParams']['cm']
    taum = P['quadParams']['taum']

    # Unpack the state vector's elements
    rI       = X[0:3]
    vI       = X[3:6]
    RBI      = X[6:15].reshape(3,3)
    omegaB   = X[15:18]
    omegaVec = X[18:22]

    if np.linalg.norm(vI) > 0:
        vI_u = (vI/np.linalg.norm(vI))
    else:
        vI_u = np.zeros_like(vI)

    # First order differential equation
    rI_dot     = vI 
    vI_dot     = -da(RBI@e3,vI,P)*vI_u -g*e3 + (RBI.T @ FB(omegaVec,P))/m + distVec/m
    RBI_dot    = -crossProductEquivalent(omegaB) @ RBI
    omegaB_dot = np.linalg.inv(Jq) @ ( NB(omegaVec,P) - crossProductEquivalent(omegaB) @ Jq @ omegaB )
    omegaVec_dot = (cm*eaVec - omegaVec)/taum

    # Return the state vector's time derivative 
    return np.concatenate( (rI_dot, vI_dot, RBI_dot.flatten(), omegaB_dot, omegaVec_dot) )

def FB(omegaVec, P):
    kF = P['quadParams']['kF']

    FB = np.zeros(3)
    for i in range(4):
        FB += kF[i] * omegaVec[i]**2 * e3
    
    return FB

def NB(omegaVec, P):
    kN = P['quadParams']['kN']
    kF = P['quadParams']['kF']
    s = -P['quadParams']['omegaRdir']
    r =  P['quadParams']['rotor_loc']
    
    NB = np.zeros(3)
    for i in range(4):
        Fi = kF[i]*omegaVec[i]**2 * e3
        Ni = s[i]*kN[i]*omegaVec[i]**2 * e3
        NB +=  Ni + crossProductEquivalent(r[:,i]) @ Fi
    
    return NB

def da( zBody_I, vI, P) -> float:
    Cd  = P['quadParams']['Cd']
    Ad  = P['quadParams']['Ad']
    rho = P['constants']['rho']
    
    return 0.5*Cd*Ad*rho*fd( zBody_I, vI )

def fd( zBody_I, vI ) -> float:
    return np.inner(zBody_I,vI)*np.linalg.norm(vI)