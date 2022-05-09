import numpy as np
from quadParamsScript import quadParams

def voltageConverter( Fk, NBk, P ):
# voltageConverter : Generates output voltages appropriate for desired
#                    torque and thrust.
#
#
# INPUTS
#
# Fk --------- Commanded total thrust at time tk, in Newtons.
#
# NBk -------- Commanded 3x1 torque expressed in the body frame at time tk, in
#              N-m.
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
# eak -------- Commanded 4x1 voltage vector to be applied at time tk, in
#              volts. eak(i) is the voltage for the ith motor.
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  
#+==============================================================================+  
    r = P['quadParams']['rotor_loc']
    kT= P['quadParams']['kN']/P['quadParams']['kF']
    s = -P['quadParams']['omegaRdir']
    
    alpha = 1   # TODO: adjust the alpha so that the forces or each propeller never exceed fmax
    beta = 0.9

    fmax = Fmax(P)
    f = min(Fk, 4*beta*fmax)
    N = alpha*NBk

    G = np.ones(4)
    G = np.vstack( (G,  r[1]) )
    G = np.vstack( (G, -r[0]) )
    G = np.vstack( (G,  kT*s  ) )

    F = np.linalg.inv(G) @ np.concatenate( (np.array([f]),N) )
            
    # saturate output
    F = np.where(F < 0, 0, F)


    # translate the motors' forces into voltages
    omegak = np.sqrt( F/P['quadParams']['kF'] )
    eak = omegak/P['quadParams']['cm']
    return eak

def Fmax(P):
    cm = np.min(P['quadParams']['cm'])
    kF = np.min(P['quadParams']['kF'])
    eaMax = quadParams['eamax']
    return kF*(eaMax*cm)**2