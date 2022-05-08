from ast import Param
import numpy as np
from euler2dcm import euler2dcm
from dcm2euler import dcm2euler

e1 = np.array( [1,0,0] )
e2 = np.array( [0,1,0] )
e3 = np.array( [0,0,1] )

def stateVtrFormat1(r, v, RBI, omegaB):
    return np.concatenate( (r, v, RBI.flatten(), omegaB) )

def stateVtrFormat2(r, v, e, omegaB):
    return np.concatenate( (r, v, e, omegaB) )

def stateVtrFormat1ToFormat2(X):
    r, v, RBI, omegaB = parseXFormat1(X)
    e = dcm2euler(RBI)
    return np.concatenate( (r, v, e, omegaB) )

def stateVtrFormat2ToFormat1(X):
    r, v, e, omegaB = parseXFormat2(X)
    RBI = euler2dcm(e)
    return stateVtrFormat1(r, v, RBI.flatten(), omegaB)

def parseXFormat1(X):
    r = X[0:3]
    v = X[3:6]    
    RBI = X[6:15].reshape(3,3)
    omegaB = X[15:18]
    return r, v, RBI, omegaB

def parseXFormat2(X):
    r = X[0:3]
    v = X[3:6]
    e = X[6:9]
    omegaB = X[9:12]
    return r, v, e, omegaB
    
def parseParams(P):
    m = P['m']
    g = P['g']
    J = P['J']
    return m, g, J