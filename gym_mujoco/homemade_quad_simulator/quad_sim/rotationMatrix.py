import numpy as np 
from crossProductEquivalent import crossProductEquivalent

def rotationMatrix( aHat, phi ):
# rotationMatrix : Generates the rotation matrix R corresponding to a rotation
#                  through an angle phi about the axis defined by the unit
#                  vector aHat. This is a straightforward implementation of
#                  Euler’s formula for a rotation matrix.
#
# INPUTS
#
# aHat ------- 3-by-1 unit vector constituting the axis of rotation.
#
# phi -------- Angle of rotation, in radians.
#
#
# OUTPUTS
#
# R ---------- 3-by-3 rotation matrix
#
#+------------------------------------------------------------------------------+
# References:
#
#
# Author: Alejandro Moreno (ale.moreno991@utexas.edu)
#+==============================================================================+
    if 1 != np.linalg.norm(aHat):
        print( "Call to rotationMatrix: aHat is not a unit vector!")
        print( "I'll proceed by normalizing it" )
        aHat = aHat/np.linalg.norm(aHat)

    R = np.cos(phi)*np.eye(3)
    R += (1 - np.cos(phi))*np.outer( aHat, aHat )
    R -= np.sin(phi)*crossProductEquivalent( aHat ) 
    
    return R