import numpy as np 

def euler2dcm(e):
# euler2dcm : Converts Euler angles phi = e(1), theta = e(2), and psi = e(3)
#             (in radians) into a direction cosine matrix for a 3-1-2 rotation.
#
# Let the world (W) and body (B) reference frames be initially aligned.  In a
# 3-1-2 order, rotate B away from W by angles psi (yaw, about the body Z
# axis), phi (roll, about the body X axis), and theta (pitch, about the body Y
# axis).  R_BW can then be used to cast a vector expressed in W coordinates as
# a vector in B coordinates: vB = R_BW * vW
#
# INPUTS
#
# e ---------- 3-by-1 vector containing the Euler angles in radians: phi =
#              e(1), theta = e(2), and psi = e(3)
#
#
# OUTPUTS
#
# R_BW ------- 3-by-3 direction cosine matrix 
# 
#+------------------------------------------------------------------------------+
# References:
#
#
# Author:  Alejandro Moreno (ale.moreno991@utexas.edu)
#+==============================================================================+  
    yaw   = e[2]
    roll  = e[0]
    pitch = e[1]

    return R2(pitch) @ R1(roll) @ R3(yaw) 

def R3( psi ):
    cos_psi   = np.cos(psi)
    sin_psi   = np.sin(psi)

    return np.array([[ cos_psi, sin_psi, 0],
                     [-sin_psi, cos_psi, 0],
                     [ 0      , 0      , 1]]
              )

def R1( phi ):
    cos_phi   = np.cos(phi)
    sin_phi   = np.sin(phi)

    return np.array([[ 1, 0      ,   0],
                     [ 0, cos_phi, sin_phi],
                     [ 0,-sin_phi, cos_phi]]
              )

def R2( theta ):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array([[ cos_theta, 0, -sin_theta],
                     [ 0        , 1,  0        ],
                     [ sin_theta, 0,  cos_theta]]
              )


if __name__ == '__main__' :
    e = np.array([0,0, np.deg2rad(60) ])
    print(euler2dcm( e ))
