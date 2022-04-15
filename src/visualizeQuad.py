from timeit import repeat
from euler2dcm import euler2dcm
from auxiliar import e1, e2, e3

from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 

def visualizeQuad(S):
# visualizeQuad : Takes in an input structure S and visualizes the resulting
#                 3D motion in approximately real-time.  Outputs the data
#                 used to form the plot.
#
#
# INPUTS
#
# S ---------- Structure with the following elements:
#
#           rMat = 3xM matrix of quad positions, in meters
#
#           eMat = 3xM matrix of quad attitudes, in radians
#
#           tVec = Mx1 vector of times corresponding to each measurement in
#                  xevwMat
#
#  plotFrequency = The scalar number of frames of the plot per each second of
#                  input data.  Expressed in Hz.
#
#         bounds = 6x1, the 3d axis size vector
#
#    makeGifFlag = Boolean (if true, export the current plot to a .gif)
#
#    gifFileName = A string with the file name of the .gif if one is to be
#                  created.  Make sure to include the .gif exentsion.
#
#
# OUTPUTS
#
# P ---------- Structure with the following elements:
#
#          tPlot = Nx1 vector of time points used in the plot, sampled based
#                  on the frequency of plotFrequency
#
#          rPlot = 3xN vector of positions used to generate the plot, in
#                  meters.
#
#          ePlot = 3xN vector of attitudes used to generate the plot, in
#                  radians.
#
#+------------------------------------------------------------------------------+
# References: "visualizeQuad.m" by Nick Montalbano
#
#
# Author:  Alejandro Moreno
#+==============================================================================+
  
    # Important params
    P = {}
    figureNumber = 42
    fig = plt.figure(num=figureNumber, figsize=(20,15))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.cla()
    

    fcounter = 0; #frame counter for gif maker
    m = len(S['tVec'])

    # Parameters for the rotors
    r_rotor = .062
    rotorLocations = np.array( [[0.105,  0.105, -0.105, -0.105],
                                [0.105, -0.105,  0.105, -0.105],
                                [0    ,  0    ,  0    ,  0]] )

    # Determines the location of the corners of the body box in the body frame,
    # in meters
    bpts = np.array([[120,  120, -120, -120,  120,  120, -120, -120],
                     [28,  -28,   28,  -28,   28,  -28,   28,  -28 ],
                     [20,   20,   20,   20,   -30, -30,  -30,  -30 ]] )*1e-3
    
    body, circpts = set_quad_geometry( r_rotor, rotorLocations, bpts )

    if m == 1: # Plot single epoch if m==1
        # Extract params
        RIB = euler2dcm(S['eMat'][0]).T
        r = S['rMat'][0]
        
        draw_quad( r, RIB, body, circpts, rotorLocations, ax, axis_bounds=S['bounds'] )
        fig.show()

        P['tPlot'] = S['tVec']
        P['rPlot'] = S['rMat']
        P['ePlot'] = S['eMat']

    elif m > 1: # Interpolation must be used to smooth timing
        
        # Create time vectors
        tf = 1/S['plotFrequency']
        tmax = S['tVec'][-1] 
        tmin = S['tVec'][0]
        tPlot = np.arange( tmin, tmax, tf )
        tPlotLen = len(tPlot)
        
        # Interpolate to regularize times
        t2unique, indUnique = np.unique(S['tVec'],  return_index=True)
        
        rPlot = np.empty( (len(tPlot),3) )
        rPlot[:,0] = np.interp(tPlot, t2unique, S['rMat'][indUnique,0])
        rPlot[:,1] = np.interp(tPlot, t2unique, S['rMat'][indUnique,1])
        rPlot[:,2] = np.interp(tPlot, t2unique, S['rMat'][indUnique,2])

        ePlot = np.empty( (len(tPlot),3))
        ePlot[:,0] = np.interp(tPlot, t2unique, S['eMat'][indUnique,0])
        ePlot[:,1] = np.interp(tPlot, t2unique, S['eMat'][indUnique,1])
        ePlot[:,2] = np.interp(tPlot, t2unique, S['eMat'][indUnique,2])
        
        ani = FuncAnimation(
            fig, 
            animate, 
            fargs=(rPlot, ePlot, body, circpts, rotorLocations, ax, S['bounds']),  
            frames=tPlotLen, 
            interval=int(tf*1000), 
            repeat=False 
        )

        if True == S['makeGifFlag']:
            path = '../output/'
            os.makedirs( path, exist_ok=True )
            ani.save( path + S['gifFileName'], fps=int(1/tf) )
        plt.show()            
        
        P['tPlot'] = tPlot
        P['ePlot'] = ePlot
        P['rPlot'] = rPlot
    
    return P

def animate( i, rPlot, ePlot, body, circpts, rotorLocations, ax, axis_bounds ):
    # Extract data
    r = rPlot[i,:]
    RIB = euler2dcm(ePlot[i,:]).T

    draw_quad( r, RIB, body, circpts, rotorLocations, ax, axis_bounds )



def draw_quad( r, RIB, body, circpts, rotorLocations, ax, axis_bounds ):
    ax.cla()
    # RBG scaled on [0,1] for the color orange
    rgbOrange = [1, .4, 0]

    r = r.reshape(3,1)
    
    # Translate, rotate, and plot the rotors
    rotor1_circle = r@np.ones((1,20)) + RIB@( circpts + rotorLocations[:,[0]]@np.ones((1,20)) )
    rotor1plot = ax.plot3D(rotor1_circle[0,:], rotor1_circle[1,:], rotor1_circle[2,:], color=rgbOrange)
    
    rotor2_circle = r@np.ones((1,20)) + RIB@( circpts + rotorLocations[:,[1]]*np.ones((1,20)) )
    rotor2plot = ax.plot3D(rotor2_circle[0,:], rotor2_circle[1,:], rotor2_circle[2,:], color=rgbOrange)
    
    rotor3_circle = r@np.ones((1,20)) + RIB@( circpts + rotorLocations[:,[2]]@np.ones((1,20)) )
    rotor3plot = ax.plot3D(rotor3_circle[0,:], rotor3_circle[1,:], rotor3_circle[2,:], color=rgbOrange)
    
    rotor4_circle = r@np.ones((1,20)) + RIB@( circpts + rotorLocations[:,[3]]@np.ones((1,20)) )
    rotor4plot = ax.plot3D(rotor4_circle[0,:], rotor4_circle[1,:], rotor4_circle[2,:], color=rgbOrange)
    
    # Plot the body 
    b1r = r@np.ones((1,4)) + RIB@body[0] 
    b2r = r@np.ones((1,4)) + RIB@body[1]
    b3r = r@np.ones((1,4)) + RIB@body[2]
    b4r = r@np.ones((1,4)) + RIB@body[3] 
    b5r = r@np.ones((1,4)) + RIB@body[4]
    b6r = r@np.ones((1,4)) + RIB@body[5]
    X = np.stack( (b1r[0,:].T, b2r[0,:].T, b3r[0,:].T, b4r[0,:].T, b5r[0,:].T, b6r[0,:].T) )
    Y = np.stack( (b1r[1,:].T, b2r[1,:].T, b3r[1,:].T, b4r[1,:].T, b5r[1,:].T, b6r[1,:].T) )
    Z = np.stack( (b1r[2,:].T, b2r[2,:].T, b3r[2,:].T, b4r[2,:].T, b5r[2,:].T, b6r[2,:].T) )
    ax.plot_surface(X,Y,Z, color='gray')

    # Plot the body axes
    bodyX = 0.2*RIB@e1 
    bodyY = 0.2*RIB@e2
    bodyZ = 0.2*RIB@e3
    axis1 = ax.quiver( r[0], r[1], r[2], bodyX[0], bodyX[1], bodyX[2], arrow_length_ratio=0.2 ,color='red'   )
    axis2 = ax.quiver( r[0], r[1], r[2], bodyY[0], bodyY[1], bodyY[2], arrow_length_ratio=0.2 ,color='blue'  )
    axis3 = ax.quiver( r[0], r[1], r[2], bodyZ[0], bodyZ[1], bodyZ[2], arrow_length_ratio=0.2 ,color='green' )
    
    ax.set_xlim(axis_bounds[0:2])
    ax.set_ylim(axis_bounds[2:4])
    ax.set_zlim(axis_bounds[4:6])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    

def set_quad_geometry( r_rotor, rotorLocations, bpts ):
    # Rectangles representing each side of the body box
    body = []
    body.append( bpts[:, [0, 4, 5, 1] ] )
    body.append( bpts[:, [0, 4, 6, 2] ] )
    body.append( bpts[:, [2, 6, 7, 3] ] )
    body.append( bpts[:, [0, 2, 3, 1] ] )
    body.append( bpts[:, [4, 6, 7, 5] ] )
    body.append( bpts[:, [1, 5, 7, 3] ] )

    # Create a circle for each rotor
    t_circ = np.linspace( 0, 2*np.pi, 20)
    circpts = np.zeros((3,20))
    for i in np.arange(0,20):
        circpts[:,i] = r_rotor* np.array( [np.cos(t_circ[i]), np.sin(t_circ[i]), 0] )

    return body, circpts