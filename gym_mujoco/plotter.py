import matplotlib.pyplot as plt
import numpy as np

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.001, filename=""):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    if not filename=="":
        print("saving plot....")
        plt.savefig(filename)

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line1.set_xdata(x_vec)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # adjust time limit
    if np.min(x_vec)<=line1.axes.get_xlim()[0] or np.max(x_vec)>=line1.axes.get_xlim()[1]:
        plt.xlim([np.min(x_vec),np.max(x_vec)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1
