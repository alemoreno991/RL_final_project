from gym_multirotor import utils
import numpy as np

roll = np.deg2rad(85)
pitch= np.deg2rad(-85)
yaw  = np.deg2rad(10)

quat = utils.euler2quat([roll, pitch, yaw])

print(quat)