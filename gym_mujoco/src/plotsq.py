import pickle

data = pickle.load(open("./data/square.pkl", 'rb'))

states = data["test0"]["states"]

x = [st[0] for st in states]
y = [st[1] for st in states]

x0 = [0, 2, 2, 0, 0]
y0 = [0, 0, 2, 2, 0]

import matplotlib.pyplot as plt

plt.title('Square Trajectory Top-Down View')
plt.ylabel('Y Position (m)')
plt.xlabel('X Position (m)')
plt.plot(x0, y0)
plt.scatter(x, y)
plt.show()