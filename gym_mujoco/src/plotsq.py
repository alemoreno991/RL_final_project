import pickle

data = pickle.load(open("./data/square.pkl", 'rb'))

states = data["test0"]["states"]

x = [st[0] for st in states]
y = [st[1] for st in states]

x0 = [0, 1, 1, 0, 0]
y0 = [0, 0, 1, 1, 0]

import matplotlib.pyplot as plt

plt.plot(x0, y0)
plt.scatter(x, y)
plt.show()