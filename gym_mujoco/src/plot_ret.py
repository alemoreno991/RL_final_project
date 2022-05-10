import matplotlib.pyplot as plt
import numpy as np
import glob

f = glob.glob('./trained_agents/0_attempt/*_SAC_vanilla*/progress.txt')

avg_ret = []
avg_stddev = []

for name in f:
  file = open(name, 'r')
  lines = file.readlines()
  del lines[0]
  a = [float(i.split()[1]) for i in lines]
  b = [float(i.split()[2]) for i in lines]
  if ('_5' in name):
    print(a)
    print(b)
    print()
  if (len(a) > 21):
    avg_ret.append(a[:34])
    avg_stddev.append(b[:34])

clr = ['blue','grey', 'red', 'orange', 'green', 'yellow', 'black']
avg_ret = np.array(avg_ret)
avg_stddev = np.array(avg_stddev)

ctr = 0
for i in range(0,5):
  if clr[i] == 'grey':
    continue
  plt.plot(list(range(len(avg_ret[i]))), avg_ret[i], color=clr[i ], label="Run {}".format(ctr))
  plt.fill_between(list(range(len(avg_ret[i]))), avg_ret[i]-avg_stddev[i],avg_ret[i]+avg_stddev[i], color=clr[i ], alpha=0.2)
  ctr+=1
  # plt.errorbar(list(range(len(avg_ret[i]))), avg_ret[i], yerr=avg_stddev[i])
plt.title("SAC Avg & Std Dev Episode Return (Vanilla)")
plt.xlabel("Epoch")
plt.ylabel("Return")
plt.legend()
plt.show()