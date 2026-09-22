import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(10000) * (5.0 - 0.001) + 0.001
x = np.log(x)
t = np.random.rand(10000)
xx, tt = x.reshape(-1, 1), t.reshape(-1, 1)

plt.figure()
plt.scatter(xx, tt, s=5)
plt.show()