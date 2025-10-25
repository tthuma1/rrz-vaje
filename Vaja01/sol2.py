import numpy as np
import matplotlib.pyplot as plt

proj_fn = lambda f, p: (-f*p[0]/p[2], -f*p[1]/p[2])

Z0 = 1000
Y = 250
f = 10
acc = 50
meritev_na_sekundo = 10
st_meritev = 30 * meritev_na_sekundo + 1

y = []
for i in range(st_meritev):
    t = i / meritev_na_sekundo
    Z = Z0 + 1/2 * acc * t**2 if t != 0 else Z0
    y.append(f * Y/Z)

print(y)

times = np.linspace(0, 30, st_meritev)

plt.clf()
plt.plot(times, y, "r.")
plt.show()
