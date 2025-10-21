import numpy as np
import matplotlib.pyplot as plt

proj_fn = lambda f, p: (-f*p[0]/p[2], -f*p[1]/p[2])

Z = 1000
speed = 0
Y = 250
f = 10
acc = 50

z = []
for i in range(30):
    z.append(f * Y/Z)
    speed += acc
    Z += speed

print(z)

times = np.linspace(0, 29, 30)

plt.clf()
plt.plot(times, z, "r.")
plt.show()
