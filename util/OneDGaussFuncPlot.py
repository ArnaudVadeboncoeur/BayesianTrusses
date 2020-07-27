

import matplotlib.pyplot as plt
import numpy as np

# 100 linearly spaced numbers
x = np.linspace(-10,10,100)

# the functions, which are y = sin(x) and z = cos(x) here

sig1 = 1
mu1  = 0
y = 1/(np.sqrt(2*np.pi)*sig1) * np.exp( -1/(2.*sig1**2)*(x-mu1)**2 )


sig2 = 2
mu2  = 1
z = 1/(np.sqrt(2*np.pi)*sig2) * np.exp( -1/(2.*sig2**2)*(x-mu2)**2 )

sig3 = 3
mu3  = 2
w = 1/(np.sqrt(2*np.pi)*sig3) * np.exp( -1/(2.*sig3**2)*(x-mu3)**2 )

sig4 = 4
mu4  = 3
v = 1/(np.sqrt(2*np.pi)*sig4) * np.exp( -1/(2.*sig4**2)*(x-mu4)**2 )

# setting the axes at the centre
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# plot the functions
plt.plot(x,y, 'c', label='sig = 1; mu = 0')
plt.plot(x,z, 'b', label='sig = 2; mu = 1')
plt.plot(x,w, 'r', label='sig = 3; mu = 2')
plt.plot(x,v, 'g', label='sig = 4; mu = 3')

plt.legend(loc='upper left')

# show the plot
plt.show()


