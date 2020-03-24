'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
from random import seed, randint
import time

data1 = []
myFile1 = input("input file Path Surface: ")

data2 = []
myFile2 = input("input file Path Opt Line: ")

def is_float(string):
	try:
		return float(string)
	except ValueError:
		return False

with open(myFile1,'r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data1.append([float(line) if is_float(line) else line  for line in split ])

with open(myFile2,'r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data2.append([float(line) if is_float(line) else line  for line in split ])



#print(data)
data1 = np.array(data1, dtype = float)
data2 = np.array(data2, dtype = float)
try:
	columns = len(data1[0]) 
except:
	columns =1

ndim = columns
print("ndim = ", ndim)


if ndim < 3 :
    plt.scatter( data1[:, 0],data1[:, 1] )
    plt.show()
    exit()

#data.sort(axis=0)
#print(data)
x = data1[:, 0]
y = data1[:, 1]
Z = data1[:, 2]

x2 = data2[:, 0]
y2 = data2[:, 1]
try:
	z2 = data2[:, 2]
except:
	z2 = np.full( len(x2), np.max(Z)*1.1 )

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.

#print(x, y, Z)
#surf = ax.scatter(x, y, Z)
ax.scatter(x, y, Z)
ax.scatter(x2, y2, z2, c= 'r')
ax.plot(x2, y2, z2, c= 'r')

# Customize the z axis.
#ax.set_zlim(0, 100)
#ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()



