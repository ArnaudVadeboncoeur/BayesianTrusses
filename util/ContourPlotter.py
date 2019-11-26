import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
from random import seed, randint
import time

data = []

def is_float(string):
	try:
		return float(string)
	except ValueError:
		return False

with open('../results.dat','r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data.append([float(line) if is_float(line) else line  for line in split ])

data = np.array(data, dtype = 'O')

try:
	columns = len(data[0]) 
except:
	columns =1

ndim = columns
print("ndim = ", ndim)

x = data[:, 0]
y = data[:, 1]
X, Y = np.meshgrid(x, y)
Z = data[:, 2]

levels = 30

print(len(X), len(Y), len(Z))
norm = cm.colors.Normalize(vmax=Z.max(), vmin=Z.min())
cmap=plt.get_cmap('inferno')
plt.contourf(X, Y, Z, levels, norm=norm, cmap=cm.get_cmap(cmap, levels))

plt.title('Contour of funtion')
plt.savefig('ContourPlot.png')
plt.show()
plt.close()
	

