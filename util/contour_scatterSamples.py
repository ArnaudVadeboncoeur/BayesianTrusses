import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
from random import seed, randint
import time
import scipy.interpolate

plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})

data1 = []
myFile1 = input("input file Path Surface: ")

data2 = []
myFile2 = input("input file Path Samples: ")

def is_float(string):
	try:
		return float(string)
	except ValueError:
		return False
		
		
surface = False
try:
	with open(myFile1,'r') as f:
		reader = f.readlines()
		for line  in reader:
			split = line.rstrip().split(" ")
			data1.append([float(line) if is_float(line) else line  for line in split ])
	data1 = np.array(data1, dtype = float)
	surface = True
except:
    data1 = [[0],[0],[0]]
    
OptLine = False
try:
    
    with open(myFile2,'r') as f:
	    reader = f.readlines()
	    for line  in reader:
		    split = line.rstrip().split(" ")
		    data2.append([float(line) if is_float(line) else line  for line in split ])
    data2 = np.array(data2, dtype = float)	
    OptLine = True
except:
    data2 = [[0],[0]]

#print(data)
if(surface == True):
	data1 = np.array(data1, dtype = float)

	x = data1[:, 0]
	y = data1[:, 1]
	z = data1[:, 2]
	z = np.where(z < 1e-4 , 1e-4, z) 

	N = 100 #number of points for plotting/interpolation

	xi = np.linspace(x.min(), x.max(), N)
	yi = np.linspace(y.min(), y.max(), N)
	zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

	#fig = plt.figure()
	plt.contourf(xi, yi, zi, 30)

if(OptLine == True):
    xo = data2[:, 0]
    yo = data2[:, 1]
    #plt.plot(xo, yo, c='r')
    plt.scatter(xo[1:], yo[1:], c='r', s=5)

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
	

