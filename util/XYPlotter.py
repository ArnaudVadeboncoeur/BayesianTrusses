import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
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
data = np.array(data, dtype = float)
print(data)

try:
	columns = len(data[0])# - 1 
except:
	columns =1

ndim = columns
print("ndim = ", ndim)


if( ndim == 1): 
	
	#plt.hist(data, bins=50, range=None, density=True, weights=None,
	#	 cumulative=False, bottom=None, histtype='step', align='mid',
	#	 orientation='vertical', rwidth=None, log=False, color=None,
	#	 label=None, stacked=False)
	histogram = sns.distplot(data[:,0],bins=None, hist=True, kde=True, rug=False,
				 fit=None, hist_kws=None, kde_kws=None, rug_kws=None,
				 fit_kws=None, color=None, vertical=False, norm_hist=True,
				 axlabel=None, label=None, ax=None)	
	plt.title('A vs Disp')
	plt.savefig('AvsDof.png')
	plt.show()
	plt.close()

if(ndim ==2 ):
	
    #data = np.array(data[:,:], dtype=[('x', float), ('y', float)])
    data.sort(axis=0)
    print(data)
    #plt.scatter(data[:,0], data[:,1], s=5)
    plt.plot(data[:,0], data[:,1])
    #matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
    #vmin=None, vmax=None, alpha=None, linewidths=None,
    #faceted=True, verts=None, hold=None, **kwargs)
    plt.title("Frequency of Displacment DOF6-lognormal member area")
    plt.savefig("FreqDisp.png")
    plt.show()
	
