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

myFile = input("input file Path: ")
with open(myFile,'r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data.append([float(line) if is_float(line) else line  for line in split ])
data = np.array(data, dtype = float)

#print(data)

try:
	columns = len(data[0])# - 1 
except:
	columns =1

ndim = columns
print("ndim = ", ndim)

ndim = 1
if( ndim == 1): 
	
	#plt.hist(data, bins=50, range=None, density=True, weights=None,
	#	 cumulative=False, bottom=None, histtype='step', align='mid',
	#	 orientation='vertical', rwidth=None, log=False, color=None,
	#	 label=None, stacked=False)
	#histogram = sns.distplot(data[:,0],bins=None, hist=True, kde=True, rug=False,
	#			 fit=None, hist_kws=None, kde_kws=None, rug_kws=None,
	#			 fit_kws=None, color=None, vertical=False, norm_hist=True,
	#			 axlabel=None, label=None, ax=None)
	plt.plot(data[:,0])	
	#plt.title('A as that vs iters')
	plt.savefig('AvsIterMH.png')
	plt.show()
	plt.close()

if(ndim ==2 ):
	
    #data = np.array(data[:,:], dtype=[('x', float), ('y', float)])
    #data.sort(axis=0)
    
    #plt.scatter(data[:,0], data[:,1], s=5)
    data[:,1] = data[:,1];
    print(data)
    Data = pd.DataFrame(data)
    Data.drop( Data[Data[1] < -4.0e-05].index,inplace = True) 
    print(Data)
    plt.scatter(Data[0], np.exp(Data[1].values) ** 10000 )
    Data.to_csv('filterData.csv')
   # df.drop(df[df['Age'] < 25].index, inplace = True) 
    #plt.scatter(data[:,0], data[:,1])
    #matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
    #vmin=None, vmax=None, alpha=None, linewidths=None,
    #faceted=True, verts=None, hold=None, **kwargs)
    #plt.title("Frequency of Displacment")
    plt.savefig("FreqDisp.png")
    plt.show()
	
