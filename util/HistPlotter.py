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


with open('/home/arnaudv/git/BayesianTrusses/app/BTrussMCMC_MH/results.dat','r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data.append([float(line) if is_float(line) else line  for line in split ])
data = np.array(data, dtype = float)
print(data)

try:
	columns = len(data[0]) - 1
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

if(ndim >=2 ):

	randInt1=0
	randInt2=1
	
	if(ndim >2):
		seed( time.time() % 100)
		randInt1 = 0
		randInt2 = 0
		while(randInt1 == randInt2):
			randInt1 = randint(0, ndim-1)
			randInt2 = randint(0, ndim-1)
		if(randInt1 == ndim or randInt1==ndim):
			print("Plotting y as one of the axes!")	
	print("Plotting dimension numbers {0} and {1}".format(randInt1,randInt2))
	df = pd.DataFrame(data[:, [randInt1,randInt2] ], columns=["x", "y"])
	data = None
	#print("Number of Dimensions is 2!")

	sns.jointplot(x="x", y="y", data=df, kind="kde")
	plt.title("A vs Disp")
	plt.savefig("A vs Disp.png")
	plt.show()
	
