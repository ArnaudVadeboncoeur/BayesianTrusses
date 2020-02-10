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

fileLoc = input("input file path: ")
with open(fileLoc,'r') as f:
	reader = f.readlines()
	for line  in reader:
		split = line.rstrip().split(" ")
		data.append([float(line) if is_float(line) else line  for line in split ])
data = np.array(data, dtype = float)

try:
	columns = len(data[0]) - 1
except:
	columns = 1

ndim = columns
print("ndim = ", ndim)

for i in range( 0 , columns + 1 ) :
    print("dim {0} mean = {1}".format(i, np.mean( data[:,i] ) ) )

dimPlot = int( input("input Dim to plot: ") )
while( dimPlot >= 0 ):	

    #plt.hist(data[:,0], bins=50, range=None, density=True, weights=None,
    #cumulative=False, bottom=None, histtype='step', align='mid',
    #orientation='vertical', rwidth=None, log=False, color=None,
    #label=None, stacked=False)

    #histogram = sns.distplot(data[:,0],bins=None, hist=True, kde=True, rug=False,
    #			 fit=None, hist_kws=None, kde_kws=None, rug_kws=None,
    #			 fit_kws=None, color=None, vertical=False, norm_hist=True,
    #			 axlabel=None, label=None, ax=None)
    
    histogram = sns.distplot(data[:,dimPlot])	
    #plt.title('A vs Disp')
    plt.savefig('AvsDof.png')
    plt.show()
    plt.close()
    dimPlot = int(input("input dim to plot: ") )

more = 'y'
if(dimPlot == -2):
    while(more == 'y'):
        dim1 = int(input("input dimension 1: ") )
        dim2 = int(input("input dimension 2: ") )

        print("Plotting dimension numbers {0} and {1}".format( dim1,dim2 ))
        df = pd.DataFrame(data[:, [dim1,dim2] ], columns=["x", "y"])
        data = None
        #print("Number of Dimensions is 2!")

        sns.jointplot(x="x", y="y", data=df, kind="kde")
        #plt.title("A vs Disp")
        plt.savefig("A vs Disp.png")
        plt.show()
        more = input("continue? y/n: ")

