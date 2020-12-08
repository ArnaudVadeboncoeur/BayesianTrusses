import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from random import seed, randint
import time


plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})
#data = []

def is_float(string):
	try:
		return float(string)
	except ValueError:
		return False
moreFiles = "true"
ctr = 0
while(moreFiles == "true"):
	data = []
	ctr += 1
	myFile = input("input file Path: ")
	print("positive int for column to plot 1D")
	print("-1 to exit")
	print("-2 for multi-D mode")


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


	dimPlot = int( input("input Dim to plot : ") )
	while( dimPlot >= 0 ):

	    plt.plot(data[:,dimPlot])
	    plt.ylabel(r"$\|Pertubation\|_2$")
	    plt.xlabel("Iterations")
	    plt.savefig( 'Dim{}-Row.png'.format(dimPlot) )
	    plt.show()
	    plt.close()
	    dimPlot = int( input("input Dim to plot : ") )

	if(dimPlot == -2):

	    dimPlot1 = int( input("input Dim to plot : ") )
	    dimPlot2 = int( input("input Dim to plot : ") )

	if(ctr == 1):
		label = "True distribution"
	elif(ctr == 2):
		label = "Laplace approximation"
	plt.plot(data[:,dimPlot1],data[:,dimPlot2], label = label )
	plt.legend()

	plt.savefig('Dim{0} - Dim{1}.png'.format( dimPlot1, dimPlot2 ) )

	moreFiles = input("more files? true/false: ")

plt.show()
plt.close()

















#if(ndim == 99999 ):

    #data = np.array(data[:,:], dtype=[('x', float), ('y', float)])
    #data.sort(axis=0)

    #plt.scatter(data[:,0], data[:,1], s=5)
    #data[:,1] = data[:,1];
    #print(data)
    #Data = pd.DataFrame(data)
    #Data.drop( Data[Data[1] < -4.0e-05].index,inplace = True)
    #print(Data)
    #plt.scatter(Data[0], np.exp(Data[1].values) ** 10000 )
    #Data.to_csv('filterData.csv')
   # df.drop(df[df['Age'] < 25].index, inplace = True)
    #plt.scatter(data[:,0], data[:,1])
    #matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
    #vmin=None, vmax=None, alpha=None, linewidths=None,
    #faceted=True, verts=None, hold=None, **kwargs)
    #plt.title("Frequency of Displacment")
    #plt.savefig("FreqDisp.png")
    #plt.show()

