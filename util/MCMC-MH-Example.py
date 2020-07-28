#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:33:23 2020

@author: avxps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pdf(x):
    sig1 = 0.5
    sig2 = 1.5
    sig3 = 2.0
    
    lik = 1/(np.sqrt(2*np.pi)*sig1) * np.exp( -1/(2.*sig1**2)*(x - 0)**2 ) \
          + 1/(np.sqrt(2*np.pi)*sig2) * np.exp( -1/(2.*sig2**2)*(x - 3)**2 ) \
          + 1/(np.sqrt(2*np.pi)*sig3) * np.exp( -1/(2.*sig3**2)*(x - 1.5)**2 )
          
          
    return lik

x = np.linspace(-10,10,1000)
dx = (10 + 10) / 1000
y = pdf(x)
evidence = np.sum(y * dx)
#y = y / evidence

plt.plot(x,y,label='true posterior')
#plt.legend(loc='upper left')
plt.savefig('truePost.pdf')  
plt.show()
plt.close()

xCurr = 0
yCurr = pdf(xCurr)
samples = np.array([])
sigma = 2
for i in range(0, 100000):
    
    xProp = np.random.normal(xCurr, sigma, 1)
    yProp = pdf(xProp)
    
    r = min ( 1 , yProp / yCurr)
    u = np.random.uniform(0, 1)
    
    if r > u :
        xCurr = xProp
        yCurr = yProp
    
    samples = np.append(samples, xCurr)

samples = np.delete(samples, np.arange(0, samples.size, 2))
histogram = sns.distplot(samples, label='histogram')	
#plt.legend(loc='upper left')
plt.savefig('histogramPost-3.pdf')
plt.show()
plt.close()


plt.plot(samples)#,  label=' bad random walk')
#plt.legend(loc='upper left')
plt.savefig('randomWalk-3.pdf')
plt.show()
plt.close()

