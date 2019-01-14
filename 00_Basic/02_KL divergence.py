# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:33:32 2019

@author: gist
"""


""" 
https://www.science-emergence.com/Articles
/Divergence-de-Kullback-Leibler-avec-python-et-matplotlib/
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

from scipy.stats import norm
from scipy.integrate import quad


def p(x):
    # p(x) ~ N(0,2)
    return norm.pdf(x,0,2)

def q(x):
    # q(x) ~ N(2,2)
    return norm.pdf(x,4,2)

# 두 확률분포간의 거리를 측정한것으로 서로 비슷할수록 값이 작게나옴
def KL(x):
    return p(x)* np.log( p(x) /q(x) )

def linear(x):
    return 2*x + 1 

scale = np.arange(-10,10,0.001)

KL_integrate,err = quad(KL,-10,10)
print('KL: ',KL_integrate)
#print(quad(linear,0,2))

fig = plt.figure(figsize=(18,10), dpi=72)

#-------------------- Fist plot

ax = fig.add_subplot(1,2,1)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(-10 , 10)       # x 범위 지정
ax.set_ylim(-0.1 , 0.25)    # y 범위 지정

ax.text(-2.5,0.17,'p(x)',horizontalalignment = 'center',
        fontsize=17,color='b')    
ax.text(4.5,0.17,'q(x)',horizontalalignment = 'center',
        fontsize=17,color='g')  


plt.plot(scale,p(scale))
plt.plot(scale,q(scale))

#ax.fill_between(scale,0,q(scale))

#------------------- Second plot
ax = fig.add_subplot(1,2,2)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(-10 , 10)       # x 범위 지정
ax.set_ylim(-1 , 1)    # y 범위 지정

ax.text(3.5, 0.17, r'$DK_{KL}(p||q)$', horizontalalignment='center',fontsize=17,color='b')

ax.plot(scale, KL(scale))

ax.fill_between(scale, 0, KL(scale))

plt.savefig('KullbackLeibler.png',bbox_inches='tight')
plt.show()


