# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:28:37 2019

@author: wook

"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def Binomial():
    print('Bindomial distribution')
    
    """ E(x) = np, Var(x) = np(1-p) """
    
    n = 10
    p = 0.3 # head
    k = np.arange(0,21)
    print(k)
    
    binomial = stats.binom.pmf(k,n,p)
    print(binomial)
    
    plt.plot(k,binomial,'o-')
    plt.title('Binomial: n=%i , p=%.2f' % (n,p),fontsize=15)
    plt.xlabel('Number of success')
    plt.ylabel('Probabilty of successes')
    plt.show()
    
def Gaussian():
    
    """ p(x) ~ N(0 , 0.1^2) """
    
    mu = 0      # mean
    sigma = 1 # standard deviation
    
    # np.random.normal 함수를 이용해서 평균 0
    # 표준 편차가 0.1 인 sample들을 1000개 추출한다.
    
    sample = np.random.normal(mu,sigma,size=100)
    print(sample)
    
    # sample들의 histogram을 출력한다.
    count, bins ,ignored = plt.hist(sample,30,density=True)
    
    # sample들을 이요애서 Gaussian Distribution의 shape를 재구축해서 line으로 그린다.
    plt.plot(bins,1/(sigma*np.sqrt(2*np.pi))*
             np.exp( - (bins-mu)**2 / (2 * sigma**2) ),linewidth=2,color='r')
    plt.show()
    
    # 코드를 실행하면 sampling 결과로 아래와 같이 종모양으로 표본들(samples) 이 추출되는것을 
    # 볼수 있음. random sampling을 이용해서 표본을 추출했기 때문에 재구축할때마다 매번 달라짐
    
def main():
    Gaussian()



if __name__=='__main__':
    main()