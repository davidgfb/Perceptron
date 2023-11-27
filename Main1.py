#https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide
from math import exp
from numpy import array, round

a, w, b, y, eps, sig_r, r = 0.1, 0.5, 1.83, 0.03, 6, 0, int(1e3)
#eps valor dorado #sig rdada
 
for i in range(r):    
    sig = 1/(1+exp(-(a*w+b)))
    dMSE_dy = y-sig
    gradMSE_dy = eps * dMSE_dy
    b += gradMSE_dy  
    w += a * gradMSE_dy * (1 + sig * (1 - sig))

    sig_r = round(sig,2)
    if sig_r == y: break
                
print('eps:', eps, 'N:', i, 'sig:', sig_r, 'debe ser eps: 6 N: 4 sig: 0.03') 

'''TODO: 
q el programa determine el valor dorado
c = y_msig**2/2 #repr error en matplotlib
'''
