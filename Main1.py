#https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide
from math import exp, pi 
from numpy import array, round
from matplotlib.pyplot import xlabel, ylabel, plot, show
from random import random, seed

a, w, b, y, eps, sig_r, r, MSEs, prec_decs, tipo_Aprox, semilla =\
   random(), random(), random(), pi/10, 6, 0, int(1e3), [], 16, 0,\
   round(random(),2) #PREcond 0 < y < 1; pi/10 -> 16 decimales

seed(semilla) #para fijar los valores aleatorios

for tipo in range(2): 
    for i in range(r):
        z = a*w+b
        sig = 1/(1+exp(-z))
        gradMSE_dy = eps * (y-sig) #dMSE_dy
        b += gradMSE_dy  
        w += a * gradMSE_dy * (1 + sig * (1 - sig))

        MSE = (sig-y)**2/2
        MSE_rdado = round(MSE,prec_decs)
        MSEs.append(MSE_rdado)
            
        sig_rdada = round(sig,prec_decs)    
        cond = sig_rdada == y    
        
        if tipo_Aprox == 1: cond = MSE_rdado == 0
        
        if cond: break

    print('semilla =', semilla, ', y = pi/10 =', round(y,prec_decs),
          '?= sigm(z)\n z =', round(z,prec_decs),
          'debe ser -0.7807453636875454')

    plot(MSEs, 'o-')
    xlabel('N =' + str(i))
    ylabel('tipo =' + str(tipo))
    show()

#print('eps:', eps, 'N:', i, 'sig:', sig_r, 'debe ser eps: 6 N: 4 sig: 0.03') 

'''TODO: 
q el programa determine el valor dorado
c = y_msig**2/2 #repr error en matplotlib
'''
