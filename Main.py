from random import random, seed
from numpy import array, round
from math import exp
from matplotlib.pyplot import plot,show

##[-1...1]=[-1,1]?
#sem = 0.97
sem = round(random(),2)

seed(sem)

(a,w,b,y), eps, errs, debug, adagrad =\
           (*(random() for i in range(4)),), 1e-1, [], 0, 0

print('sem, eps =',round(sem,2), eps)

if debug: print('a,w,b =', round(array((a,w,b)),2), '\neps =',eps)
#imp_rdado(w,b)

prim_vez = 1
for i in range(100):
    #forprop
    z = a*w+b
    sigm = 1/(1+exp(-z)) #act, y
    err = (sigm-y)**2/2
   
    if len(errs) > 0 and err > errs[-1]: break

    errs.append(err)
         
    #backprop
    dc_db = sigm * (1-sigm) * (a-y)
   
    b -= eps * dc_db
    w -= eps * a * dc_db
  
    if debug: print('\n',20*'-',i,20*'-',
                    '\n\nforprop:\n--------\nz, sigm, err =',
                    round(array((z, sigm, err)),2),
                    '\n\nbackprop:\n--------\nw,b,dc_db =',                 
                    round(array((w,b,dc_db)),2))

print('\n',40*'-','\n\ny, sigm =',round(array((y, sigm)),2),
      '\neps, i =', eps, i)

plot(errs, 'o-')
show()
