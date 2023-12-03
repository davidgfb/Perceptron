from numpy import array, round
from numpy.random import randint,rand,seed
#from numpy import exp
from math import exp
from matplotlib.pyplot import plot,show

sem = randint(100)
#sem = 75
seed(sem)

res, eps = 1920 * 1080, 0.1
(a_s, ws), (b, y) = (*(round(rand(res), 2) for i in range(2)),),\
                    round(rand(2), 2)

print('sem, b, y:', sem, b, y)

for i in range(20):
    #forprop
    z = a_s @ ws + b
    relu = max(0,z) 
    err = (relu-y)**2/2

    #backprop
    dc_db = (relu - y) * (0 if z < 0 else 1) #dying relu!!   
    b -= eps * dc_db

    for i in range(res): ws[i] -= eps * a_s[i] * dc_db

    print('\nz, relu, err, dc_db, b:', z, relu, err, dc_db, b)

#print('z, relu, err, dc_db, b:', z, relu, err, dc_db, b)

