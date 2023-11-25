from math import exp

def d_F_Error(predi, obj): #prediccion - objetivo
    return predi - obj

def z(a, w, b):
    return a * w + b

def f_Error(predi, obj): 
    return (d_F_Error(predi, obj)) ** 2 / 2

def f_Act(a, w, b, tipo = 0): #normaliza valores en [0, 1]
    _Z = z(a, w, b)
    f_Act = 1 / (1 + exp(-_Z)) #sigmoide

    if tipo == 1: f_Act = max(0, _Z) #ReLU

    return f_Act

def d_F_Act(a, w, b, tipo = 0): 
    sigm = f_Act(a, w, b)
    _f_Act = sigm * (1 - sigm) #d_Sigmoide

    if tipo == 1: #d_ReLU
        _f_Act = 0

        if z(a, w, b) > 0: _f_Act = 1 

    return _f_Act

'''
#PROBADOR
print(round(d_F_Act(1,2,3), 2), 'debe ser 0.01\n',
      d_F_Act(1,2,-3, 1), 'debe ser 0\n', d_F_Act(1,2,3, 1),
      'debe ser 1\n', f_Error(1,2), 'debe ser 0.5\n', f_Error(3,1),
      'debe ser 2\n', round(f_Act(1,2,3),2), 'debe ser 0.99\n',
      f_Act(1,2,3, 1), 'debe ser 5')
'''
