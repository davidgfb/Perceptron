import numpy as np
from matplotlib import pyplot as plt

# Definir el radio del círculo
R = 2

# Definir el ángulo de 0 a 2*pi
theta = np.linspace(0, 2*np.pi, 100)

# Calcular las coordenadas x y y del objeto en función del ángulo
x = R* np.cos(theta)
y = R * np.sin(theta)

# Graficar el círculo
plt.plot(x, y)

plt.axis('equal')
'''
#plt.xlabel('Posición x')
#plt.ylabel('Posición y')
#plt.title('Círculo')
'''

plt.grid(True)
plt.show()
