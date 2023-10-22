import matplotlib.pyplot as plt

# Definir las coordenadas x e y del cuadrado
x = [0, 1, 1, 0, 0]
y = [0, 0, 1, 1, 0]

# Graficar el cuadrado
plt.plot(x, y)

'''
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Posición y')
plt.title('Cuadrado')
'''

plt.grid(True)
plt.show()
