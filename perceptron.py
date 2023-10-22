from matplotlib.pyplot import imread, imshow, show
from numpy import array, dot

bias = 0

'''
 el clasificador compara la imagen de activacion con la imagen
 peso. debe haber varias imagenes peso.
 si el clasficador no encuentra parecido entre la imagen
 activacion y la imagen peso el usuario decide
 si añadir esa nueva imagen de activacion como imagen peso
'''

img_Act = imread('ml/circunferencia/c_b&n.png')
img_Pesos = imread('pesos.png')

print(array(img_Act).flatten() @ array(img_Pesos).flatten()> bias)
