from PIL.Image import open
from matplotlib.pyplot import imread, imshow, show
from numpy import array, dot, reshape

bias = 0
#es_Circunferencia = True

'''
El clasificador compara la imagen de activacion con la imagen
peso. debe haber varias imagenes peso.
Si el clasficador no encuentra parecido entre la imagen
activacion y la imagen peso el usuario decide
si añadir esa nueva imagen de activacion como imagen peso

TODO: Hay problema con el rango de valores. Entrada: 0...1 check,
pesos: -7...7? Bias: 4...9? salida: 0, 1, imgPesos: -6...6?
creo q no hay problema en q el rango de la img pesos sea [-2...2]
[-2*255...2*255]
cuela un cuadrado al final y ajusta bias

m_Fila_Img_Ent, m_Fila_Img_Pesos = (array(imread(ruta)).flatten()
    for ruta in ('ml/circunferencia/c_b&n.png', 'pesos.png'))
'''

m_Fila_Img_Pesos = array(imread('pesos.png')).flatten()

rutas = (*('ml/circunferencia/c' + str(i) + '_b&n.png'
         for i in range(5)),)

#print(rutas)

def check():
    pass

for ruta in rutas:
    m_Fila_Img_Ent = array(imread(ruta)).flatten()
    
    salida = m_Fila_Img_Ent @ m_Fila_Img_Pesos

    print(salida)

    '''
    #print(salida) #298895.5
    #print(not(salida > bias == es_Circunferencia), True)
    #print(not(True, False)) #wtf!?

    la salida es distinta a/no coincide con la realidad.
    todavia no exporta png pesos
    '''
    if salida <= bias: # = es_Circunferencia:
        m_Fila_Img_Pesos += m_Fila_Img_Ent #-=
        
        exportacion = reshape(m_Fila_Img_Pesos,
            open(ruta).size[::-1])

        '''
        imshow(exportacion)
        show()
        '''
        
        print('añado nueva etiqueta') #, m_Fila_Img_Pesos)

    else:
        print('imagen reconocida!')
