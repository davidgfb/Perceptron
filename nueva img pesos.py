from PIL import Image

# Crea una nueva imagen de 640x480 píxeles
img = Image.new('L', (640, 480))

# Guarda la imagen en formato PNG
img.save('pesos.png')


'''
from PIL import Image

# Abre la imagen
img = Image.open('ruta/a/tu/imagen.png')

# Convierte la imagen a blanco y negro
img = img.convert('L')

# Guarda la imagen en formato PNG
img.save('ruta/a/tu/imagen_bn.png', 'PNG')
'''

