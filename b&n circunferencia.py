from PIL import Image

s = 'ml/circunferencia/c4'

# Abre la imagen
img = Image.open(s + '.png')

# Convierte la imagen a blanco y negro
img = img.convert('L')

# Guarda la imagen en formato PNG
img.save(s + '_b&n.png')
