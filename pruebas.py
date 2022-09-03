import torch

print('Cargando el modelo')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

def check_image(img):
    """ Revisar una URL de imagen pasada en "img" como string """
    print('Inferir resultados')
    results = model(img)

    # Imprimir resultados
    results.print()
    return results


# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# check_image(img)
# """
# image 1/1: 720x1280 2 persons, 2 ties
# Speed: 871.6ms pre-process, 52.2ms inference, 1.3ms NMS per image at shape (1, 3, 384, 640)
# """

# check_image('https://img.freepik.com/vector-gratis/avatares-personas-felices_52683-34515.jpg?w=2000')
# """
# image 1/1: 1333x2000 1 person, 1 kite
# Speed: 262.6ms pre-process, 61.4ms inference, 1.4ms NMS per image at shape (1, 3, 448, 640)
# """

img = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Argentina_celebrando_copa_%28cropped%29.jpg/153px-Argentina_celebrando_copa_%28cropped%29.jpg'
res = check_image(img)
"""
image 1/1: 240x153 3 persons, 1 sports ball
Speed: 1323.4ms pre-process, 57.1ms inference, 0.5ms NMS per image at shape (1, 3, 640, 416)
"""
res.show()
pd = res.pandas()

print(f'pandas={pd}')
