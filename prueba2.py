import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
for f in ['zidane.jpg', 'bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
im0 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Argentina_celebrando_copa_%28cropped%29.jpg/153px-Argentina_celebrando_copa_%28cropped%29.jpg'
im1 = Image.open('zidane.jpg')  # PIL image
im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [im0, im1, im2]  # batch of images

# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()  
results.save()  # or .show()

print('VER DATOS')
for n in range(3):
    print('***************************************************')
    print(f'Datos de la imagen {n}')
    print(results.xyxy[n])  # im1 predictions (tensor)
    print(results.pandas().xyxy[n])  # im1 predictions (pandas)
