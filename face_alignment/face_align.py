from retinaface import RetinaFace
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

path = 'D:/_data/MJ/'
file_name = 'aram.png'
img = cv2.imread(path + file_name)

resp = RetinaFace.detect_faces(img_path = path + file_name)
print(resp)

x1, y1 = resp["face_1"]['landmarks']['right_eye']
x2, y2 = resp["face_1"]['landmarks']['left_eye']

a = abs(y1 - y2)
b = abs(x2 - x1)
c = math.sqrt(a*a + b*b)

print(a, b, c)

cos_alpha = (b*b + c*c - a*a) / (2*b*c)
print(cos_alpha)

alpha = np.arccos(cos_alpha) # radius
alpha = -(alpha * 180) / math.pi
print(alpha)

aligned_img = Image.fromarray(img)
aligned_img = np.array(aligned_img.rotate(alpha))
aligned_img = aligned_img[:,:,::-1]
# print(aligned_img.shape) # (838, 834, 3)

im = Image.fromarray(aligned_img)
im.save(f'D:/_data/alignment/{file_name}_align.jpg')


# plt.imshow(aligned_img[:,:,::-1])
# plt.show()
