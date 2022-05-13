import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import glob
import random

def brightness(gray, val):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = int(random.uniform(-val, val))
    if brightness > 0:
        gray = gray + brightness
    else:
        gray = gray - brightness
    gray = np.clip(gray, 10, 255)
    return gray

def contrast(gray, min_val, max_val):
    #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    alpha = int(random.uniform(min_val, max_val)) # Contrast control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
    return adjusted

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def vertical_shift_down(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def vertical_shift_up(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(0.0, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

images = sorted(glob.glob('dataset/train/11/*.png'))
i = 0

print(len(images))

 dir = 'dataset/train/11/' 

  for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 30)
    img = contrast(img, 1, 1.5)
    img = horizontal_flip(img, 1)
    img = rotation(img, 180)
    img = horizontal_shift(img, 0.1)
    #if random.uniform(0,1) > 0.5:
    #    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 10000:
        break

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 10)
    img = contrast(img, 1, 1.2)
    img = horizontal_flip(img, 1)
    img = rotation(img, 180)
    img = horizontal_shift(img, 0.2)
    #if random.uniform(0,1) > 0.5:
    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 10000:
        break

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 20)
    img = contrast(img, 1, 1.3)
    img = horizontal_flip(img, 1)
    img = rotation(img, 90)
    img = horizontal_shift(img, 0.3)
    #if random.uniform(0,1) > 0.5:
    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 10000:
        break

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 20)
    img = contrast(img, 1, 1.3)
    img = horizontal_flip(img, 1)
    img = rotation(img, 90)
    img = horizontal_shift(img, 0.3)
    #if random.uniform(0,1) > 0.5:
    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 10000:
        break


for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 20)
    img = contrast(img, 1, 1.5)
    img = horizontal_flip(img, 1)
    img = rotation(img, 180)
    img = horizontal_shift(img, 0.1)
    #if random.uniform(0,1) > 0.5:
    #    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 9500:
        break

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 10)
    img = contrast(img, 1, 1.5)
    img = horizontal_flip(img, 1)
    img = rotation(img, 180)
    img = horizontal_shift(img, 0.1)
    #if random.uniform(0,1) > 0.5:
    #    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 9500:
        break

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
    img = brightness(img, 30)
    img = contrast(img, 1, 1.4)
    img = horizontal_flip(img, 1)
    img = rotation(img, 180)
    img = horizontal_shift(img, 0.3)
    #if random.uniform(0,1) > 0.5:
    img = vertical_flip(img, 1)
    file_name = dir + str(i) + '.png'
    #file_name = 'aug_image/' + str(i) + '.png'
    cv2.imwrite(file_name, img)
    i = i + 1
    if i > 9500:
        break


# 출처: https://hagler.tistory.com/189 [Hagler's Blog]

