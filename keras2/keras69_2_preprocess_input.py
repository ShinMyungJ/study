from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = '../_data/image/cat_dog/cat_dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print("============================image.img_to_array(img)==========================")
print(x, '\n', x.shape)     #  (224, 224, 3)

x = np.expand_dims(x, axis=0)
print("============================image.img_to_array(img)==========================")
print(x, '\n', x.shape)     #  (1, 224, 224, 3)

x = preprocess_input(x)
print("============================ preprocess_input ==========================")
print(x, '\n', x.shape)     #  (1, 224, 224, 3)

preds = model.predict(x)
print(preds, '\n', preds.shape)     #  (1, 1000)

print('결과는 : ', decode_predictions(preds, top=10)[0])