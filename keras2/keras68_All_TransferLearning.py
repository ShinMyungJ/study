
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

# model = VGG16()
# model = VGG19()
# model = ResNet50()
# model = ResNet50V2()
# model = ResNet101()
# model = ResNet101V2()
model = ResNet152()
# model = ResNet152V2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = MobileNetV3Small()
# model = MobileNetV3Large()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
# model = EfficientNetB1()
# model = EfficientNetB7()
# model = Xception()
model.trainable = False
model.summary()

print("============================================================")
print("모델명 : ", model.name)
print("전체 가중치 갯수 : ", len(model.weights))
print("훈련 가능 가중치 갯수 : ", len(model.trainable_weights))

# 모델명 : vgg16
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 전체 가중치 갯수 :  32
# 훈련 가능 가중치 갯수 :  0

# 모델명 :  vgg19
# Total params: 143,667,240
# Trainable params: 0
# Non-trainable params: 143,667,240
# 전체 가중치 갯수 :  38
# 훈련 가능 가중치 갯수 :  0

# 모델명 :  resnet50
# 전체 가중치 갯수 :  320
# 훈련 가능 가중치 갯수 :  0
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712

# 모델명 :  resnet50v2
# 전체 가중치 갯수 :  272
# 훈련 가능 가중치 갯수 :  0
# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800

# 모델명 :  resnet101
# 전체 가중치 갯수 :  626
# 훈련 가능 가중치 갯수 :  0
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176

# 모델명 :  resnet101v2
# 전체 가중치 갯수 :  544
# 훈련 가능 가중치 갯수 :  0
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560




