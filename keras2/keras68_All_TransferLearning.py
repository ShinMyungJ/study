
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
# model = ResNet152()
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
model = Xception()
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

# 모델명 :  resnet152
# 전체 가중치 갯수 :  932
# 훈련 가능 가중치 갯수 :  0
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944

# 모델명 :  resnet152v2
# 전체 가중치 갯수 :  816
# 훈련 가능 가중치 갯수 :  0
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648

# 모델명 :  densenet121
# 전체 가중치 갯수 :  606
# 훈련 가능 가중치 갯수 :  0
# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504

# 모델명 :  densenet169
# 전체 가중치 갯수 :  846
# 훈련 가능 가중치 갯수 :  0
# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880

# 모델명 :  densenet201
# 전체 가중치 갯수 :  1006
# 훈련 가능 가중치 갯수 :  0
# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984

# 모델명 :  inception_v3
# 전체 가중치 갯수 :  378
# 훈련 가능 가중치 갯수 :  0
# Total params: 23,851,784
# Trainable params: 0
# Non-trainable params: 23,851,784

# 모델명 :  inception_resnet_v2
# 전체 가중치 갯수 :  898
# 훈련 가능 가중치 갯수 :  0
# Total params: 55,873,736
# Trainable params: 0
# Non-trainable params: 55,873,736

# 모델명 :  mobilenet
# 전체 가중치 갯수 :  137
# 훈련 가능 가중치 갯수 :  0
# Total params: 4,253,864
# Trainable params: 0
# Non-trainable params: 4,253,864

# 모델명 :  mobilenetv2
# 전체 가중치 갯수 :  262
# 훈련 가능 가중치 갯수 :  0
# Total params: 3,538,984
# Trainable params: 0
# Non-trainable params: 3,538,984

# 모델명 :  MobilenetV3small
# 전체 가중치 갯수 :  210
# 훈련 가능 가중치 갯수 :  0
# Total params: 2,554,968
# Trainable params: 0
# Non-trainable params: 2,554,968

# 모델명 :  MobilenetV3large
# 전체 가중치 갯수 :  266
# 훈련 가능 가중치 갯수 :  0
# Total params: 5,507,432
# Trainable params: 0
# Non-trainable params: 5,507,432

# 모델명 :  NASNet
# 전체 가중치 갯수 :  1546
# 훈련 가능 가중치 갯수 :  0
# Total params: 88,949,818
# Trainable params: 0
# Non-trainable params: 88,949,818

# 모델명 :  NASNet
# 전체 가중치 갯수 :  1126
# 훈련 가능 가중치 갯수 :  0
# Total params: 5,326,716
# Trainable params: 0
# Non-trainable params: 5,326,716

# 모델명 :  efficientnetb0
# 전체 가중치 갯수 :  314
# 훈련 가능 가중치 갯수 :  0
# Total params: 5,330,571
# Trainable params: 0
# Non-trainable params: 5,330,571

# 모델명 :  efficientnetb1
# 전체 가중치 갯수 :  442
# 훈련 가능 가중치 갯수 :  0
# Total params: 7,856,239
# Trainable params: 0
# Non-trainable params: 7,856,239

# 모델명 :  efficientnetb7
# 전체 가중치 갯수 :  1040
# 훈련 가능 가중치 갯수 :  0
# Total params: 66,658,687
# Trainable params: 0
# Non-trainable params: 66,658,687




