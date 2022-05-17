from Detector_all import *

detector = Detector_seg(model_type="IS") #object detection : model_type="OD"

# detector.onImage("images/1.jpg")

detector.onVideo("street.mp4")
