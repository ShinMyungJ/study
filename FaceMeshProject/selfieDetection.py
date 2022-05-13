from sre_constants import SUCCESS
import cv2
from mediapipe.python.solutions import face_mesh, drawing_utils, face_mesh_connections, selfie_segmentation
import numpy as np
import time

mpDraw = drawing_utils
mp_selfie_segmentation = selfie_segmentation

BG_COLOR = (192, 192, 192)  # grey

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/2.mp4")

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    bg_image = None
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            
        start = time.time()
            
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        
        # Pass the image through the model
        results = selfie_segmentation.process(image)
        cv2.imshow('Segmentation Mask', results.segmentation_mask)
        
        image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw_selfie_segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.15
        
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image, (55,55), 0)
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            
            # a)
            
            #bg_image = cv2.imread('office.jpg')
            #bg_image = cv2.imread('beach.jpg')
            #bg_image = cv2.resize(bg_image, (640, 480))
            
            # b)
            bg_image = cv2.GaussianBlur(image, (55,55), 0)
            
        output_image = np.where(condition, image, bg_image)
        
        end = time.time()
        totalTime = end - start
        
        fps = 1 / totalTime
        print("FPS: ", fps)
        
        cv2.putText(output_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
    