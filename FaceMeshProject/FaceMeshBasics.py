import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh, drawing_utils, face_mesh_connections
import time


cap = cv2.VideoCapture("videos/trump.mp4")
# cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = drawing_utils
mpFaceMesh = face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1) # 점의 두께, 원의 지름 조정

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
            drawSpec, drawSpec)
            for lm in faceLms.landmark:
                # print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                # print(id, x, y)


    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3) # 동영상 fps 표기
    cv2.imshow('image', img)
    cv2.waitKey(1)