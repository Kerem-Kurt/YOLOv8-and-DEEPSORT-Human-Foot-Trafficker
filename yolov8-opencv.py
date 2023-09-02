# https://youtu.be/QMBMWvn9DJc
#
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

def RGB(event, x,y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x,y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture(0) #'vidyolov8.mp4'  'rtsp://data_analytic:TcAnTaRa9721881@10.158.8.19'

count=0
while True:
    ret,frame = cap.read()
#    frame=cv2.resize(frame,(640,480))
    if not ret:
        break
    count += 1
    if count % 3 != 0: #Inference every 3 frames
        continue
    start_time = time.time()
    results=model.predict(frame)
    # a=results[0].boxes.boxes                   # For machine no GPU only
    a=torch.Tensor.cpu(results[0].boxes.boxes)   # For machine has GPU only such as Jeson Nano
    px=pd.DataFrame(a).astype("float")
    for index,row in px.iterrows():
        # print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

exit()

