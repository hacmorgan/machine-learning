#!/usr/bin/env python

import cv2
import numpy as np
import csv
import time

start_time = time.time()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

with open("../training_data/webcam.csv", "a+") as datafile:
    writer = csv.writer(datafile)
    while True:
        ret, frame = cap.read()
        cap_time = time.time()
        if ret == False:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        downsampled = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        downsampled_list = downsampled.reshape((64*48)).tolist()
        writer.writerow(downsampled_list)
        print("Image saved at " + str(cap_time))
        # LPs can't exceed 90 minutes, but we still want some data around the centre
        if cap_time - start_time > 92*60:
            break
        time.sleep(10)
        
cap.release()
cv2.destroyAllWindows()
