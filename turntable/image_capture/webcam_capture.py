#!/usr/bin/env python

import cv2
import numpy as np
import csv
import time
import sys
import argparse
import signal

def signal_handler(sig, frame):
    global datafile
    if datafile is not None:
        datafile.close()
    sys.exit(0)

def get_args():    
    parser = argparse.ArgumentParser(description='Collect training data from the webcam')
    parser.add_argument('--outfile', '-o', type=str, help='file to write the data to')
    return parser.parse_args()


def main(args):
    # Handle SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    start_time = time.time()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    global datafile
    datafile = open(args.outfile, "a+")

    writer = csv.writer(datafile)
    while True:
        ret, frame = cap.read()
        cap_time = time.time()
        if ret == False:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        downsampled = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        downsampled_list = downsampled.reshape((64*48)).tolist()
        downsampled_list.append(0)
        writer.writerow(downsampled_list)
        print("Image saved at " + str(cap_time))
        # LPs can't exceed 90 minutes, but we still want some data around the centre
        if cap_time - start_time > 92*60:
            break
        time.sleep(10)

    datafile.close()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(get_args()))
