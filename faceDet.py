#!/usr/bin/env python

'''
face detection using haar cascades
USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# local modules
from video import create_capture

def calculateFaceRatioFromRotatedRectangle(points):
    left_most_point = sorted(points, key=lambda p: p[0])[0]
    right_most_point = sorted(points, key=lambda p: p[0], reverse=True)[0]
    top_most_point = sorted(points, key=lambda p: p[1])[0]
    bottom_most_point = sorted(points, key=lambda p: p[1], reverse=True)[0]

    ab = math.sqrt(math.pow(left_most_point[0] - top_most_point[0], 2) + math.pow(left_most_point[1] - top_most_point[1], 2))
    bc = math.sqrt(math.pow(right_most_point[0] - top_most_point[0], 2) + math.pow(right_most_point[1] - top_most_point[1], 2))

    if ab == 0 or bc == 0:
        return 0

    return min(ab/bc, bc/ab)

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "opencv/data/haarcascades/haarcascade_frontalface_alt.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('opencv/samples/data/lena.jpg')))

    first_time_face_tracked = True
    original_face_ratio = 0

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))

        if len(rects) > 0:
            # Found at least one face -> Track first face that is found using CAMShift
            # Determine new face position with CAMShift
            # First calculate HSV histogram of region of face we detected with HAAR cascades
            (face_x, face_y, face_x2, face_y2) = tuple(rects[0])

            track_window = (face_x, face_y, face_x2-face_x, face_y2-face_y)

            roi = vis[face_y:face_y2, face_x:face_x2]
            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

            hsv_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
            cv.normalize(hsv_hist, hsv_hist, 0, 255, cv.NORM_MINMAX)

            term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
            while True:
                ret, img = cam.read()
                width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

                hsv_screen = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv_screen], [0], hsv_hist, [0, 180], 1)

                th1 = cv.GaussianBlur(dst,(5,5),0)
                ret1,th1 = cv.threshold(th1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

                track_box, track_window = cv.CamShift(dst, track_window, term_crit)
                pts = cv.boxPoints(track_box)
                pts = np.int0(pts)

                if first_time_face_tracked:
                    original_face_ratio = calculateFaceRatioFromRotatedRectangle(pts)
                    first_time_face_tracked = False

                # Check if ratio of face width and height changes much from original, then probably CAMShift needs updated
                # probability map (run facedetect with HAARs again)
                current_face_ratio = calculateFaceRatioFromRotatedRectangle(pts)

                if current_face_ratio < original_face_ratio-.2 or current_face_ratio > original_face_ratio+.2:
                    # Strong anomaly, probably background changed and old face histogram is no longer a good
                    # representation of the face.
                    print("old: " + str(original_face_ratio) + " new: " + str(current_face_ratio))
                    first_time_face_tracked = True
                    break

                vis = cv.polylines(img, [pts], True, (0,0,255), 5)

                # Try to find the hand from the probability map
                th1 = cv.fillPoly(th1, [pts], (0, 0, 0))
                th1 = cv.blur(th1, (10, 10))
                ret, th1 = cv.threshold(th1, 50, 255, cv.THRESH_BINARY)
                cv.imshow('test', dst)

                contours, hierarchy = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    biggest_contour = max(contours, key=cv.contourArea)

                    x2,y2,w2,h2 = cv.boundingRect(biggest_contour)
                    vis = cv.rectangle(vis, (x2,y2), (x2+w2, y2+h2), (0,255,255), 5)

                cv.imshow('facedetect', vis)

                if cv.waitKey(5) == 27:
                    break

        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()