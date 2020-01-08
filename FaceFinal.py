
from __future__ import print_function

import numpy as np
import cv2 as cv
import time

#local modules
from video import create_capture
from common import clock, draw_str

#Define Function for face detection
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


#detectMultiScale: general function to detect objects. Parameters:
#img: it takes the image of the whole webcam
#scaleFactor: specifies how much the image size is reduced at each image scale
#minNeighbors: specifies how many neighbors the current object should have before it declares the face found
#minSize: size of each moving window

# Draw rectangles
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        # Diagonal coordinates
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
# Displaying online Histogram bar
def show_hist(hist):

    bin_count = hist.shape[0] # Defining the size of the hist equal to hist size
    bin_w = 24 #the width of the bin

    # Creating empty image to show there our hist, (matrix of zeros of type int)
    img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
    # Loop for siplaying multiple bins which contain the color values,
    for i in range(bin_count):

        h = int(hist[i])


        cv.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h), (int(180.0 * i / bin_count), 255, 255),
                     -1)
    # Converting from HSV to BGR color space
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    cv.imshow('hist', img)


if __name__ == '__main__':

    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade='])

# Loading pre-trained model
    model = cv.ml.ANN_MLP_create()
    model = model.load('model_letter.txt')

    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    # Initializing haar cascades, to find a face
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))

    # Taking capture of the video stream
    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))

    # Defining anf initializing back projection
    rects = []
    show_backproj = False

    #Reading continuous data coming from the camera and analysing each frame
    while True:
        #Reading image from the camera
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Transferring to graspace ti calculate hist and to easier work
        gray = cv.equalizeHist(gray)  # Equalize the histogram of the captured image

        # Making a copy of the orihinal image
        vis = img.copy()
        # creating a copy of original pic but in HSV space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        t = clock()

        # rects[] are empty means no face has been detected yet
        if len(rects) == 0: #facedetect
            # So run the detection
            rects = detect(gray, cascade)
            # drawing rectangles on detected face
            draw_rects(vis, rects, (0, 255, 0))

        else:
            # If face has been detected, then the diagonal coordinates are saved in the rects[]
            [[x0, y0, x1, y1]] = rects

            # Now as we found area of the interest, we knw the coordinates of the face and can convert to hsv and mask that area
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]

            # Calculate histogram of the area of interest
            # cv2.calcHist(images, channels , mask, histSize, ranges[, hist[, accumulate]])
            # channels = 0 for grayscale, calculating hist on particular image so we need a mask image
            # Hist size is 16, range is from 0 to 180
            hist = cv.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])

            # Normilize from 0 to 256
            cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            # Transpose of histogram matrix
            hist = hist.reshape(-1)
            # Calling function to display the histogram
            show_hist(hist)

            # Initializing the area of tracking
            # Calculating from diag points, width and height
            track_window = (x0, y0, x1 - x0, y1 - y0)

            # Calculating the back proj of a hist (images, chanels, hist, ranges, scale[,dist})
            # obtain probability of each pixel of that region belonging to a face or not (depending on color)
            prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)

            #prob = prob & mask
            prob &= mask

            term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1) #termination criteria for camshift

            # Running camshift (first time)
            track_box, track_window = cv.CamShift(prob, track_window, term_crit)

             # Getting the coordinates of the face region and a little enalrging the box
            ((x3, y3), (w, z), k) = track_box
            x0 = int(x3 - w / 2)
            x1 = int(x3 + w / 2)
            y0 = int(y3 - z / 2)
            y1 = int(y3 + z / 2)

            # Then need to set probability for face to 0 and call  camshift seond time to track hand
            prob[y0:y1, x0:x1] = 0

            #Setting new track window
            track_window2 = (0, 0, vis.shape[0], vis.shape[1])

            # Calling again camshift
            track_box2, track_window = cv.CamShift(prob, track_window2, term_crit)

            # Obtaining the value of the new region, where hand is supposed to be.
            ((xx0, yy0), (ww, zz), k2) = track_box2
            xx1 = int(xx0 - ww / 2)
            xx2 = int(xx0 + ww / 2)
            yy1 = int(yy0 - zz / 2)
            yy2 = int(yy0 + zz / 2)

            if show_backproj:
                vis[:] = prob[..., np.newaxis]

            try:
                cv.ellipse(vis, track_box2, (0, 0, 255), 2)  # Ellipse if the hand is detected
            except:

                print(track_box)

        cv.imshow('facedetect-camshift', vis)

        # For input from keyboard, initialization
        ch = cv.waitKey(5)
        # Press esc in order to exit
        if ch == 27:
            break
        # Press "b" to show the back projection
        if ch == ord('b'):
            show_backproj = not show_backproj
        # Press "s" to capture the hand
        if ch == ord('s'):
            # If the hand was detected
            if track_box2:

                hand = prob[yy1:yy2, xx1:xx2]
                # Resizing to get 16 by 16 img in order to be comaptible with the training set images, also convert to float32
                hand1 = cv.resize(hand, (16, 16)).astype(np.float32)
                # Need to convert the 2D array into 1D to be fed into classification model and to take the transpose, as flatten returns one row, need column vector

                new=hand1.flatten().reshape((1,-1))

                # Calling the classifier, MLP and training on our model. Taking only the first value of the opredicted output. It outputs us the int, which
                # stands for the place of the letter in the alphabet. Need to convert it into the char and with ord("A") we define the initial place.
                prediction = model.predict(new)[0]
                print(prediction)
                print(chr(ord("A")+int(prediction)))

                #t = time.time()
                # In order to save the original image of the hands in the training set
                #cv.imwrite('hand' + str(t) + '.jpg', hand1)




cv.destroyAllWindows()
