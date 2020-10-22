import numpy as np
import cv2 as cv
from picamera.array import PiRGBArray
from time import sleep

def get_area(mask):

    # Morphological closing to get whole particles; opening to get rid of noise
    img_mop = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
    img_mop = cv.morphologyEx(img_mop, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
    # Find contours
    _, cnts, _ = cv.findContours(img_mop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Get bounding rectangles for the scale and the particles
    if len(cnts) > 0:  
        areas = []
        for cnt in cnts:
            areas.append(cv.contourArea(cnt))
        return max(areas)
    else: # No blobs were detected.
        return 0

def return_orientation(IMG):
    # blur images. If slow performance, can probably get away with remove some blurs
    blur = cv.blur(IMG,(5,5))
    blur = cv.medianBlur(blur,5)
    blur= cv.GaussianBlur(blur,(5,5),0)
    
    # convert to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    ####### CREATE COLOUR MASKS ##########
    
    # blue
    low_blue = np.array([80, 100, 50])
    high_blue = np.array([150, 255, 250])
    blue_mask = cv.inRange(hsv, low_blue, high_blue)
    erode_kernel = np.ones((11,11), np.uint8) 
    blue_mask = cv.erode(blue_mask, erode_kernel, iterations=3)
    dilate_kernel = np.ones((5,5), np.uint8) 
    blue_mask = cv.dilate(blue_mask, dilate_kernel, iterations=1)
    
    # green
    low_grn = np.array([50, 30, 30])
    high_grn = np.array([85, 150, 180])
    grn_mask = cv.inRange(hsv, low_grn, high_grn)
    erode_kernel = np.ones((11,11), np.uint8) 
    grn_mask = cv.erode(grn_mask, erode_kernel, iterations=3)
    # We dilated a lot because there is a lot of green noise
    dilate_kernel = np.ones((5,5), np.uint8) 
    grn_mask = cv.dilate(grn_mask, dilate_kernel, iterations=5)
    
    # yellow
    low_yel = np.array([15, 100, 120]) 
    high_yel = np.array([50, 255, 255])
    yel_mask = cv.inRange(hsv, low_yel, high_yel)
    erode_kernel = np.ones((3,3), np.uint8) 
    yel_mask = cv.erode(yel_mask, erode_kernel, iterations=1)
    dilate_kernel = np.ones((3,3), np.uint8) 
    yel_mask = cv.dilate(yel_mask, dilate_kernel, iterations=1)
    
    # red
    # Red exists at both ends of the color spectrum.
    low_red_one = np.array([0, 150, 50])
    high_red_one = np.array([10, 255, 255])
    red_mask_one = cv.inRange(hsv, low_red_one, high_red_one)

    low_red_two = np.array([150, 150, 50])
    high_red_two = np.array([255, 255, 255])
    red_mask_two = cv.inRange(hsv, low_red_two, high_red_two)

    red_mask = red_mask_one + red_mask_two
    erode_kernel = np.ones((5,5), np.uint8) 
    red_mask = cv.erode(red_mask, erode_kernel, iterations=2)
    dilate_kernel = np.ones((5,5), np.uint8) 
    red_mask = cv.dilate(red_mask, dilate_kernel, iterations=2)

    ########### End creation of colour masks ##########

    masks = [red_mask, grn_mask, blue_mask, yel_mask]
    
    biggest_area = 0
    biggest_mask = 0
    second_biggest_area = 0
    second_biggest_mask = 0

    for i,mask in enumerate(masks):
        current_area = get_area(mask)
        if current_area > biggest_area:
            second_biggest_area = biggest_area
            second_biggest_mask = biggest_mask
            biggest_area = current_area
            biggest_mask = i+1
        elif current_area > second_biggest_area:
            second_biggest_area = current_area
            second_biggest_mask = i+1

    # 0 = none, 1 = red, 2 = green, 3 = blue, 4 = yellow
    if biggest_area < second_biggest_area*1.2: # arbitary threshold to see if two targets are visible.
        if set([biggest_mask, second_biggest_mask]) == set([1,4]):
            return 4.5
        else:
            return (biggest_mask + second_biggest_mask)/2
    else: 
        return biggest_mask

def sense_target(camera):
    # initialize the camera and grab a reference to the raw camera capture
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    sleep(0.1)
    # grab an image from the camera.
    # we don't actually need continuous capture if we are just doing a single sensory reading
    camera.capture(rawCapture, format="bgr")
    IMG = rawCapture.array
    return return_orientation(IMG)