#!/usr/bin/python3
import os
# initialize asebamedulla in background and wait 0.3s to let
# asebamedulla startup
os.system("(asebamedulla ser:name=Thymio-II &) && sleep 0.3")
import matplotlib.pyplot as plt
from time import sleep
import dbus
import dbus.mainloop.glib
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2 as cv
import numpy as np

############################ IMAGE PROCESSING FUNCTIONS #########################
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
    low_blue = np.array([70, 70, 50])
    high_blue = np.array([150, 255, 200])
    blue_mask = cv.inRange(hsv, low_blue, high_blue)
    erode_kernel = np.ones((11,11), np.uint8) 
    blue_mask = cv.erode(blue_mask, erode_kernel, iterations=4)
    dilate_kernel = np.ones((5,5), np.uint8) 
    blue_mask = cv.dilate(blue_mask, dilate_kernel, iterations=1)
    
    # green
    low_grn = np.array([50, 30, 30])
    high_grn = np.array([80, 150, 180])
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
        return biggest_mask, second_biggest_mask
    else: 
        return biggest_mask

def sense_target():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    sleep(0.1)
    # grab an image from the camera.
    # we don't actually need continuous capture if we are just doing a single sensory reading
    camera.capture(rawCapture, format="bgr")
    IMG = rawCapture.array
    return return_orientation(IMG)

########################### ROBOT CLASS ##################################################

class Thymio:
    def __init__(self):
        self.aseba = self.setup()

    def drive(self, left_wheel_speed, right_wheel_speed):
        print("Left_wheel_speed: " + str(left_wheel_speed))
        print("Right_wheel_speed: " + str(right_wheel_speed))
        
        left_wheel = left_wheel_speed
        right_wheel = right_wheel_speed
        
        self.aseba.SendEventName("motor.target", [left_wheel, right_wheel])

    def stop(self):
        left_wheel = 0
        right_wheel = 0
        self.aseba.SendEventName("motor.target", [left_wheel, right_wheel])

    def sens(self):
        while True:
            prox_horizontal = self.aseba.GetVariable("thymio-II", "prox.horizontal")
            print("Sensing:")
            print(prox_horizontal[0])
            print(prox_horizontal[1])
            print(prox_horizontal[2])
            print(prox_horizontal[3])
            print(prox_horizontal[4])

############## Bus and aseba setup ######################################

    def setup(self):
        print("Setting up")
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        asebaNetworkObject = bus.get_object("ch.epfl.mobots.Aseba", "/")

        asebaNetwork = dbus.Interface(
            asebaNetworkObject, dbus_interface="ch.epfl.mobots.AsebaNetwork"
        )
        # load the file which is run on the thymio
        asebaNetwork.LoadScripts(
            "thympi.aesl", reply_handler=self.dbusError, error_handler=self.dbusError
        )

        # scanning_thread = Process(target=robot.drive, args=(200,200,))
        return asebaNetwork

    def stopAsebamedulla(self):
        os.system("pkill -n asebamedulla")

    def dbusReply(self):
        # dbus replys can be handled here.
        # Currently ignoring
        pass

    def dbusError(self, e):
        # dbus errors can be handled here.
        # Currently only the error is logged. Maybe interrupt the mainloop here
        print("dbus error: %s" % str(e))


#------------------ Main loop here -------------------------

def main():
    robot = Thymio()

    #robot.sens() 

    thread = Thread(target=robot.sens)
    thread.daemon = True
    thread.start()

    print(sense_target())

    ## doesn't actually drive. but sensing does work.
    robot.drive(200, 200)
    sleep(5)
    robot.stop()
       

#------------------- Main loop end ------------------------

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping robot")
        exit_now = True
        sleep(1)
        os.system("pkill -n asebamedulla")
        print("asebamodulla killed")
