#!/usr/bin/python3
import os
# initialize asebamedulla in background and wait 0.3s to let
# asebamedulla startup
os.system("(asebamedulla ser:name=Thymio-II &) && sleep 0.3")
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from picamera import PiCamera
from time import sleep
import dbus
import dbus.mainloop.glib
from adafruit_rplidar import RPLidar
from math import cos, sin, pi, floor
import threading
from picamera.array import PiRGBArray


print("Starting robot")

#-----------------------init script--------------------------
camera = PiCamera()

def dbusError(self, e):
    # dbus errors can be handled here.
    # Currently only the error is logged. Maybe interrupt the mainloop here
    print('dbus error: %s' % str(e))


# init the dbus main loop
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    
# get stub of the aseba network
bus = dbus.SessionBus()
asebaNetworkObject = bus.get_object('ch.epfl.mobots.Aseba', '/')
    
# prepare interface
asebaNetwork = dbus.Interface(
    asebaNetworkObject,
    dbus_interface='ch.epfl.mobots.AsebaNetwork'
)
    
# load the file which is run on the thymio
asebaNetwork.LoadScripts(
    'thympi.aesl',
    reply_handler=dbusError,
    error_handler=dbusError
)

#signal scanning thread to exit
exit_now = False

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME)
#This is where we store the lidar readings
scan_data = [0]*360
#--------------------- init script end -------------------------

def testCamera():
    print("Camera test")
    camera.start_preview()
    sleep(5)
    #we capture to openCV compatible format
    #you might want to increase resolution
    camera.resolution = (320, 240)
    camera.framerate = 24
    sleep(2)
    image = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(image, 'bgr')
    cv2.imwrite('out.png', image) 
    camera.stop_preview()
    print("saved image to out.png")

def testThymio():
    left_wheel = 20
    right_wheel = 200
    asebaNetwork.SendEventName(
        'motor.target',
        [left_wheel, right_wheel]
    )
    print("motor should be running now")
    sleep(5)
    asebaNetwork.SendEventName(
        'motor.target',
        [0, 0]
    )


#NOTE: if you get adafruit_rplidar.RPLidarException: Incorrect descriptor starting bytes
# try disconnecting the usb cable and reconnect again. That should fix the issue
def lidarScan():
    print("Starting background lidar scanning")
    for scan in lidar.iter_scans():
        if(exit_now):
            return
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance

#----------------------------- IMAGE PROCESSING FUNCTIONS --------------------------
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
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    sleep(0.1)
    # grab an image from the camera.
    # we don't actually need continuous capture if we are just doing a single sensory reading
    camera.capture(rawCapture, format="bgr")
    IMG = rawCapture.array
    return return_orientation(IMG)

# -----------------------------------------------------------------

scanner_thread = threading.Thread(target=lidarScan)
scanner_thread.daemon = True
scanner_thread.start()

#------------------ Main loop here -------------------------

def mainLoop():
    #do stuff
    lidar = [scan_data[0], scan_data[45], scan_data[90], scan_data[135], scan_data[180], scan_data[225], scan_data[270], scan_data[315]]
    colour = sense_target()
    print(lidar + [sense_target()])
    testThymio()
    sleep(3)
    print()

#------------------- Main loop end ------------------------

#this enables the prox.com communication channels
asebaNetwork.SendEventName( "prox.comm.enable", [1])
asebaNetwork.SendEventName("prox.comm.rx",[0])
    

def sendInformation(number):
    #asebaNetwork.SetVariable("thymio-II", "prox.comm.tx", [number])
    asebaNetwork.SendEventName(
            "prox.comm.tx",
            [number]
            )
def get_rx_reply(r):
     print("found", r)
def get_rx_error(e):
    print("error:", e)
    print(str(e))
def receiveInformation():
     rx = asebaNetwork.GetVariable("thymio-II", "prox.comm.rx")
     #asebaNetwork.GetVariable("thymio-II", "prox.comm.rx", \
     #        reply_handler=get_rx_reply,error_handler=get_rx_error)
     print(rx[0])

if __name__ == '__main__':
    #testCamera()
    #testThymio()
    #testLidar()
    try:
        while True:
           # receiveInformation()
            mainLoop()
    except KeyboardInterrupt:
        print("Stopping robot")
        exit_now = True
        sleep(1)
        lidar.stop()
        lidar.disconnect()
        os.system("pkill -n asebamedulla")
        print("asebamodulla killed")
