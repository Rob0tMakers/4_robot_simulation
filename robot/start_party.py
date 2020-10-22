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
import time
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
# this is where we store the IR readings
prox_horiz = [0]*5
# store receiving signals here
rx = [0]
#--------------------- init script end -------------------------

#NOTE: if you get adafruit_rplidar.RPLidarException: Incorrect descriptor starting bytes
# try disconnecting the usb cable and reconnect again. That should fix the issue
def lidarScan():
    print("Starting background lidar scanning")
    for scan in lidar.iter_scans():
        if(exit_now):
            return
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance
            
def infraredScan():
    while True: 
        sensor_readings = asebaNetwork.GetVariable("thymio-II", 'prox.horizontal')
        for i in range(5):
            prox_horiz[i] = int(sensor_readings[i])
            # we can try to save 5 and 6 as they are back sensors
        for i in range(6,8):
            prox_back[i] = int(sensor_readings[i])


def followWall(direction):
    if direction == "cw":
        side_sensor = 270
        L_turnfromwall = 150
        R_turnfromwall = 10
        L_turntowall = 40
        R_turntowall = 20
        L_turncorner = 200
        R_turncorner = 0
        far_threshold = 175
        near_threshold = 75
        wait_time = 3

    else:
        side_sensor = 90
        L_turnfromwall = 100
        R_turnfromwall = 30
        L_turntowall = 150
        R_turntowall = 20
        L_turncorner = 0
        R_turncorner = 50
        far_threshold = 200
        near_threshold = 150
        wait_time = 2

    if any(sensor > 2000 for sensor in prox_horiz) and scan_data[0] > 250 : #obstacle found
        asebaNetwork.SendEventName(
        'motor.target', [0,0])
        return True

    elif scan_data[0] < 200: # sees a corner ahead
        print('corner')
        asebaNetwork.SendEventName(
        'motor.target',
        [L_turncorner, R_turncorner]) 
        sleep(wait_time)

    elif scan_data[side_sensor] < near_threshold: #Too close to the wall (adjust right)
        asebaNetwork.SendEventName(
        'motor.target',
        [L_turnfromwall, R_turnfromwall]) 

    elif scan_data[side_sensor] > far_threshold: #Too far from the wall (adjust left)
        asebaNetwork.SendEventName(
        'motor.target',
        [L_turntowall, R_turntowall]) 
      
    else: #move forward
        asebaNetwork.SendEventName(
        'motor.target',
        [200, 50]) 

    return False

def calibrate():
    targetSeen = False
    seen = sense_target()
    if seen != 0:
        # wait to see if we see the same thing multiple times
        seen2 = sense_target()
        if seen == seen2:
            targetSeen = True
    while not targetSeen:
        # turn counterclockwise
        asebaNetwork.SendEventName(
        'motor.target', [0,15])
        sleep(0.5)
        asebaNetwork.SendEventName(
        'motor.target', [0,0])
        seen = sense_target()
        if seen != 0:
            # wait to see if we see the same thing multiple times
            seen2 = sense_target()
            if seen == seen2:
                targetSeen = True
    return seen
    
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

# -----------------------------------------------------------------

# -------------------- Party functions ----------------------------

def benchWarm():
    wait_time = 60
    t_0 = time.time()
    print("It is now " + str(t_0))
    end_time = t_0 + wait_time
    print("We will wait for a dance partner until " + str(end_time))
    gender = np.random.randint(1,2) #perhaps create class robot so you can set this to a variable defining the robot?
    if gender == 1:
        # set color to blue
        asebaNetwork.SendEventName("leds.top", [0,0,32])
    else:
        # set color to red
        asebaNetwork.SendEventName("leds.top", [32,0,0])
    
    while time.time() < end_time: ##### See if we can actually detect a dance partner this way
        sendInformation(gender)
        if rx[0] != gender:
            print("Partner found <3 <3 <3")
            if rx[0] in [3,4,5,6]:
                dancefloor = rx[0]
                moveToDanceFloor(dancefloor)

    print("Time to go find someone myself!")
    findDancePartner(gender)

def findDancePartner(gender):
    print("Finding a dance partner")
    # move clockwise. First turn along the wall blindly
    rx = [gender]
    asebaNetwork.SendEventName('motor.target', [0,100]) #adjust so a 90deg turn is done to the left.
    sleep(2.5)
    obstacle = False
    while not obstacle:
        sendInformation(gender)
        obstacle = followWall('cw')
    # check to see if we have received a signal
    if rx[0] != gender:
        sendInformation(gender)
        print("Partner found!!!!")
        dancefloor = np.random.randint(3,6)
        for i in range(5):
            sendInformation(dancefloor) #after it is sent 5 times, assume its been seen
        moveToDanceFloor(dancefloor)
    # if we haven't seen the right gender
    sleep(3)
    print("Lets try the other direction")
    # try opposite directions
    asebaNetwork.SendEventName('motor.target', [200,0])
    sleep(8.25)
    obstacle = False
    while not obstacle:
        sendInformation(gender)
        obstacle = followWall('ccw')
    if rx[0] != gender:
        sendInformation(gender)
        print("Partner found!!!!")
        dancefloor = np.random.randint(3,6)
        for i in range(5):
            sendInformation(dancefloor) #after it is sent 5 times, assume its been seen
        moveToDanceFloor(dancefloor)
    sleep(3)
    print('no one again')
    
    # turn so we face outagain
    returnToRest()

def moveToDanceFloor(dancefloor):
    asebaNetwork.SendEventName("leds.top", [32,0,32])
    #locate where you are (particleFilter) --> should this be a speerate thread?
    # determine how to move to dancefloor area
    # execute move
    # doublecheck you are in the riht spot?
    pass

def dance():
    # make up a dance!! Max time 15 seconds then returnToRest
    # moving and then doing a circle?
    # randomly move in the parameter. (w sensors calibrated)
    pass

def returnToRest():
    # Locate area
    # Determine nearest wall
    # Move to nearest wall
    # Turn away from the wall
    # benchWarm() again.
    pass

# ----------------------------------------------------------

scanner_thread = threading.Thread(target=lidarScan)
scanner_thread.daemon = True
scanner_thread.start()

IR_thread = threading.Thread(target=infraredScan)
IR_thread.daemon = True
IR_thread.start()

receiving_thread = threading.Thread(target=receiveInformation)
receiving_thread.daemon = True
receiving_thread.start()

#------------------ Main loop here -------------------------

def mainLoop():
    #benchWarm()
    print(calibrate())
    print('Calibration is over.')

#------------------- Main loop end ------------------------


if __name__ == '__main__':
    #testCamera()
    #testThymio()
    #testLidar()
    sleep(1)
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
        asebaNetwork.SendEventName(
        'motor.target',
        [0, 0])
        os.system("pkill -n asebamedulla")
        print("asebamodulla killed")
