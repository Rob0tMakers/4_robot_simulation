#!/usr/bin/python3
import os
# initialize asebamedulla in background and wait 0.3s to let
# asebamedulla startup
os.system("(asebamedulla ser:name=Thymio-II &) && sleep 0.3")
import matplotlib.pyplot as plt
import numpy as np
from picamera import PiCamera
from time import sleep
import time
import dbus
import dbus.mainloop.glib
from adafruit_rplidar import RPLidar
from math import cos, sin, pi, floor
import threading
from image_processing import sense_target
from particle_filtering import approximateLocation


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
# for particle filtering
x_hat = 0
y_hat = 0
toggle = True # toggling between random sampling and gaussian sampling
# camera output
camera_output = 0
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


def locationFinder():
    while True:
        if camera_output > 0:
            approximateLocation(np.array(scan_data), camera_output, x_hat, y_hat, toggle)



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

    if receiveInformation() in [1,2] and receiveInformation() not gender:
        asebaNetwork.SendEventName(
        'motor.target', [0,0])
        return True

    if any(sensor > 1000 for sensor in prox_horiz) and scan_data[0] > 250 : #obstacle found
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
    seen = sense_target(camera)
    if seen != 0:
        # wait to see if we see the same thing multiple times
        seen2 = sense_target(camera)
        if seen == seen2:
            targetSeen = True
    while not targetSeen:
        # turn counterclockwise
        asebaNetwork.SendEventName(
        'motor.target', [0,15])
        sleep(0.5)
        asebaNetwork.SendEventName(
        'motor.target', [0,0])
        seen = sense_target(camera)
        if seen != 0:
            # wait to see if we see the same thing multiple times
            seen2 = sense_target(camera)
            if seen == seen2:
                targetSeen = True
    return seen

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


# -----------------------------------------------------------------

# -------------------- Party functions ----------------------------

def benchWarm():
    wait_time = 60
    t_0 = time.time()
    print("It is now " + str(t_0))
    end_time = t_0 + wait_time
    print("We will wait for a dance partner until " + str(end_time))
    gender = np.random.randint(1,3) #perhaps create class robot so you can set this to a variable defining the robot?
    if gender == 1:
        # set color to blue
        asebaNetwork.SendEventName("leds.top", [0,0,32])
    else:
        # set color to red
        asebaNetwork.SendEventName("leds.top", [32,0,0])
    
    while time.time() < end_time: ##### See if we can actually detect a dance partner this way
        sendInformation(gender)
        rx = asebaNetwork.GetVariable("thymio-II", "prox.comm.rx")
        print(rx[0])
        if rx[0] not in set([0, gender]):
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
    rx = asebaNetwork.GetVariable("thymio-II", "prox.comm.rx")
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
    rx = asebaNetwork.GetVariable("thymio-II", "prox.comm.rx")
    if rx[0] not in set([0, gender]):
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

scanner_thread = threading.Thread(target=lidarScan, daemon = True)
scanner_thread.start()

IR_thread = threading.Thread(target=infraredScan, daemon = True)
IR_thread.start()

# receiving_thread = threading.Thread(target=receiveInformation, daemon = True)
# receiving_thread.start()

location_thread = threading.Thread(target=locationFinder, daemon = True)
location_thread.start()

#------------------ Main loop here -------------------------

def mainLoop():
    findDancePartner(2)

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
