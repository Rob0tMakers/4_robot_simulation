from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.rotation = 180
camera.start_preview()
sleep(5)
for i in range(20):
  camera.capture('/home/pi/Pictures/img%s.jpg' % i)
  sleep(1)
camera.stop_preview()
