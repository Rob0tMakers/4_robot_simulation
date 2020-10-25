import numpy as np
from shapely.geometry import Point

x_hat = 0.59 / 2
y_hat = 0
q_hat = 0

floor_red = [0.295, 0.485]
floor_yellow = [-0.295, 0.485]
floor_green = [0.295, -0.485]
floor_blue = [-0.295, -0.485]

def dtr(deg):
    return deg * (np.pi / 180)

def getDistanceToTarget(target_x, target_y):
    target = Point(target_x, target_y)
    return Point(x_hat, y_hat).distance(target)

def getAngleToTarget(target_x, target_y):
    delta_x = target_x - x_hat
    delta_y = target_y - y_hat

    if delta_x == 0:
      delta_x = 0.001

    angle = np.arctan(delta_y / delta_x)

    return angle - q_hat

result = getAngleToTarget(floor_red[0], floor_red[1])

print(result)