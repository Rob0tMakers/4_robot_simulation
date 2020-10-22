import shapely
import numpy as np
from shapely.geometry import LinearRing, LineString, Point
from numpy import sin, cos, pi, sqrt
from random import random

### Constants ###

# arena measurements

R = 0.021  # radius of wheels in meters
L = 0.094  # distance between wheels in meters

W = 1.18  # width of arena
H = 1.94  # height of arena
margin = 0.05 # offset position of robot from wall

# floor/ceil coordinates for particle distribution
x_floor = -W/2 + margin
x_ceil = W/2 - margin
y_floor = -H/2 + margin
y_ceil = H/2 - margin

# range for particle distribution
range_x = W - 2 * margin
range_y = H - 2 * margin

# the world is a rectangular arena with width W and height H
world = LinearRing([(W/2,H/2),(-W/2,H/2),(-W/2,-H/2),(W/2,-H/2)])

### Functions ###

# converts degrees to radians
def dtr(deg):
    return deg * (np.pi / 180)


# this function returns a normal distributed array of view angles (q), based on camera input
# the reason why "right" is q = 0, is because Casper's simulation was built that way
# IMPORTANT: only call this function, when camera_int != 0. The camera has to report something, otherwise spin robot until it does.
def distributeViewAngles(camera_int):
    view_angle = 0 # this is the direction where the robot camera and lidar[0] are facing => 0 >> right (default)
    spread = 20 # this is the std. deviation 
    
    # case red >> top right corner
    if camera_int == 1:
        view_angle = 45

    # case red/green >> right
    if camera_int == 1.5:
        view_angle = 0
    
    # case green >> bottom right corner
    if camera_int == 2:
        view_angle = 315
        
    # case green/blue >> bottom
    if camera_int == 2.5:
        view_angle = 270

    # case blue >> bottom left corner
    if camera_int == 3:
        view_angle = 225
    
    # case blue/yellow >> left
    if camera_int == 3.5:
        view_angle = 180
        
    # case yellow >> top left corner
    if camera_int == 4:
        view_angle = 135
        
    # case yellow/red >> top
    if camera_int == 4.5:
        view_angle = 90
        
    return dtr(np.random.normal(view_angle, spread, (100, 1)))


# this function does the initial particle placement, random spread across the whole map (minus a small margin)
# calls distributeViewAngles for normal distribution of q
def spawnParticles(camera_int):
    start_x = x_floor
    start_y = y_ceil
    
    random_values = np.random.rand(100, 2)
    
    calc_pos_x = np.vectorize(lambda x: start_x + range_x * x)
    calc_pos_y = np.vectorize(lambda y: start_y - range_y * y)
    
    x_col = calc_pos_x(random_values[:,:1])
    y_col = calc_pos_y(random_values[:,1:2])
    
    q_col = distributeViewAngles(camera_int)

    particles = np.concatenate((x_col, y_col, q_col), axis=1)
    return particles


# this function returns a normal distributed array of particles based on the previous location approximation
def resampleParticles(x, y, camera_int):
    x_col = np.clip(np.random.normal(x, 0.1, (100, 1)), x_floor, x_ceil)
    y_col = np.clip(np.random.normal(y, 0.1, (100, 1)), y_floor, y_ceil)
    
    q_col = distributeViewAngles(camera_int)
    
    return np.concatenate((x_col, y_col, q_col), axis=1)


# this utility functions allows application of simulateParticleLidar() to a matrix, using matrix operations
# it is a wrapper function for simulateParticleLidar()
# takes an array [x, y, q] as input
def rollLinewise(vector):
    rangeMat = np.arange(8).reshape((8,1))
    
    laserMat = np.array([vector,]*8)
    
    joinMat = np.concatenate((laserMat, rangeMat), axis=1)
    
    return np.apply_along_axis(simulateParticleLidar, 1, joinMat)


# this function returns an array of 8 lidar measurements with a step of 45 degrees
# takes a matrix of [[x, y, q, i], ... ] as input, where i is a multiplicator for 45 degrees

def simulateParticleLidar(mat):
    line = LineString([(mat[0], mat[1]), (mat[0]+cos(mat[2]+dtr(mat[3]*45))*2*W,(mat[1]+sin(mat[2]+dtr(mat[3]*45))*2*H))])
    s = world.intersection(line)
    distance = sqrt((s.x-mat[0])**2+(s.y-mat[1])**2)
    
    return distance


### Exportable Function ###
def approximateLocation(robot_lidar, camera_output, x_hat, y_hat, q_hat, toggle):
    particles = np.array([])

    if toggle == True:
        particles = spawnParticles(camera_output) # set particles [[x, y, q], ...] n=100
        toggle = False
    else:
        # we are using the previous approximation (x_hat, y_hat) to approximate the next particle
        particles = resampleParticles(x_hat, y_hat, q_hat) # set particles [[x, y, q], ...]

    # this array contains the distances to the walls, simulating our lidar laser range finder's output
    particles_lidar = np.apply_along_axis(rollLinewise, 1, particles)
    
    # delta matrix [[0, 1, 2, 3, 4, 5, 6, 7], ...] n=100
    delta = np.absolute(particles_lidar - robot_lidar)
    
    # delta array [0, 1, 2, ...] n=100
    sum_delta = np.apply_along_axis(np.sum, 1, delta)
    
    # lowest delta index
    delta_min_i = np.argmin(sum_delta)
    
    # location approximation
    approx = particles[delta_min_i]
    
    # if an error threshold is passed, the particle filtering should start from the beginning with a random sample
    if sum_delta[delta_min_i] > 2.5:
        toggle = True

    return approx, toggle