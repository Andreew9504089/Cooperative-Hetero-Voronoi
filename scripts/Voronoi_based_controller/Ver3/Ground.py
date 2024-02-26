#!/usr/bin/env python3

import rospy
from voronoi_cbsa.msg import TargetInfoArray, TargetInfo

import pygame
import numpy as np
from time import time, sleep
from control import PTZCamera
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
import itertools

def norm(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]**2

    return sqrt(sum)

def TargetDynamics(x, y, v, seed=0, cnt=0):
    spd = 0.005
    #np.random.seed(seed[cnt])
    a = np.random.rand(1)[0] + 1
    turn = np.random.randint(-10*a, 10*a)/180*np.pi if a > 0 else 0
    rot = np.array([[cos(turn), -sin(turn)],
                    [sin(turn), cos(turn)]])
    v = rot@v.reshape(2,1)
    vx = v[0] if v[0]*spd + x > 0 and v[0]*spd + x < 24 else -v[0]
    vy = v[1] if v[1]*spd + y > 0 and v[1]*spd + y < 24 else -v[1]
    
    return (x,y), np.asarray([[0],[0]])
    #return (np.round(float(np.clip(v[0]*spd + x, 0, 24)),3), np.round(float(np.clip(v[1]*spd + y, 0, 24)),3)),\
    #        np.round(np.array([[vx],[vy]]), len(str(spd).split(".")[1]))

def RandomUnitVector():
    v = np.asarray([np.random.normal() for i in range(2)])
    return v/norm(v)

if __name__ == "__main__":
    rospy.init_node('ground_control_station', anonymous=True, disable_signals=True)
    rate = rospy.Rate(60)

    target_pub = rospy.Publisher("/target", TargetInfoArray, queue_size=10)
    
    def random_pos(pos=(0,0)):
        if pos == (0,0):
            x = np.random.random()*15 + 5
            y = np.random.random()*15 + 5
            return np.array((x,y))
        else:
            return np.asarray(pos)
        
    # targets = [[random_pos((8,16)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']],
    #            [random_pos((16,8)), 0.8, 10,np.array([1.,1.]), ['camera', 'smoke_detector']]]
    
    #targets = [[random_pos((16,16)), 2, 10,np.array([1.,1.]), ['camera', 'manipulator']]]
    
    targets = [[random_pos((12,8)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']],
               [random_pos((10,14)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']],
               [random_pos((14,14)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']]]
    
    # targets = [[random_pos((12,12)), 1.2, 10,np.array([1.,1.]), ['camera', 'manipulator', 'smoke_detector']],
    #            [random_pos((8,16)), 1.2, 10,np.array([1.,1.]), ['camera', 'smoke_detector']],
    #            [random_pos((16,8)), 1.2, 10,np.array([1.,1.]), ['camera', 'manipulator']]]
    
    # targets = [[random_pos((12,6)), 1.2, 10,np.array([1.,1.]), ['manipulator']],
    #            [random_pos((6,18)), 1.2, 10,np.array([1.,1.]), ['smoke_detector']],
    #            [random_pos((18,18)), 1.2, 10,np.array([1.,1.]), ['camera']]]
    
    # targets = [[random_pos((5,5)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']],
    #            [random_pos((19,19)), 0.8, 10,np.array([1.,1.]), ['camera', 'smoke_detector']],
    #            [random_pos((5,19)), 0.8, 10,np.array([1.,1.]), ['camera', 'manipulator']],
    #            [random_pos((19,5)), 0.8, 10,np.array([1.,1.]), ['camera', 'smoke_detector']]]
    
    cnt = 0
    a = 5
    seeds = [range(10000*a - 9999, 10000*a), range(20001*a - 10000, 20000*a),
             range(30001*a - 10000, 30000*a),range(40001*a - 10000, 40000*a)]
    
    while not rospy.is_shutdown():
            
        grid_size = rospy.get_param("/grid_size", 0.1)
        tmp = []
        cnt += 1
        for i in range(len(targets)):
            pos, vel = TargetDynamics(targets[i][0][0], targets[i][0][1], targets[i][3])
            #pos = np.array([6*np.cos(((-1)**i)*(cnt/5)/180*np.pi) + 12, 6*np.sin((((-1)**i)*cnt/5)/180*np.pi) + 12])
            targets[i][0] = pos
            targets[i][3] = vel

            target_msg = TargetInfo()
            target_msg.id = i
            target_msg.position.x = pos[0]
            target_msg.position.y = pos[1]
            target_msg.standard_deviation = targets[i][1]
            target_msg.weight = targets[i][2]
            target_msg.velocity.linear.x = vel[0]
            target_msg.velocity.linear.y = vel[1]
            target_msg.required_sensor = targets[i][4]

            tmp.append(target_msg)
        
        targets_array = TargetInfoArray()
        targets_array.targets = [tmp[i] for i in range(len(tmp))]

        target_pub.publish(targets_array)
        rate.sleep()