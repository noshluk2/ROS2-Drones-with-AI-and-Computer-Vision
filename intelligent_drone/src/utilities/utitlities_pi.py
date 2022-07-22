import rospy
from gazebo_msgs.srv import GetModelState
from math import degrees,atan2,asin,tan,radians


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians



def calc_focalLen_h_pix(image_width,hfov_deg = 90,drone_orientation = "nadir"):
    
    hfov_pix = 0
    if drone_orientation=="nadir":
        hfov_pix = (image_width/2) / (tan(radians(hfov_deg/2)))
    
    return hfov_pix