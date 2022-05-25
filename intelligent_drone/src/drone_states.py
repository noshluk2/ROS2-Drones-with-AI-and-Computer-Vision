#!/usr/bin/env python3
import rospy
from math import degrees,atan2,asin
from gazebo_msgs.srv import GetModelState


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


# Command line to get a model information
## rosservice call /gazebo/get_model_state "drone_model: 'chirya'"
if __name__ == "__main__":
    model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    # keep below lines in the loop
    response_variable = model_state_obj("chirya","ground_plane")
    x_pos=response_variable.pose.position.x
    y_pos=response_variable.pose.position.y
    quaternions = response_variable.pose.orientation
    print( "X / Y =" , x_pos," / " , y_pos )
    print( "quaternions =" , quaternions )    
    (roll,pitch,yaw)=euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
    yaw_deg = degrees(yaw)
    print( "yaw_deg =" , yaw_deg )    
