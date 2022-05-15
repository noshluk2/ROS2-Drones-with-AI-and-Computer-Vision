#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import GetModelState

# Command line to get a model information
## rosservice call /gazebo/get_model_state "drone_model: 'chirya'"

if __name__ == "__main__":
    model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    # keep below lines in the loop
    response_variable = model_state_obj("chirya","ground_plane")
    x_pos=response_variable.pose.position.x
    y_pos=response_variable.pose.position.y
    print( "X / Y =" , x_pos," / " , y_pos )