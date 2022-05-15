#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import GetModelState
import sys

# Command line to get a model information
## rosservice call /gazebo/get_model_state "drone_model: 'chirya'"


def model_state_obj_client(drone_model,reference_model):
    rospy.wait_for_service('/gazebo/get_model_state')
    model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    service_response_variableponse = model_state_obj(drone_model,reference_model)
    return service_response_variableponse

if __name__ == "__main__":
    drone_model = sys.argv[1] # user inputs
    reference_model = sys.argv[2]

    response_variable = model_state_obj_client(drone_model,reference_model)
    print("Position")
    print("X :  ", response_variable.pose.position.x)
    print("Y :  ", response_variable.pose.position.y)
    print("Z :  ", response_variable.pose.position.z)
    print("\nAngular W ", response_variable.pose.orientation.w)