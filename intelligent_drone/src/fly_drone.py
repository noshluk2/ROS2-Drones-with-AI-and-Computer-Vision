#!/usr/bin/env python3
# license removed for brevity
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from gazebo_msgs.srv import GetModelState

def drone_flyer():
    global velocity_msg , pub
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/sonar_height", Range, sonar_callback)
    rospy.init_node('fly_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    velocity_msg=Twist()
    rospy.spin()

def sonar_callback(msg):
    global velocity_msg , pub
    sonar_data=msg.range
    if(sonar_data>=2.0):
        velocity_msg.linear.z=0.0
        print("Reached the Point")
        # rospy.signal_shutdown("As")
    else:
        velocity_msg.linear.z=0.5
        print("Reaching set point")


    pub.publish(velocity_msg)

    print("Sonar Values : " , sonar_data)




if __name__ == '__main__':
    model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    response_variable = model_state_obj("chirya","ground_plane")

    try:
        drone_flyer()
    except rospy.ROSInterruptException:
        pass