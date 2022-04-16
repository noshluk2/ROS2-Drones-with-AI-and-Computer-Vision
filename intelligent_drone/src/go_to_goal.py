#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from math import pow, atan2, sqrt
from turtlesim.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class TurtleBot:

    def __init__(self):
        rospy.init_node('turtlebot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)
        self.pose_msg = ModelStates()
        self.robot_pos_x =0.0;self.robot_pos_y=0.0;self.robot_orientation_z=0.0;
        self.angle_to_goal=0.0;self.distance_to_goal=0.0;
        self.rate = rospy.Rate(10)
        
    def pose_callback(self, data):
        self.pose_msg = data
        self.robot_pos_x = round(self.pose_msg.pose[1].position.x, 4)
        self.robot_pos_y = round(self.pose_msg.pose[1].position.y, 4)
        self.robot_orientation_z = round(self.pose_msg.pose[1].orientation.z, 4)
        orientation_list = [self.pose_msg.pose[1].orientation.x, self.pose_msg.pose[1].orientation.y, 
                            self.pose_msg.pose[1].orientation.z, self.pose_msg.pose[1].orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        # print(self.robot_pos_x," / ",self.robot_pos_y," / ",self.robot_orientation_z ,"\n")
        # print(round(roll,3)," / ",round(pitch,3)," / ",round(yaw,3) ,"\n")

    def euclidean_distance(self, goal_pose):
        self.distance_to_goal=sqrt(pow((goal_pose.x - self.robot_pos_x), 2) +pow((goal_pose.y - self.robot_pos_y), 2))
        return self.distance_to_goal
    def linear_vel(self, goal_pose, constant=1.5):
        return constant * self.euclidean_distance(goal_pose)
# orientation_q = msg.pose.pose.orientation
#     orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
#     (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

    def move2goal(self):
        goal_pose = Pose()
        goal_pose.x = 0.2#float(input("Set your x goal: "))
        goal_pose.y = 6.2 #float(input("Set your y goal: "))
        distance_tolerance = 0.3
        vel_msg = Twist()
        while self.euclidean_distance(goal_pose) >= distance_tolerance:
            self.robot_orientation_z = self.robot_orientation_z * self.radian_to_degree
            angle_to_goal=(atan2(goal_pose.y - self.robot_pos_y ,goal_pose.x - self.robot_pos_y) ) *self.radian_to_degree
            heading_to_goal=angle_to_goal - self.robot_orientation_z
            # print("AG : " , round(heading_to_goal,4), "RA : ",round(self.robot_orientation_z,4),"DG : ",round(self.distance_to_goal,4))
            # vel_msg.linear.x = self.linear_vel(goal_pose)
            # vel_msg.angular.z = self.angular_vel(goal_pose)
            # self.velocity_publisher.publish(vel_msg)
            
            self.rate.sleep()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        
        # self.velocity_publisher.publish(vel_msg)
        rospy.spin()

if __name__ == '__main__':
    try:
        x = TurtleBot()
        x.move2goal()
    except rospy.ROSInterruptException:
        pass