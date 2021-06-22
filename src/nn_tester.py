from re import X
from numpy.core.fromnumeric import shape
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
from PIL import Image
import rospy
from std_msgs import msg
from std_msgs.msg import String
from robotics_demo.msg import TestMsg
from geometry_msgs.msg import Twist
import message_filters

# This part of the code saves the model to be used afterwards
# model = onnx.load('/home/iris/Desktop/Training_ball/results/rb_06/RollerBall/RollerBall-4858.onnx') # Load the ONNX file
# tf_model = prepare(model) # Import the ONNX model to Tensorflow
# tf_model.export_graph('/home/iris/Desktop/Training_ball/results/rb_06/b_test')
# print(shape(tf_model.inputs)) # Input nodes to the model
# print(tf_model.outputs) # Output nodes from the model



#!/usr/bin/env python
# license removed for brevity

class NNconnector:
    def __init__(self):# /home/iris/Desktop/Training_ball/results/rb_06/b_test
        self.tf_model = tf.saved_model.load('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall_saved')
        rospy.init_node('NNconnector', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.sub = message_filters.Subscriber('odometry',TestMsg,queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([self.sub], queue_size=1, slop=1, allow_headerless=True)
        ts.registerCallback(self._callback,self.pub)
        self.msg = Twist()
        # rospy.Rate(50)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
        
    def _callback(self,data_sub,pub_vel):
        # cube_pos1 = data_sub.cube_pos[0]
        # cube_pos2 = data_sub.cube_pos[1]
        # cube_pos3 = data_sub.cube_pos[2]
        
        # ball_pos1 = data_sub.ball_pos[0]
        # ball_pos2 = data_sub.ball_pos[1]
        # ball_pos3 = data_sub.ball_pos[2]
        
        # ball_vel1 = data_sub.ball_vel[0]
        # ball_vel2 = data_sub.ball_vel[1]
        
        # X = np.array([[data_sub.cube_pos[0],data_sub.cube_pos[1],data_sub.cube_pos[2],data_sub.ball_pos[0],data_sub.ball_pos[1],data_sub.ball_pos[2],data_sub.ball_vel[0],data_sub.ball_vel[1]]]).astype(np.float32)
        # y = self.tf_model(obs_0=X)
        y = self.tf_model(obs_0=np.array([[data_sub.cube_pos[0],data_sub.cube_pos[1],data_sub.cube_pos[2],data_sub.ball_pos[0],data_sub.ball_pos[1],data_sub.ball_pos[2],data_sub.ball_vel[0],data_sub.ball_vel[1]]]).astype(np.float32))
        
        # print(y[2][0][0].numpy())
        # print(y[2][0][1].numpy())
        self.msg.linear.x = y[2][0][0].numpy()
        self.msg.linear.z = y[2][0][1].numpy()
        self.pub.publish(self.msg)
        

if __name__ == '__main__':
    try:
        thisnode = NNconnector()
    except rospy.ROSInterruptException:
        pass
