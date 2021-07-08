from re import X
from numpy.core.fromnumeric import shape
# import onnx
import onnxruntime as rt
# from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import rospy
from std_msgs import msg
from std_msgs.msg import String
from robotics_demo.msg import TestMsg
from geometry_msgs.msg import Twist
import message_filters


class NNconnector:
    def __init__(self):
        self.sess = rt.InferenceSession('/home/iris/Desktop/Training_ball/results/rb_06/RollerBall/RollerBall-4858.onnx')
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[2].name
        rospy.init_node('NNconnector', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.sub = message_filters.Subscriber('odometry',TestMsg,queue_size=1,buff_size=2**24)
        ts = message_filters.ApproximateTimeSynchronizer([self.sub], queue_size=10, slop=1, allow_headerless=True)
        ts.registerCallback(self._callback,self.pub)
        self.msg = Twist()
        rospy.Rate(50)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
        
    def _callback(self,data_sub,pub_vel):
        
        
        X = np.array([[data_sub.cube_pos[0],data_sub.cube_pos[1],data_sub.cube_pos[2],data_sub.ball_pos[0],data_sub.ball_pos[1],data_sub.ball_pos[2],data_sub.ball_vel[0],data_sub.ball_vel[1]]]).astype(np.float32)
        result = self.sess.run(None, {self.input_name: X})        
        self.msg.linear.x = result[2][0][0]
        self.msg.linear.z = result[2][0][1]
        print(result[2][0])
        self.pub.publish(self.msg)
        

if __name__ == '__main__':
    try:
        thisnode = NNconnector()
    except rospy.ROSInterruptException:
        pass



