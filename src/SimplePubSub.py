from re import X
from numpy.core.fromnumeric import shape
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import math
from PIL import Image
from tensorflow.python.ops.numpy_ops import np_dtypes
import rospy
from std_msgs import msg
# from tensorboardX import SummaryWriter
from std_msgs.msg import String
from robotics_demo.msg import OdometryMsg
from geometry_msgs.msg import Twist
import message_filters
import random

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator


        
def _callback(data_sub):
    global pubActions
    global msgActions
    print('aaaaa')
    
    msgActions.linear.x = 0.3
    
    pubActions.publish(msgActions)
    
            


    


########### Basic initialization of everyhting! ###########

rospy.init_node('NNconnector', anonymous=True)
global pubActions
pubActions = rospy.Publisher('/BallActions',Twist,queue_size=10)
pubStates = rospy.Publisher('ResetScene',OdometryMsg,queue_size=1)
# self.sub = message_filters.Subscriber('BallOdometry',OdometryMsg,queue_size=1)
# ts = message_filters.ApproximateTimeSynchronizer([self.sub], queue_size=1, slop=1, allow_headerless=True)
# ts.registerCallback(self._callback,self.pubActions, self.pubStates)
sub = rospy.Subscriber('BallOdometry',OdometryMsg,_callback,queue_size=1)
global msgOdom
msgOdom = OdometryMsg()
global msgActions
msgActions = Twist()

###########################################################
        
while not rospy.is_shutdown():
    rospy.spin()
