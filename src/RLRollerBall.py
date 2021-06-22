from re import X
from numpy.core.fromnumeric import shape
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import math
from PIL import Image
import rospy
from std_msgs import msg
from std_msgs.msg import String
from robotics_demo.msg import ObservationMsg
from geometry_msgs.msg import Twist
import message_filters
import random

CONTINUOUS = False

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 2048
BATCH_SIZE = 256
NUM_ACTIONS = 4
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


# This code is in charge of the RL algorithm

class RLRollerBall:

    def __init__(self):# /home/iris/Desktop/Training_ball/results/rb_06/b_test
        self.tf_model = tf.saved_model.load('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall_saved')
        self.episodeNum = 1
        self.m_StepCount = 0
        self.max_steps = 600
        self.done = False
        self.val = False  # to do or not to do random actions
        self.reward = []
        rospy.init_node('NNconnector', anonymous=True)
        self.pub = rospy.Publisher('/BallActions',ObservationMsg,queue_size=10)
        self.sub = message_filters.Subscriber('BallOdometry',ObservationMsg,queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([self.sub], queue_size=1, slop=1, allow_headerless=True)
        ts.registerCallback(self._callback,self.pub)
        self.msg = ObservationMsg()
        rospy.spin()
        
    def _callback(self,data_sub,pub_vel):
        self.cube_pos = data_sub.cube_pos
        self.ball_pos = data_sub.ball_pos
        self.ball_vel = data_sub.ball_vel

        # y = self.tf_model(obs_0=np.array([[data_sub.cube_pos[0],data_sub.cube_pos[1],data_sub.cube_pos[2],data_sub.ball_pos[0],data_sub.ball_pos[1],data_sub.ball_pos[2],data_sub.ball_vel[0],data_sub.ball_vel[1]]]).astype(np.float32))
        # self.msg.linear.x = y[2][0][0].numpy()
        # self.msg.linear.z = y[2][0][1].numpy()
        self.pub.publish(self.msg)

    def SetReward(self, value):
        #do something
        print("a")
        self.reward.append(value)

    def EndEpisode(self):

        self.episodeNum += 1
        if self.episodeNum % 100 == 0:
            self.val = True
        else:
            self.val = False
        
        # ResetData();
        self.m_StepCount = 0
        self.OnEpisodeBegin()

    def step(self):

        for _ in range(self.max_steps):
            # Take an action


            # Set the message
            self.msg.ball_vel[0] = 0 # X vel of the ball
            self.msg.ball_vel[1] = 0 # Z vel of the ball
            
            distanceToTarget = math.sqrt( ((self.cube_pos[0]-self.ball_pos[0])**2)+((self.cube_pos[2]-self.ball_pos[2])**2) )
            # Are we touching the target?
            if (distanceToTarget<1.42):
                self.SetReward(1)
                self.EndEpisode()
                self.done = True

            elif (self.cube_pos[2]<0.5):
                self.SetReward(-1)
                self.EndEpisode()
                self.done = True

        
        if self.done is False:
            # The number of steps is over but we didnt touch
            # the target
            self.SetReward(-0.1)
            self.EndEpisode()
            self.done = True
        


            
                
                # SetReward
        

    def run(self):
        while self.episodes < EPISODES:
            print("Episode: ", self.episode)


            
            
            
            
        




    def OnEpisodeBegin(self):
        self.msg.ball_vel[0] = 0 # X velocity of the ball
        self.msg.ball_vel[1] = 0 # Y velocity of the ball
        self.msg.ball_pos[0] = 0 # X position of ball
        self.msg.ball_pos[1] = 0.5 # Y position of ball
        self.msg.ball_pos[2] = 0 # Z position of ball
        self.msg.cube_pos[0] = random.uniform(0, 1)*8-4  # X position of the target
        self.msg.cube_pos[0] =  0.5 # Y position of the target
        self.msg.cube_pos[0] = random.uniform(0, 1)*8-4  # X position of the target



if __name__ == '__main__':
    try:
        thisnode = RLRollerBall()
    except rospy.ROSself.ball_poserruptException:
        pass
