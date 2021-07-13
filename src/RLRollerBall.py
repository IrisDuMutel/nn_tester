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



CONTINUOUS = True

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 2048
BATCH_SIZE = 256
NUM_ACTIONS = 2
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


# This code is in charge of the RL algorithm



def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss

class RLRollerBall:

    def __init__(self):# /home/iris/Desktop/Training_ball/results/rb_06/b_test
        # self.tf_model = tf.saved_model.load('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall_saved')
        self.critic = self.build_critic()
        self.actor = self.build_actor_continuous()
        global episodeNum 
        episodeNum = 0
        global m_StepCount 
        m_StepCount = 0
        global max_steps 
        max_steps = 600
        # self.gradient_steps = 0
        global actor_loss_memory
        actor_loss_memory = []
        global critic_loss_memory
        critic_loss_memory = []
        global reward_memory
        reward_memory = []
        global done 
        done = False # completed episode or not
        global val
        val = False  # to do or not to do random actions
        global batch_is_full
        batch_is_full = False
        global is_new_episode
        is_new_episode = True
        global reward 
        reward = []
        
    # def get_action_continuous(self,state):
    #     self.observation = np.array(state)
    #     # print(self.observation)
    #     # print(self.observation.reshape(1, NUM_STATE)).reshape(1, NUM_STATE)
    #     # self.observation.reshape(1, NUM_STATE)
    #     # print(shape(self.observation.reshape(1,NUM_STATE)))
    #     # print(shape(DUMMY_VALUE))
    #     # print(shape(DUMMY_ACTION))
    #     p = self.actor.predict([self.observation, DUMMY_VALUE, DUMMY_ACTION])
    #     if self.val is False:
    #         action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
    #     else:
    #         action = action_matrix = p[0]
    #     return action, action_matrix, p
    
    def build_actor_continuous(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, name='output', activation='tanh')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()
        model.save('actor.h5')

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        model.save('critic.h5')

        return model
    
   
        
def _callback(data_sub):
    global callback_msg
    global observation
    global agent
    global max_steps
    global m_StepCount
    global done
    global episodeNum
    global batch
    global tmp_batch
    global batch_is_full
    global is_new_episode
    global reward
    global reward_memory
    global val
    global msgOdom
    global msgActions
        

    
    callback_msg = data_sub
    
    cube_pos = data_sub.cube_pos
    ball_pos = data_sub.ball_pos
    ball_vel = data_sub.ball_vel
    
    observation = [ data_sub.cube_pos[0],
                    data_sub.cube_pos[1],
                    data_sub.cube_pos[2],
                    data_sub.ball_pos[0],
                    data_sub.ball_pos[1],
                    data_sub.ball_pos[2],
                    data_sub.ball_vel[0],
                    data_sub.ball_vel[1]]
        
    if episodeNum < EPISODES:
        
        if is_new_episode:
            print("Episode: ", episodeNum)
            batch = [[], [], [], []]
            tmp_batch = [[], [], []]
            is_new_episode = False
            
        if len(batch[0]) < BUFFER_SIZE:
            ###### GET_ACTION_CONTINUOUS CODE ############
            obs = np.array(observation)
            # print(self.observation)
            # print(self.observation.reshape(1, NUM_STATE)).reshape(1, NUM_STATE)
            # self.observation.reshape(1, NUM_STATE)
            # print(shape(self.observation.reshape(1,NUM_STATE)))
            # print(shape(DUMMY_VALUE))
            # print(shape(DUMMY_ACTION))
            predicted_action = agent.actor.predict([obs, DUMMY_VALUE, DUMMY_ACTION])
            if val is False:
                action = action_matrix = predicted_action[0] + np.random.normal(loc=0, scale=NOISE, size=predicted_action[0].shape)
            else:
                action = action_matrix = predicted_action[0]
            
            #############################################
            
            # action, action_matrix, predicted_action = agent.get_action_continuous(observation)
            
        ######### STEP CODE ###############    
        msgActions.linear.x = action[0]
        msgActions.linear.z = action[2]
        pubActions.publish(msgActions)
        observation = [ data_sub.cube_pos[0],
                        data_sub.cube_pos[1],
                        data_sub.cube_pos[2],
                        data_sub.ball_pos[0],
                        data_sub.ball_pos[1],
                        data_sub.ball_pos[2],
                        data_sub.ball_vel[0],
                        data_sub.ball_vel[1]]
        
        if m_StepCount < max_steps:
            m_StepCount += 1
            # Take an action
            # self.msg = action
            # self.pub.publish(self.msg)
            distanceToTarget = math.sqrt( ((cube_pos[0]-ball_pos[0])**2)+((cube_pos[2]-ball_pos[2])**2) )
            # Are we touching the target?
            if (distanceToTarget<1.42):
                reward.append(1)
                if episodeNum % 100 == 0:
                    val = True
                else:
                    val = False        
                done = True
            elif (cube_pos[1]<0.5):
                reward.append(-1)
                if episodeNum % 100 == 0:
                    val = True
                else:
                    val = False        
                done = True
                
        # We surpassed the max number of steps. Start new episode
        else:
            # The number of steps is over before touching target
            reward.append(-0.1)
            if episodeNum % 100 == 0:
                val = True
            else:
                val = False        
            done = True 
            
            
            ############### STEP CODE END ##############

            
        # informat = [0] # What is information, BTW?
        tmp_batch[0].append(observation)
        tmp_batch[1].append(action_matrix)
        tmp_batch[2].append(predicted_action)
        
        # For any reason, the episode is over, so we
        if done:
            print("Done!")
            for j in range(len(reward) - 2, -1, -1):
                reward[j] += reward[j + 1] * GAMMA
            if val is False:  # for now this is always true
                for i in range(len(tmp_batch[0])):
                    observation, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = reward[i]
                    batch[0].append(observation)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
            tmp_batch = [[], [], []]
            
            # End episode code run ##########
            episodeNum += 1
            if episodeNum % 100 == 0:
                val = True
            else:
                val = False        
            m_StepCount = 0
            msgOdom.ball_vel = [0, 0]
            msgOdom.ball_pos = [0, 0.5, 0]
            msgOdom.cube_pos = [random.uniform(0, 1)*8-4, 0.5, random.uniform(0, 1)*8-4]
            
            observation = [msgOdom.cube_pos[0],
                                msgOdom.cube_pos[1],
                                msgOdom.cube_pos[2],
                                msgOdom.ball_pos[0],
                                msgOdom.ball_pos[1],
                                msgOdom.ball_pos[2],
                                msgOdom.ball_vel[0],
                                msgOdom.ball_vel[1]]
            done = False
            is_new_episode = True
            pubStates.publish(observation)
            ######################################
                
        print("Batch full, training nets...")
        observation, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        # return obs, action, pred, reward
        
        observation, action, pred, reward = observation[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
        old_prediction = pred
        pred_values = agent.critic.predict(observation)
        
        advantage = reward - pred_values
        actor_loss = agent.actor.fit([observation, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, callbacks=[agent.tensorboard_callback])
        critic_loss = agent.critic.fit([observation], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, callbacks=[agent.tensorboard_callback])
        # self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
        # self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
        # print(actor_loss.history['loss'][-1])
        agent.actor_loss_memory.append(actor_loss.history['loss'][-1])
        agent.critic_loss_memory.append(critic_loss.history['loss'][-1])
        reward_memory.append(np.mean(reward))
        # self.gradient_steps += 1
            


    


agent = RLRollerBall()

rospy.init_node('NNconnector', anonymous=True)
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

        
while not rospy.is_shutdown():
    rospy.spin()
