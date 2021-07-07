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
# from tensorboardX import SummaryWriter
from std_msgs.msg import String
from robotics_demo.msg import ObservationMsg
from geometry_msgs.msg import Twist
import message_filters
import random

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam


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
        self.episodeNum = 0
        self.m_StepCount = 0
        self.max_steps = 600
        # self.gradient_steps = 0
        self.actor_loss_memory = []
        self.critic_loss_memory = []
        self.reward_memory = []
        self.done = False # completed episode or not
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
        
        self.run()
        

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
        
    def get_action_continuous(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        else:
            action = action_matrix = p[0]
        return action, action_matrix, p

    def step(self):

        for _ in range(self.max_steps):
            # Take an action
            self.pub.publish(self.msg)
            
            # Get observation
            obs = []
            obs = [self.cube_pos,
                      self.ball_pos,
                      self.ball_vel]
            
            distanceToTarget = math.sqrt( ((self.cube_pos[0]-self.ball_pos[0])**2)+((self.cube_pos[2]-self.ball_pos[2])**2) )
            # Are we touching the target?
            if (distanceToTarget<1.42):
                reward = self.SetReward(1)
                self.EndEpisode()
                self.done = True

            elif (self.cube_pos[1]<0.5):
                reward = self.SetReward(-1)
                self.EndEpisode()
                self.done = True

        # We surpassed the max number of steps. Start new episode
        if self.done is False:
            # The number of steps is over but we didnt touch
            # the target
            reward = self.SetReward(-0.1)
            self.EndEpisode()
            self.done = True
            
        done = self.done
        informat = [0] # What is information, BTW?
            
# TODO hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        return obs, reward, done, informat
    
    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action_continuous()
        observation, reward, done, info = self.step(action)
        # self.reward.append(reward) ALREADY DONE INSIDE STEP()
        
        tmp_batch[0].append(self.observation)
        tmp_batch[1].append(action_matrix)
        tmp_batch[2].append(predicted_action)
        self.observation = observation
        
        if done:
            print("Done!")
            self.transform_reward()
            if self.val is False:  # for now this is always true
                for i in range(len(tmp_batch[0])):
                    obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
            tmp_batch = [[], [], []]
            self.EndEpisode()
            
        print("Batch full, training nets...")
        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward
            
    # DONT KNOW VERY WELL WHAT THIS FUNCTION DOES     
    def transform_reward(self):
        # if self.val is True:
        #     self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        # else:
        #     self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA
        

    def run(self):
        while self.episodes < EPISODES:
            print("Episode: ", self.episode)
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)
            
            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, callbacks=[self.tensorboard_callback])
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, callbacks=[self.tensorboard_callback])
            # self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            # self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            # print(actor_loss.history['loss'][-1])

            self.actor_loss_memory.append(actor_loss.history['loss'][-1])
            self.critic_loss_memory.append(critic_loss.history['loss'][-1])
            self.reward_memory.append(np.mean(reward))
            # self.gradient_steps += 1
            
            


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
