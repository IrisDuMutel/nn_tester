####################### USING GYM ENVS #################################
# 1. It renders instances for 500 timesteps, performing random actions.
# import gym
# env = gym.make('Acrobot-v1')
# # env = gym.make('Taxi-v3')
# env.reset()
# for _ in range(500):
#     env.render()
#     env.step(env.action_space.sample())
# # 2. To check all env available, uninstalled ones are also shown.
# from gym import envs 
# print(envs.registry.all())



import gym
env = gym.make('MountainCarContinuous-v0') # try for different environments
observation = env.reset()
for t in range(100):
        env.render()
        print(type(observation))
        print( observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print (observation, reward, done, info)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break


#################### USING HIGHWAY ENV #######################
# # general package imports
# import os
# import time
# # RL specific package imports
# import gym
# import highway_env

# # plotting specific import statements
# import numpy as np
# print('numpy: '+np.version.full_version)
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 20})
# import matplotlib.image as mpimg
# from mpl_toolkits.mplot3d import Axes3D 
# import matplotlib.animation as animation
# import matplotlib
# print('matplotlib: '+matplotlib.__version__)

# # creating an instance of roundabout environment
# env_h = gym.make("highway-v0")
# # converting the roundabout environment into a finite mdp
# mdp_h = env_h.unwrapped.to_finite_mdp()

# print("Lane change task MDP Transition Matrix shape: "+str(mdp_h.transition.shape))
# print("Lane change task Reward Matrix shape: "+str(mdp_h.reward.shape))

# # this calculates evaluates the deterministic policy
# # for the deterministic version of roundabout environment
# def determine_policy(mdp, v, gamma=1.0):
#     policy = np.zeros(mdp.transition.shape[0])
#     for s in range(mdp.transition.shape[0]):
#         q_sa = np.zeros(env.action_space.n)
#         for a in range(env.action_space.n):
#             s_ = mdp.transition[s][a]
#             r = mdp.reward[s][a]
#             q_sa[a] += (1 * (r + gamma * v[s_]))
#         policy[s] = np.argmax(q_sa)
#     return policy


# # value iteration algorithm's baseline implementation
# def value_iteration(mdp, env, gamma=0.99):
#     value = np.zeros(mdp.transition.shape[0])
#     max_iterations = 10000
#     eps = 1e-10

#     for i in range(max_iterations):
#         prev_v = np.copy(value)
#         for s in range(mdp.transition.shape[0]):
#             q_sa = np.zeros(env.action_space.n)
#             for a in range(env.action_space.n):
#                 s_ = mdp.transition[s][a]
#                 r = mdp.reward[s][a]
#                 q_sa[a] += (1 * (r + gamma * prev_v[s_]))
#             value[s] = max(q_sa)
#             ind_ = np.argmax(q_sa)
#         if (np.sum(np.fabs(prev_v - value)) <= eps):
#             print('Problem converged at iteration %d.' % (i + 1))
#             break
#     return value

# def main():
#     # inline code execution for value iteration
#     # and policy determination functions
#     gamma = 0.99
#     env = gym.make('roundabout-v0')
#     mdp = env.unwrapped.to_finite_mdp()
#     optimal_value_func = value_iteration(mdp, env, gamma)
#     start_time = time.time()
#     policy = determine_policy(mdp, optimal_value_func, gamma)
#     print("Best Policy Values Determined for the MDP.\n")
#     print(policy)

# if __name__ == "__main__":
#     main()