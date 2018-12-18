#code by Jackson Wagner
import numpy as np
import pdb
import math
import random
import matplotlib.pyplot as plt
import gym
env = gym.make('LunarLander-v2')
num_trials = 10000
timestep_max = 10000
gamma = 0.9

alpha = .1
epsilon = 1.0

score_lists = []
alpha_list = []
eps_list = []

def discretize_state(state):
    map_to_bucket_index = []
    for counter, feature in enumerate(state):
        if feature < state_bounds[counter][0]:
             bucket_index = 0
        elif feature > state_bounds[counter][1]:
            bucket_index = num_discretized_sections[counter] - 1
        else:
            bucket_index = assignIntermediateIndex(counter, state)
        map_to_bucket_index.append(bucket_index)
    return tuple(map_to_bucket_index)

def assignIntermediateIndex(counter, state):
    offset = (num_discretized_sections[counter] - 1) * state_bounds[counter][0] / (state_bounds[counter][1] - state_bounds[counter][0])
    scale_factor = (num_discretized_sections[counter] - 1) / (state_bounds[counter][1] - state_bounds[counter][0])
    return int(round(scale_factor * state[counter] - offset))

# (x, x', theta, theta')
for num_buckets in range(10):
    score_list = []
    num_b = num_buckets + 1
    score = 0

    num_discretized_sections = (num_b,num_b,num_b,num_b,num_b,num_b,2,2)

    # (left, right)
    num_actions = env.action_space.n

    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))  # [(l,u), (l,u), (l,u), (l,u)]

    #TEST TO FIND BOUNDS FOR STATES
    # for i in range(len(state_bounds)):
    #     print state_bounds[i]
    #
    #
    # arr = []
    #
    # for trial in range(1000):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         arr.append(obs)
    #         #print obs
    #         #action = random.randint(0, num_actions - 1)
    #         action = num_trials % num_actions
    #         obs, reward, done, _ = env.step(action)

    # arr = np.array(arr)
    # max_arr = np.max(arr,axis=0)
    # min_arr = np.min(arr, axis = 0)
    #
    # print "max: ", list(max_arr)
    # print "min: ", list(min_arr)

    state_bounds[0] = [-1,1]
    state_bounds[1] = [-1,1]
    state_bounds[2] = [-1,1]
    state_bounds[3] = [-3,1]
    state_bounds[4] = [-4,4]
    state_bounds[5] = [-4,4]
    state_bounds[6] = [0,1]
    state_bounds[7] = [0,1]



    q_table = np.zeros(num_discretized_sections + (num_actions,))



    for trial in range(num_trials):
        score = 0
        obs = env.reset()  # (4,)

        prev_state = discretize_state(obs)

        #set epsilon set alpha#####################################################
        zero_point = 5000
        if trial >= zero_point:
            epsilon = 0
            alpha = .001
        else:
            epsilon = (1-float(trial)/zero_point)
            alpha = (1-float(trial)/zero_point) * 0.5


        #epsilon = .2
        #alpha = .2


        for t in range(timestep_max):
            # env.render()
            random_num = random.random()
            if random_num > epsilon:
                action = np.argmax(q_table[prev_state])
            else:
                action = random.randint(0, num_actions - 1)


            #print("taking action {}".format(action))
            # print "state ", obs
            # print "discretized state ", prev_state
            # print "q values", q_table[prev_state]
            # print ""
            # print ""
            obs, reward, done, _ = env.step(action)
            score += reward
            #print "reward ", reward
            next_state = discretize_state(obs)
            # q_table[prev_state + (action,)] += reward + gamma * np.amax(q_table[next_state])
            old_value = q_table[prev_state + (action,)]
            #print "q before ", q_table[prev_state + (action,)]
            q_table[prev_state + (action,)] = (1 - alpha) * old_value + alpha * (reward + gamma * np.amax(q_table[next_state]))
            #print "q after ", q_table[prev_state + (action,)]
            prev_state = next_state
            if done:
                # if failed
                print('trial:{}/{} finished at timestep:{}'.format(
                    trial, num_trials, t))
                score_list.append(score)
                # alpha_list.append(alpha)
                # eps_list.append(epsilon)
                break
    score_lists.append(score_list)
# plt.plot(score_list)
# plt.ylabel('total_reward')
# plt.xlabel('trial_number')
# plt.title('epsilon = 0.2, alpha = 0.7')
# plt.show()


# plt.figure(1)
# plt.subplot(311)
# plt.ylabel('alpha')
# plt.plot(alpha_list)
#
# plt.subplot(312)
# plt.ylabel("epsilon")
# plt.plot(eps_list)
#
# plt.subplot(313)
# plt.ylabel("total reward")
# plt.xlabel('episode number')
# plt.plot(score_list)


for i in range(len(score_lists)):
    plt.plot(score_lists[i], label = i + 1)

plt.xlabel("episode number")
plt.ylabel("total reward for episode")
#plt.title("Q learning with 6 buckets for theta, 3 buckets for theta' one bucket for x and x'")
plt.legend()
plt.show()
