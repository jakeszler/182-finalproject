
import numpy as np
import pdb
import math
import random
import matplotlib.pyplot as plt
import gym
env = gym.make('CartPole-v0')
num_trials = 400
timestep_max = 200
gamma = 0.9
score = 0
alpha = .1
epsilon = 1.0

time_alive_lists = []
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


for num_buckets in range(10):
    time_alive_list = []
    num_b = num_buckets + 1
    num_discretized_sections = (1, 1, num_b, num_b)


    num_actions = env.action_space.n

    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-np.radians(50), np.radians(50)]

    q_table = np.zeros(num_discretized_sections + (num_actions,))



    for trial in range(num_trials):
        obs = env.reset()

        prev_state = discretize_state(obs)

        #set epsilon set alpha#####################################################
        zero_point = 200
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
                time_alive_list.append(t)
                # alpha_list.append(alpha)
                # eps_list.append(epsilon)
                break
    time_alive_lists.append(time_alive_list)
# plt.plot(time_alive_list)
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
# plt.plot(time_alive_list)


for i in range(len(time_alive_lists)):
    plt.plot(time_alive_lists[i], label = i + 1)

plt.xlabel("episode number")
plt.ylabel("total reward for episode")
plt.title("Q learning with 6 buckets for theta, 3 buckets for theta' one bucket for x and x'")
plt.legend()
plt.show()
