import gym
import numpy as np
import pdb
env = gym.make('CartPole-v0')
num_trials = 100
timestep_max = 200
gamma = 0.9
score = 0

# (x, x', theta, theta')
num_discretized_sections = (1, 1, 5, 3)

# (left, right)
num_actions = env.action_space.n

state_bounds = list(zip(env.observation_space.low, env.observation_space.high))  # [(l,u), (l,u), (l,u), (l,u)]
state_bounds[1] = [-0.5, 0.5]  # bound criteria
state_bounds[3] = [-np.radians(50), np.radians(50)]  # bound criteria

q_table = np.zeros(num_discretized_sections + (num_actions,))  # (1,1,6,3,2)

def discretize_state(state):
    map_to_bucket_index = []
    for counter, feature in enumerate(state):
        bucket_index = None
        if feature < state_bounds[counter][0]:
             bucket_index = 0
        elif feature > state_bounds[counter][1]:
            bucket_index = num_discretized_sections[counter] - 1
        else:
            offset = (num_discretized_sections[counter] - 1) * state_bounds[counter][0] / (state_bounds[counter][1] - state_bounds[counter][0])
            scale_factor = (num_discretized_sections[counter] - 1) / (state_bounds[counter][1] - state_bounds[counter][0])
            bucket_index = int(round(scale_factor * state[counter] - offset))
        map_to_bucket_index.append(bucket_index)
    return tuple(map_to_bucket_index)

for trial in range(num_trials):
    obs = env.reset()  # (4,)
    prev_state = discretize_state(obs)

    for t in range(timestep_max):
        # env.render()
        action = np.argmax(q_table[prev_state])
        obs, reward, done, _ = env.step(action)
        score += reward
        next_state = discretize_state(obs)
        q_table[prev_state + (action,)] += reward + gamma * np.amax(q_table[next_state])
        prev_state = next_state
        if done:
            # if failed
            print('trial:{}/{} finished at timestep:{}'.format(
                trial, num_trials, t))
            break
