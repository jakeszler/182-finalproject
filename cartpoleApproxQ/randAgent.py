import gym
import queue
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import copy

"""================================================================================
Hill Climbing Agent
   ================================================================================"""

class randomAgent:

	def __init__(self, envName, maxEpisodes=200):

		self.env = gym.make(envName)
		self.env._max_episode_steps = maxEpisodes


		self.legalActions = [0,1]
		self.env.observation_space.n = len(self.env.observation_space.low)
		

		self.weights = np.random.rand(self.env.observation_space.n + 1) * 2 - 1
		self.learnedWeights = []

		print ("Init complete, weights initialized to\n", self.weights)

	def chooseAction(self, state):
		value = np.dot(np.append(state, 1), self.weights)
		if value < 0:
			action = 0
		else:
			action = 1

		return action

	def runEpisode(self, render=False):
		# initialize obs
		prevObs = self.env.reset()
		done = False
		totalReward = 0.0

		# set new random weights for

		while not done:
			if render:
				self.env.render()
			action = self.chooseAction(prevObs)					# choose an action
			obs, reward, done, info = self.env.step(action)		# observe the result

			prevObs = obs
			totalReward += reward

		return totalReward


	def test(self, render=False):
		"""
		Check if episode solved: 100 epsiodes with maxepisodes = 200 again.
		Goal is average over 195
		"""
		self.env._max_episode_steps = 200

		testEpisodes = []
		testScores = []

		for i in range(100):
			reward = self.runEpisode(render)
			testEpisodes.append(i)
			testScores.append(reward)

		avgScore = np.mean(testScores)
		print("Average Score = ", avgScore)

		return avgScore



testScores = []
for i in range(100):
	agent = randomAgent("CartPole-v0", 1)
	testScores.append(agent.test(0))

print("Scores:", testScores)
plt.hist(testScores, bins='auto')
plt.title("Test Scores of 100 Random Weights")
plt.xlabel("Number of Occurances")
plt.ylabel("Average Score over 100 Trials")
plt.show()