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

class hillClimbAgent:

	def __init__(self, envName, maxEpisodes=200):

		self.env = gym.make(envName)
		self.env._max_episode_steps = maxEpisodes


		self.legalActions = [0,1]
		self.env.observation_space.n = len(self.env.observation_space.low)
		

		self.weights = np.random.rand(self.env.observation_space.n + 1) * 2 - 1
		self.gamma = 1

		self.gammas = []
		self.episodes = []
		self.rewards = []
		self.epsilons = []
		self.learnedWeights = []
		self.bestRewards = []

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


	def learn(self, episodes, render=False):
		episodesPlot = range(episodes)

		prevWeights = self.weights.copy()
		bestReward = float('-inf')

		for episode in range(episodes):
			# randomly explore nearby weights
			self.weights = self.weights + (np.random.rand(self.env.observation_space.n + 1) * 2 - 1) * self.gamma

			reward = self.runEpisode(render)
			self.rewards.append(reward)

			if reward > bestReward:
				bestReward = reward
			else:
				self.weights = prevWeights

			self.episodes.append(episode)
			self.learnedWeights.append(self.weights.copy())
			self.bestRewards.append(bestReward)

			print("Episode", episode, "reward", reward)

			# if np.array_equal(self.weights, prevWeights):
			# 	print ("No change episode", episode)
			# 	print("self.weights", self.weights)
			# 	# exit()
			prevWeights = self.weights.copy()

			self.gammas.append(copy.copy(self.gamma))
			self.gamma *= 0.995


		print("Learned Weights:", self.weights)

	def plot(self):
		plotDim = [4,1]

		def epPlot(yVals):
			plt.plot(self.episodes, yVals)

		plt.tight_layout()

		plt.subplot(plotDim[0], plotDim[1],1)
		epPlot(self.gammas)
		plt.title("Exploration Over Time")

		plt.subplot(plotDim[0], plotDim[1],2)
		epPlot(self.rewards)
		plt.title("Rewards")

		learnedWeights = np.array(self.learnedWeights.copy())
		print (learnedWeights)

		plt.subplot(plotDim[0], plotDim[1],3)
		for i in range(learnedWeights.shape[1]):
			epPlot(learnedWeights[:,i])
		plt.title("Weights")

		plt.subplot(plotDim[0], plotDim[1], 4)
		epPlot(self.bestRewards)
		plt.title("Running Best Reward")

		plt.tight_layout()


		plt.show(block=False)
		input("Press Enter to Finish")
		plt.close()

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


agent = hillClimbAgent("CartPole-v0", 1000)
agent.learn(1000, 0)
agent.plot()
agent.test(0)

"""
	Some issues with hill climbing:
	- 	The best reward possible is 200, so if a set of weights randomly and occasionaly produces
		a 200 time step episode, the weights will never improve if this controller has a lower
		success rate than a better one.  Solutions include:
			- 	increasing the max_episode steps:
				env.tags['wrapper_config.TimeLimit.max_episode_steps'] = desiredSteps
			-	try each set of weights over multiple episodes and improve based on the average

	- 	This seems to be a version of First-choice hill climbing though the random successor
		generation is limited in distance by the self.gamma factor

	-	The hill climbing does tend to get stuck often, it would be good to measure this by
		running the learn function multiple times (basically random restarts) and see what
		percentage of the time a good solution is found.
			- Might also try stochastic hill climbing


	Some Plots to make:
	- Test Score vs Max Training Score (need to take averages)
	- Test Score vs averaging filter on standard 200 max score (can sweep filter width)

"""