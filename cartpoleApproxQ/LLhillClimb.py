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

	def __init__(self, envName, gamma0=1, gamma_l=0.995, maxEpisodes=200):

		self.env = gym.make(envName)
		self.env._max_episode_steps = maxEpisodes


		self.legalActions = [0,1,2,3]
		self.env.observation_space.n = len(self.env.observation_space.low)
		

		self.weights = np.random.rand(self.env.observation_space.n + 1) * 2 - 1
		self.gamma = gamma0
		self.gamma_decay = gamma_l

		self.gammas = []
		self.episodes = []
		self.rewards = []
		self.epsilons = []
		self.learnedWeights = []
		self.bestRewards = []

		# print ("Init complete, weights initialized to\n", self.weights)

	def chooseAction(self, state):
		value = np.dot(np.append(state, 1), self.weights)

		# Bin the value into (:,-1), [-1, 0), [0, 1), [0, :)
		if value < -1:
			action = self.legalActions[0]
		elif value < 0:
			action = self.legalActions[1]
		elif value < 1:
			action = self.legalActions[2]
		else:
			action = self.legalActions[3]

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

			# print("Episode", episode, "reward", reward)

			# if np.array_equal(self.weights, prevWeights):
			# 	print ("No change episode", episode)
			# 	print("self.weights", self.weights)
			# 	# exit()
			prevWeights = self.weights.copy()

			self.gammas.append(copy.copy(self.gamma))
			self.gamma *= self.gamma_decay


		# print("Learned Weights:", self.weights)

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

		return avgScore




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
""" SINGLE TEST """
# agent = hillClimbAgent("CartPole-v0", 1000)
# agent.learn(1000, 0)
# agent.plot()
# agent.test(0)

""" 100 TRIAL TEST, SET PARAMETERS """

testScores = []
for i in range(100):
	agent = hillClimbAgent("LunarLander-v2")
	agent.learn(1000, 0)
	testScore = agent.test(0)
	testScores.append(testScore)
	print ("Episode ", i, "score ", testScore)

print("Scores:", testScores)
plt.hist(testScores, bins='auto')
plt.title("Test Scores of 100 Hill Climb Instantiations")
plt.xlabel("Number of Occurrences")
plt.ylabel("Average Score over 100 Trials")
plt.show()

""" Parameter Sweep """
# numTrials = 100
# gammaDs = [0.999, 0.9975, 0.995, 0.99, 0.975, 0.95, 0.9]
# param_Successes = []
# for i in range(len(gammaDs)):
# 	testScores = []
# 	for j in range(numTrials):
# 		print("Parameter ", i, " Trial ", j)
# 		agent = hillClimbAgent("CartPole-v0", 1, gammaDs[i], 200)
# 		agent.learn(1000, 0)
# 		testScore = agent.test(0)
# 		testScores.append(testScore)
# 	testScores = np.array(testScores)
# 	testSuccesses = np.where(testScores > 195)
# 	numSuccess = np.size(testSuccesses)
# 	param_Successes.append(numSuccess)

# print (param_Successes)
# plt.plot(gammaDs, param_Successes)
# plt.xlabel("Parameters")
# plt.ylabel("Number of Success Out of 100 Trials")
# plt.title("Effect of Exploration Parameter on Learning Success")
# plt.show()

""" Sweep Max Episode Length """
# numTrials = 100
# maxEpsiodeLens = [100, 200, 300, 500, 750, 1000]

# param_Successes = []
# for i in range(len(maxEpsiodeLens)):
# 	testScores = []
# 	for j in range(numTrials):
# 		agent = hillClimbAgent("CartPole-v0", 1, 0.995, maxEpsiodeLens[i])
# 		agent.learn(1000, 0)
# 		testScores.append(agent.test(0))
# 		print("Parameter", i, "Trial", j)
# 	testScores = np.array(testScores)
# 	testSuccesses = np.where(testScores > 195)
# 	numSuccess = np.size(testSuccesses)
# 	param_Successes.append(numSuccess)

# print (param_Successes)
# plt.plot(maxEpsiodeLens, param_Successes)
# plt.xlabel("Max Episode Length")
# plt.ylabel("Number of Success Out of 100 Trials")
# plt.title("Effect of Maximum Episode Length on Learning Success")
# plt.show()
