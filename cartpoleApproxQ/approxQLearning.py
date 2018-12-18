import gym
import queue
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import copy

"""================================================================================
Approximate Q Learning Agent
   ================================================================================"""

class learningAgent:

	def __init__(self, envName, maxEpisodes):

		random.seed(10)

		self.env = gym.make(envName)
		self.env._max_episode_steps = maxEpisodes

		self.legalActions = [0, 1]
		self.env.observation_space.n = len(self.env.observation_space.low)


		# for the Approximate Q Learning:
		self.weights = np.random.rand(self.env.action_space.n, self.env.observation_space.n + 1) * 2 - 1 # need to have a constant
		self.discount = 0.9
		self.alpha = 0.9		# learning rate
		self.epsilon = 1		# exploration rate

		# keep track of metadata for plotting
		self.trials = []
		self.rewards = []
		self.epsilons = []
		self.learnedweights = []



		print ("Init complete, weights initialized to\n", self.weights)


	def getQVal(self, state, action):
		weights = self.weights[action].copy()	# action is either 0 or 1
		features = np.append(state, 1)

		QVal =  np.dot(weights, features)
		return QVal

	def maxQVal(self, state):
		"""
		Returns (bestQVal, bestAction) for a given state
		"""
		QVals = []
		for action in self.legalActions:
			QVals.append(self.getQVal(state, action))
		# get the max Q Val
		bestQVal = np.amax(QVals)
		bestActions = np.argwhere(QVals == bestQVal)	# find which actions lead to the bestQVal
		choosenAction = np.random.choice(np.ndarray.flatten(bestActions))
		# print(bestQVal, choosenAction)
		return (bestQVal, choosenAction)

	def chooseAction(self, state):

		# proceed randomly at the exploration rate
		if random.random() < self.epsilon:
			action = self.env.action_space.sample()
			# print("action chosen randomly")
		# otherwise choose the best action according to the q values
		else:
			action = self.maxQVal(state)[1]

		# print("Chosen action", action)
		return action

	def updateWeights(self, curState, action, nextState, reward):
		# print("\nUpdateWeights before computation:")


		prevWeights = self.weights.copy()
		# print("prevWeights", prevWeights)
		features = np.append(nextState.copy(), 1)
		# print("features", features)
		difference = reward + self.discount * self.maxQVal(nextState)[0] - self.getQVal(curState, action)

		actionWeights = features.copy()

		for i in range(len(features)):
			actionWeights[i] += self.alpha * difference * features[i]

		self.weights[action] = actionWeights

		# print("\tafter computation")
		# print("prevWeights\t", prevWeights)
		# print("curState\t", curState)
		# print("action\t\t", action)
		# print("nextState\t", nextState)
		# print("reward\t\t", reward)
		# print("self.alpha\t", self.alpha)
		# print("diff\t\t", difference)
		# print("features\t", features)
		# print("New weights \t", self.weights)
		# print("\n")

		return actionWeights

	def runEpisode(self, render=False):
		# initialize obs
		prevObs = self.env.reset()
		done = False
		totalReward = 0.0

		prevWeights = self.weights.copy()

		runWeights = []
		steps = []
		step = 0
		while not done:
			if render:
				self.env.render()

			# print("runEpsiode step", step, "before action")
			# print("prevWeights\t", prevWeights)
			# print("self.wghts\t", self.weights)

			action = self.chooseAction(prevObs)					# choose an action
			obs, reward, done, info = self.env.step(action)		# observe the result
			actionWeights = self.updateWeights(prevObs, action, obs, reward)	# update learning

			# print("runEpsiode step", step, "after action")
			# print("prevWeights\t", prevWeights)
			# print("self.wghts\t", self.weights)
			# if np.array_equal(self.weights.all(), prevWeights.all()):
			# 	print("\nNo change in step", step, "self.weights", self.weights)
			# print("self.weights", self.weights)

			prevObs = obs
			totalReward += reward

			steps.append(step)
			step += 1
		# 	runWeights.append(self.weights)

		# runWeights = np.array(runWeights)
		# print(runWeights, runWeights.shape)

		# for i in range(runWeights.shape[1]):
		# 	for j in range(runWeights.shape[2]):
		# 		# print(i,j, type(i), type(j))
		# 		plt.plot(steps, runWeights[:,i,j])
		# plt.title("Evolution of weights within epsiode")
		# plt.show()



		return totalReward


	def learn(self, trials, episodesPerTrial, render=False):
		epsilonPlot = []

		prevWeights = self.weights.copy()

		for trial in range(trials):
			epsilonPlot.append(self.epsilon)
			reward = self.runEpisode(render)

			self.trials.append(trial)
			self.rewards.append(reward)
			self.epsilons.append(self.epsilon)
			self.learnedweights.append(self.weights.copy())

			self.epsilon *= 0.995

			print("trial", trial, "reward", reward)
			prevWeights = self.weights.copy()


		print("Learned Weights:", self.weights, "Final Epsilon", self.epsilon)

	def plot(self):

		plotDim = [3,1]

		def epPlot(yVals):
			plt.plot(self.trials, yVals)

		# plt.subplot(plotDim[0], plotDim[1], 1)
		# epPlot(self.epsilons)
		# plt.title("Epsilon")

		plt.subplot(plotDim[0], plotDim[1],2)
		epPlot(self.rewards)
		plt.title("Rewards")

		plt.subplot(plotDim[0], plotDim[1],3)
		learnedWeights = np.array(self.learnedweights)
		plt.title("Weights")

		print("learned weights", learnedWeights)
		# print("size learned weights", learnedWeights.shape)
		# print("learnedweights[0,:,:]", learnedWeights[:,0, 0])


		for i in range(len(learnedWeights[0])):
			for j in range(len(learnedWeights[0,i])):
				epPlot(learnedWeights[:,i,j])
		# print("learnedweights[0,0,:]", self.learnedweights[:])


		# plt.subplot(4,1,2)
		# print("learnedWeights", self.learnedweights)
		# print("\n\n\nlearnedWeights[:][0][0]", self.learnedweights[:][0][0])
		# for i in range(len(self.learnedweights[:][0])):
		# 	epPlot(self.learnedweights[:][0][i])

		plt.show(block=False)
		input("Press Enter to End")
		plt.close()


agent = learningAgent("CartPole-v0", 500)
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
"""