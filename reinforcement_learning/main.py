#0x61-0x62 seems to be pos
#score seems to be little endian, not big

import random
from collections import deque

import retro
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.optimizers import Adam

class Mem:
	def __init__(self, size, state_dim, action_dim):
		self.mem_size = size
		self.cnt = 0
		self.rdy = False

		self.state = np.zeros((self.mem_size, state_dim))
		self.new_state = np.zeros((self.mem_size, state_dim))
		self.action = np.zeros(self.mem_size, dtype="uint8")
		self.reward = np.zeros(self.mem_size)
		self.done = np.zeros(self.mem_size, dtype=bool)

	def store(self, state, action, reward, new_state, done):
		self.state[self.cnt] = state
		self.action[self.cnt] = action
		self.reward[self.cnt] = reward
		self.new_state[self.cnt] = new_state
		self.done[self.cnt] = 1 - int(done)

		self.cnt += 1
		self.cnt %= self.mem_size

	def read(self, amount):
		keys = np.random.randint(0, self.cnt, size=amount)

		states = self.state[keys]
		actions = self.action[keys]
		rewards = self.reward[keys]
		new_states = self.new_state[keys]
		done = self.done[keys]

		return states, actions, rewards, new_states, done

	def ready(self, batch_size):
		if self.cnt > batch_size:
			self.rdy = True

		return self.rdy

class Test:
	#IDENT = np.identity(9, dtype=int)[6::]
	#IDENT = np.identity(9, dtype=int)

	def __init__(self, env, size):
		mem_size = 100000

		self.env = env

		self.epsilon = 1
		self.epsilon_decay = 0.001
		self.gamma = 0.9

		self.mem = Mem(mem_size, size, 9)

		self.netw = self.build(size)
		#self.target_netw = self.build(size)
		#self.align()

	def build(self, size):
		model = Sequential([
			#Embedding(size, 10, input_length=1),
			Flatten(),
			Dense(100, activation="relu"),
			Dense(25, activation="relu"),
			Dense(7, activation=None)
		])

		model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

		return model

	#def align(self):
	#	self.netw.set_weights(self.target_netw.get_weights())

	def action(self, state, episode):
		if np.random.rand() <= np.maximum(0.01, self.epsilon - episode * self.epsilon_decay):
			return random.randint(0, 7 - 1), None

		q = self.netw(state, training=False).numpy()
		#return q.flatten() > self.activation_min
		return np.argmax(q.flatten()), q.flatten()

	def train(self, batch_size):
		if not self.mem.ready(batch_size):
			return

		states, actions, rewards, new_states, done = self.mem.read(batch_size)

		q_now = self.netw(states, training=False).numpy()
		q_new = self.netw(new_states, training=False).numpy()

		q_target = np.copy(q_now)
		q_target[range(batch_size), actions] = rewards + self.gamma * np.max(q_new, axis=1) * done

		self.netw.train_on_batch(states, q_target)

class Resampler:
	@staticmethod
	def to_int(arr):
		arr = arr.astype("uint32")
		return (np.left_shift(arr[:, :, 0], 16) + np.left_shift(arr[:, :, 1], 8) + arr[:, :, 2]) / 0xFFFFFF

	@staticmethod
	def drop_sample(arr, n=2):
		return arr[::n, ::n]

	@staticmethod
	def reshape(arr):
		return arr.reshape((1, arr.size))

	@staticmethod
	def total(arr):
		return Resampler.reshape(Resampler.to_int(Resampler.drop_sample(arr)))


class Reward:
	def __init__(self):
		self.prev_x = -1
		self.score = -1

	def compute_reward(self, info):
		xpos = info.get("xpos")
		score = info.get("score")

		if self.prev_x == -1:
			self.prev_x = xpos

		if self.score == -1:
			self.score = score

		# Prevent weird behaviour when dying
		if xpos == 0:
			xpos = self.prev_x

		new_score = (xpos - self.prev_x) + (score - self.score)

		self.prev_x = xpos
		self.score = score

		return new_score

def main():
	# 6 = left
	# 7 = right
	# 8 = jump
	action_space = np.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 0, 0, 0, 1],
		[1, 0, 0, 0, 0, 0, 0, 1, 0],
		[1, 0, 0, 0, 0, 0, 1, 0, 0],
	])

	env = retro.make(game="TinyToonAdventures-Nes")	# running custom integrations
	state = Resampler.total(env.reset())

	agent = Test(env, state.size)
	batch_size = 64

	ZERO = np.zeros(3, dtype=int)

	i = -1
	while True:
		i += 1
		state = Resampler.total(env.reset())
		rew_handler = Reward()

		done = False
		j = -1
		score = 0
		while not done:
			j += 1
			#if j % 100 == 0 and j > 0:
			#	print(f"Timestep {j}: {info}, {subrew}")
			#	print(action, q_val)

			if j % 1 == 0:
				env.render()

			action, q_val = agent.action(state, i)

			subrew = 0

			# Hold button for 6 frames, then observe
			for _ in range(6):
				next_state, _, done, info = env.step(action_space[action])
				rew = rew_handler.compute_reward(info)
				subrew += rew
				score += rew

			#for _ in range(2):
			#	next_state, _, done, info = env.step(ZERO)
			#	rew = rew_handler.compute_reward(info)
			#	subrew += rew
			#	score += rew

			next_state = Resampler.total(next_state)

			if info.get("lives") == 1:
				done = True

			agent.mem.store(state, action, rew, next_state, done)

			state = next_state
			agent.train(batch_size)

		print(f"Episode {i}, score {score}, {info}")

"""
while True:
	obs, rew, done, info = env.step(env.action_space.sample())

	env.render()

	if done:
		obs = env.reset()
		env.close()
		break
"""

if __name__ == "__main__":
	#tf.compat.v1.disable_eager_execution()
	main()