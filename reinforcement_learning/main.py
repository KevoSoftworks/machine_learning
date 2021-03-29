#0x61-0x62 seems to be pos
#score seems to be little endian, not big

import random
from collections import deque

import retro
import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.optimizers import Adam

class Test:
	def __init__(self, env, size):
		self.env = env

		self.epsilon = 0.8
		self.epsilon_decay = 0.01
		self.gamma = 0.6
		self.activation_min = 0.5

		self.replay = deque(maxlen=2**16)

		self.netw = self.build(size)
		self.target_netw = self.build(size)
		#self.align()

	def build(self, size):
		model = Sequential([
			#Embedding(size, 10, input_length=1),
			Flatten(),
			Dense(50, activation="relu"),
			Dense(25, activation="relu"),
			Dense(9, activation="softmax")
		])

		model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

		return model

	def store(self, *args):
		self.replay.append(args)

	def align(self):
		self.target_netw.set_weights(self.netw.get_weights())

	def action(self, state, episode):
		if np.random.rand() <= np.maximum(0.1, self.epsilon - episode * self.epsilon_decay):
			return self.env.action_space.sample()
		
		q = self.netw(state, training=False).numpy()
		return q.flatten() > self.activation_min

	def train(self, batch_size):
		batch = random.sample(self.replay, batch_size)
		states_batch = np.zeros((batch_size, batch[0][0].size))
		next_batch = np.zeros((batch_size, batch[0][3].size))
		reward = np.zeros(batch_size)
		action = np.zeros((batch_size, batch[0][1].size), dtype="uint8")

		i = 0
		for state, act, rew, next_state in batch:
			states_batch[i] = state
			next_batch[i] = next_state
			reward[i] = rew
			action[i] = act
			
			i += 1

		target_batch = self.netw.predict(states_batch, batch_size=batch_size, workers=10, use_multiprocessing=True)
		t = self.target_netw.predict(next_batch, batch_size=batch_size, workers=10, use_multiprocessing=True)

		#print(action.shape)
		#print(np.amax(t, axis=1))

		for i, act in enumerate(action):
			target_batch[i][act] = reward[i] + self.gamma * np.amax(t[i])

		"""print(batch)

		i = 0
		for state, act, rew, next_state, done in batch:
			target = self.netw(state, training=False).numpy().flatten()

			if done:
				target[act] = rew
			else:
				t = self.target_netw(next_state, training=False).numpy()
				target[act] = rew + self.gamma * np.amax(t)
			
			target_batch[i] = target
			states_batch[i] = state
			i += 1
		"""
		self.netw.fit(states_batch, target_batch, epochs=1, verbose=0, workers=10, use_multiprocessing=True)

class Resampler:
	@staticmethod
	def to_int(arr):
		arr = arr.astype("uint32")
		return (np.left_shift(arr[:, :, 0], 16) + np.left_shift(arr[:, :, 1], 8) + arr[:, :, 2]) / 0xFFFFFF

	@staticmethod
	def drop_sample(arr, n=32):
		return arr[::n, ::n]

	@staticmethod
	def reshape(arr):
		return arr.reshape((1, arr.size))

	@staticmethod
	def total(arr):
		return Resampler.reshape(Resampler.to_int(Resampler.drop_sample(arr)))

def main():
	env = retro.make(game="TinyToonAdventures-Nes")	# running custom integrations
	state = Resampler.total(env.reset())

	agent = Test(env, state.size)
	batch_size = 64

	for i in range(100):
		print(f"Episode {i}")
		state = Resampler.total(env.reset())
		for j in range(128):
			if j % 5 == 0 and j > 0:
				print(f"Timestep {j}")
				print(f"{info}, {rew}")
				env.render()

			action = agent.action(state, i)

			next_state, rew, done, info = env.step(action)
			next_state = Resampler.total(next_state)

			if not done:
				agent.store(state, action, rew, next_state)

			state = next_state

			if done:
				agent.align()
				break

			if len(agent.replay) > batch_size:
				agent.train(batch_size)

		agent.align()

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
	main()