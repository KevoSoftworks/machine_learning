import sys
import os
import getopt

import retro
import numpy as np

import tensorflow.keras as keras

class Memory:
	def __init__(self, size, state_dim, action_dim):
		self.mem_size = size
		self.cnt = 0
		self._rdy = False
		self._full = False

		self.state = np.zeros((self.mem_size, *state_dim))
		self.new_state = np.zeros((self.mem_size, *state_dim))
		self.action = np.zeros(self.mem_size, dtype="uint8")
		self.reward = np.zeros(self.mem_size, dtype="int32")
		self.done = np.zeros(self.mem_size, dtype=bool)

	def __len__(self):
		if self._full:
			return self.mem_size

		return self.cnt

	def store(self, state, action, reward, new_state, done):
		self.state[self.cnt] = state
		self.action[self.cnt] = action
		self.reward[self.cnt] = reward
		self.new_state[self.cnt] = new_state
		self.done[self.cnt] = 1 - int(done)

		self.cnt = (self.cnt + 1) % self.mem_size

		if self.cnt == 0:
			self._full = True

	def read(self, amount, keys=None):
		if keys is None:
			keys = np.random.randint(0, len(self), size=amount)

		states = self.state[keys]
		actions = self.action[keys]
		rewards = self.reward[keys]
		new_states = self.new_state[keys]
		done = self.done[keys]

		return states, actions, rewards, new_states, done

	def ready(self, batch_size):
		if self.cnt > batch_size:
			self._rdy = True

		return self._rdy

class ModelBuilder:
	def __init__(self, layer_nodes, input_dim, output_dim, \
				lr=0.001, loss="mean_squared_error", activation="relu"):
		self.model = keras.Sequential([
			#keras.layers.Conv2D(64, (8, 8), activation=activation, input_shape=(*input_dim, 1)),
			#keras.layers.MaxPool2D(),
			#keras.layers.Conv2D(16, (4, 4), activation=activation),
			#keras.layers.MaxPool2D(),
			keras.layers.Flatten(input_shape=input_dim),
			*[keras.layers.Dense(i, activation=activation) for i in layer_nodes],
			keras.layers.Dense(output_dim, activation=None)
		])

		self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)


class Agent:
	def __init__(self, param, env, model, dims=(None, None), target_model=None):
		self.param = param
		self.state_dim = dims[0]
		self.action_dim = dims[1]

		self.env = env
		self.model = model
		self.target_model = target_model
		self.mem = Memory(self.param.mem_size, self.state_dim, self.action_dim)

	def sync_models(self):
		if self.target_model is None:
			raise Exception("`target_model` not set, cannot sync weights")

		self.target_model.set_weights(self.model.get_weights())

	def action(self, state, epsilon = (0.01, 1)):
		if np.random.rand() <= np.maximum(*epsilon):
			return np.random.randint(self.action_dim), None

		#state = state.reshape((*state.shape, 1))

		q = self.model(state, training=False).numpy().flatten()
		iq_max = np.argmax(q)

		return iq_max, q[iq_max]

	def train(self, batch_size):
		if not self.mem.ready(batch_size):
			return

		states, actions, rewards, new_states, done = self.mem.read(batch_size)

		#states = states.reshape((*states.shape, 1))
		#new_states = new_states.reshape((*states.shape, 1))

		q_now = self.model.predict_on_batch(states)
		if self.target_model is None:
			q_new = self.model.predict_on_batch(new_states)
		else:
			q_new = self.target_model.predict_on_batch(new_states)

		q_target = np.copy(q_now)
		q_target[range(batch_size), actions] = rewards + self.param.discount_factor * np.max(q_new, axis=1) * done

		if batch_size == self.param.batch_size:
			return self.model.train_on_batch(states, q_target)
		else:
			return self.model.fit(states, q_target, batch_size=self.param.batch_size, verbose=self.param.verbose, workers=10, use_multiprocessing=True)


class Resampler:
	@staticmethod
	def to_int(arr):
		return (np.left_shift(arr[:, :, 0], 16) + np.left_shift(arr[:, :, 1], 8) + arr[:, :, 2]) / 0xFFFFFF

	@staticmethod
	def gray(arr):
		return (0.299*arr[:, :, 0] + 0.587*arr[:, :, 1] + 0.114*arr[:, :, 2]) / 255

	@staticmethod
	def drop_sample(arr, n=2):
		return arr[:185:n, ::n]

	@staticmethod
	def reshape(arr):
		return arr.reshape((1, arr.size))

	@staticmethod
	def total(arr):
		#return Resampler.reshape(Resampler.gray(Resampler.drop_sample(arr)))
		return Resampler.reshape(Resampler.drop_sample(arr) / 255)


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
		# (this should be handled by "deathstate", but just in case...)
		if xpos == 0:
			xpos = self.prev_x

		new_score = 0.1*(xpos - self.prev_x) + (score - self.score)

		self.prev_x = xpos
		self.score = score

		return new_score


class Parameters:
	batch_size = 32
	mem_size = 25000

	epsilon_min = 0.05
	epsilon_decay = 0.001
	epsilon = 1
	discount_factor = 0.9
	learning_rate = 0.001

	hold_down = 8

	nodes = (256, 64)

	render = False
	verbose = False
	target = False
	directory = None


class Runner:
	GAME = "TinyToonAdventures-Nes"		# Running custom integration files
	ACTION_SPACE = np.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0],	# Nothing
		[0, 0, 0, 0, 0, 0, 0, 0, 1],	# Jump
		[0, 0, 0, 0, 0, 0, 0, 1, 0],	# Right
		[0, 0, 0, 0, 0, 0, 1, 0, 0],	# Left
		[0, 0, 0, 0, 0, 1, 0, 0, 0],	# Down
		[1, 0, 0, 0, 0, 0, 0, 0, 1],	# Run jump
		[1, 0, 0, 0, 0, 0, 0, 1, 0],	# Run right
		[1, 0, 0, 0, 0, 0, 1, 0, 0],	# Run left
	])

	def __init__(self, param):
		self.param = param
		self.env = retro.make(game=self.GAME)

		state = Resampler.total(self.env.reset())

		mb = ModelBuilder(self.param.nodes, state.shape, self.ACTION_SPACE.shape[0], self.param.learning_rate)
		target_mb = ModelBuilder(self.param.nodes, state.shape, self.ACTION_SPACE.shape[0], self.param.learning_rate)
		if not self.param.target:
			target_mb.model = None

		self.agent = Agent(self.param, self.env, mb.model, (state.shape, self.ACTION_SPACE.shape[0]), target_model=target_mb.model)
		self.episode = 0

	def load(self):
		with open(f"{self.param.directory}/stats.csv", "rb") as file:
			file.seek(-2, os.SEEK_END)
			while file.read(1) != b"\n":
				file.seek(-2, os.SEEK_CUR)
			latest = file.readline().decode()

		vals = latest.split(",")
		self.episode = int(vals[0])
		self.param.epsilon = float(vals[3])

		self.agent.model = keras.models.load_model(f"{self.param.directory}/model")
		if self.param.target is not None:
			self.agent.sync_models()

	def save(self, ticks, avg_loss, score):
		if self.param.directory is None:
			return

		if self.param.verbose:
			print(f"Saving training data in {self.param.directory}")

		self.agent.model.save(f"{self.param.directory}/model")
		with open(f"{self.param.directory}/stats.csv", "a") as file:
			file.write(f"{self.episode},{ticks},{score},{self.param.epsilon},{avg_loss}\n")

	def run(self, cnt=-1):
		cond = lambda i: i < cnt if cnt > -1 else True

		while cond(self.episode):
			rew_handler = Reward()
			state = Resampler.total(self.env.reset())

			score = 0
			ticks = 0
			total_loss = 0
			done = False

			while not done:
				tick_reward = 0
				action, q_val = self.agent.action(state, (self.param.epsilon_min, self.param.epsilon))

				if self.param.render:
					self.env.render()

				# Hold button down
				for _ in range(self.param.hold_down):
					next_state, _, done, info = self.env.step(self.ACTION_SPACE[action])
					rew = rew_handler.compute_reward(info)

					tick_reward += rew

				next_state = Resampler.total(next_state)

				if info.get("deathstate") == 3 or ticks >= 250:
					if info.get("deathstate") == 3:
						tick_reward -= 0
					done = True

				self.agent.mem.store(state, action, tick_reward, next_state, done)
				loss = self.agent.train(self.param.batch_size)

				state = next_state
				score += tick_reward
				if loss is not None:
					total_loss += loss
				ticks += 1

				if self.param.verbose:
					print(f"Action={action}, predicted_q={q_val}, tick_rew={tick_reward}, score={score}, info={info}, loss={loss}")

			if self.param.target and self.agent.mem.ready(self.param.batch_size):
				self.agent.sync_models()

			print(f"Epsiode {self.episode}; Score: {score}; Epsilon: {self.param.epsilon}; Avg loss {total_loss / ticks}")

			self.episode += 1
			if self.param.epsilon > self.param.epsilon_min:
				self.param.epsilon -= self.param.epsilon_decay
				if self.param.epsilon < self.param.epsilon_min:
					self.param.epsilon = self.param.epsilon_min

			self.save(ticks, total_loss / ticks, score)


def main(argv):
	ACTION = None
	try:
		opts, args = getopt.getopt(argv[1:], "", [ \
			"new",
			"verbose",
			"render",
			"load",
			"save",

			"batch-size=",
			"epsilon=",
			"epsilon-min=",
			"epsilon-decay=",
			"hold-down=",
			"memory-size=",
			"learning-rate=",
			"discount=",

			"target"
		])
	except getopt.GetoptError as e:
		print(f"Unknown argument `{e.opt}`. Please refer to `{argv[0]} --help` for available arguments.")
		sys.exit(1)

	param = Parameters()

	for opt, arg in opts:
		if opt == "--new":
			ACTION = "new"
		elif opt == "--load":
			ACTION = "load"
		elif opt == "--save":
			if len(args) == 0:
				print(f"Usage: {argv[0]} --save [--opts] directory")
				sys.exit(1)

			param.directory = args[0]
		elif opt == "--verbose":
			param.verbose = True
		elif opt == "--render":
			param.render = True

		elif opt == "--batch-size":
			param.batch_size = int(arg)
		elif opt == "--epsilon":
			param.epsilon = float(arg)
		elif opt == "--epsilon-min":
			param.epsilon_min = float(arg)
		elif opt == "--epsilon-decay":
			param.epsilon_decay = float(arg)
		elif opt == "--hold-down":
			param.hold_down = int(arg)
		elif opt == "--memory-size":
			param.mem_size = int(arg)
		elif opt == "--learning-rate":
			param.learning_rate = float(arg)
		elif opt == "--discount":
			param.discount_factor = float(arg)

		elif opt == "--target":
			param.target = True

	runner = Runner(param)
	if ACTION is None:
		print(f"No action provided. Please refer to `{argv[0]} --help` for available actions.")
		sys.exit(1)
	elif ACTION == "new":
		# Do nothing
		pass
	elif ACTION == "load":
		if param.directory is None:
			if len(args) == 0:
				print(f"Usage: {argv[0]} --load [--opts] directory")
				sys.exit(1)

			param.directory = args[0]

		runner.load()

	runner.run()

if __name__ == "__main__":
	main(sys.argv)