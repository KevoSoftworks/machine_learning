import sys

import numpy as np
import matplotlib.pyplot as plt

def mov_avg(arr, n):
	return np.convolve(arr, np.ones(n), "valid") / n

def main(argv):
	if len(argv) != 2:
		print(f"Usage: {argv[0]} directory")

	directory = argv[1]

	# data: episode,ticks,score,epsilon,avg_loss
	data = np.loadtxt(f"{directory}/stats.csv", delimiter=",")

	plt.figure()
	plt.plot(data[:,0], data[:,1], label="Raw")
	plt.plot(data[:-4,0], mov_avg(data[:,1], 5), label="5 episode moving average")
	plt.xlabel("Episodes")
	plt.ylabel("Ticks")
	plt.title("Ticks per episode")
	plt.legend()

	plt.figure()
	plt.plot(data[:,0], data[:,2], label="Raw")
	plt.plot(data[:-4,0], mov_avg(data[:,2], 5), label="5 episode moving average")
	plt.xlabel("Episodes")
	plt.ylabel("Score")
	plt.title("Score per episode")
	plt.legend()

	plt.figure()
	plt.plot(data[:,0], data[:,3], label="Raw")
	plt.plot(data[:-4,0], mov_avg(data[:,3], 5), label="5 episode moving average")
	plt.xlabel("Episodes")
	plt.ylabel("Epsilon")
	plt.title("Random chance per episode")
	plt.legend()

	plt.figure()
	plt.plot(data[:,0], data[:,4], label="Raw")
	plt.plot(data[:-4,0], mov_avg(data[:,4], 5), label="5 episode moving average")
	plt.xlabel("Episodes")
	plt.ylabel("Average loss")
	plt.title("Loss per episode")
	plt.legend()

	plt.show()

if __name__ == "__main__":
	main(sys.argv)