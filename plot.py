import numpy as np
import matplotlib.pyplot as plt


def box_plot(arr):
	# Metrics is a (108, 20, 6) array; batches, batch size, number of metrics
	metrics = np.array(arr)
	# Remove NaN metric
	metrics = np.delete(metrics, 2, axis=2)
	# From metrics, take the mean of each metric in the batch
	metrics = np.mean(metrics, axis=1)

	# Create a box plot of the metrics
	plt.figure(figsize=(8, 6))
	plt.boxplot(metrics, vert=True, patch_artist=True, labels=["Edit distance", "BLEU", "Precision", "Recall", "F-score"])
	plt.ylabel('Scores')
	plt.show()


if __name__ == "__main__":
	test_metrics = np.load("test_metrics.npy")
	box_plot(test_metrics)
