import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use('classic')


def main():
	data = np.array([
		[40000, 76.40, 46.40, 8.80, 79.40, 10.80, 32.20, 44.00, 82.60],
		[67000, 84.00, 37.86, 8.57, 73.14, 9.57, 49.71, 53.71, 89.00],
		[120000, 84.50, 45.50, 8.50, 96.20, 9.90, 80.00, 61.00, 95.70],
		[305000, 90.07, 57.40, 38.47, 98.20, 10.47, 89.80, 62.13, 98.13],
		[475000, 93.36, 45.84, 81.00, 93.52, 10.12, 98.52, 88.40, 99.36],
		[630846, 97.87, 52.97, 97.07, 99.84, 11.04, 99.69, 96.05, 98.98],
	])

	data = pd.DataFrame(data, columns=['Samples', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])
	data.index = data.Samples
	data = data.iloc[:, 1:]
	fig = plt.figure(figsize=(6, 3), constrained_layout=True)
	ax = sns.lineplot(data=data, markers=True, dashes=False)
	ax.get_legend().remove()
	ax.tick_params(color='k', labelcolor='k', labelsize=10)
	ax.set(xlabel=None)
	ax.grid()
	#ax.set_xticks([0, 50000, 100000, 250000, 500000, 600000])
	ax.set_xticklabels(['0', '100k', '200k', '300k', '400k', '500k', '600k' , '700k'])
	fig.supxlabel('# Samples', fontsize=12, color='k')
	fig.supylabel('Accuracy (%)', fontsize=12, color='k')
	fig.legend(loc=8, ncol=8, prop={'size': 8})
	plt.savefig('space_vs_acc.eps', dpi=100, bbox_inches='tight', pad_inches=0.3)


if __name__ == '__main__':
	main()
