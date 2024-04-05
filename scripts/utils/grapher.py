from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['font.size'] = 12
# plt.rcParams['savefig.format'] = 'pdf'
sns.set_style('dark')


def plot_line(values: List, save_path: str):
    sns.lineplot(x=range(len(values)), y=values)
    plt.savefig(save_path, bbox_inches='tight')
