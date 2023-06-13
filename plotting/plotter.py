import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List



def plot_p(p: List[torch.Tensor]):
    print(f't: {p}')

    # Convert the list of tensors to a NumPy array
    data = np.array([t.numpy() for t in p]).T

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Create the heatmap using seaborn
    heatmap = sns.heatmap(data, cmap='viridis', ax=ax)

    # Set axis labels
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')

    # Set title
    ax.set_title('Heatmap of Tensors')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    ps = [torch.rand(54) for _ in range(5)]
    plot_p(ps)
