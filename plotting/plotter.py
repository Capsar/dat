import torch
import csv
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List


def fetch_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        p, Z, S, L = pickle.load(f)
    return p, Z, S, L


def read_file():
    with open('./wandb_epochpv2.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for ind, (step, data) in enumerate(reader):
            json_data = json.loads(data)
            print(ind, step, len(json_data['bins']), json_data['bins'])


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
    # ps = [torch.rand(54) for _ in range(5)]
    # plot_p(ps)
    #read_file()
    print(fetch_pickle(file_path='./jointspar_workerpool0-0-DR0-LR0-R0.obj'))