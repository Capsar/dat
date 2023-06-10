import os
import pickle
from typing import Tuple, List

import matplotlib.pyplot as plt

dir_name = './runs'


def get_run_info(file_name: str) -> Tuple[List, List, List]:
    with open(f'{dir_name}/{file_name}', 'rb') as f:
        return pickle.load(f)


def convert_file_name(file_name: str) -> str:
    file_name = file_name.replace('.obj', '')
    parts = file_name.split('_')
    num_epochs = parts[0].replace('epochs', '')
    batch_size = parts[1].replace('batch', '')
    jointspar_str = f' - JointSPAR' if len(parts) > 2 else ''

    return f'Batch Size: {batch_size} Epochs: {num_epochs}{jointspar_str}'


if __name__ == '__main__':
    files = []
    for file_name in os.listdir(dir_name):
        if '.obj' not in file_name:
            continue
        files.append(file_name)
        print(f'File: {file_name}')
        print(convert_file_name(file_name))

    files = [
        '100epochs_2048batch.obj',
        '100epochs_2048batch_cifarext.obj'
    ]


    loss_list = []
    accuracy_list = []
    times_list = []

    for file in files:
        if '.obj' not in file:
            continue
        losses, accuracies, times = get_run_info(file)
        loss_list.append(losses)
        accuracy_list.append(accuracies)
        times_list.append(times)

    # Loss plot
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for ind, l in enumerate(loss_list):
        plt.plot(l, label=files[ind])
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    for ind, accs in enumerate(accuracy_list):
        plt.plot(accs, label=files[ind])
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.title('Training Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    for ind, times in enumerate(times_list):
        plt.plot(times, label=files[ind])
    plt.legend()
    plt.show()
