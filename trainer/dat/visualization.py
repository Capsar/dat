import os
import pickle
from typing import Tuple, List

import matplotlib.pyplot as plt

from trainer.dat.params import Params

dir_name = './runs/cifar/100epochs'


def get_run_info(file_name: str) -> Tuple[List, List, List, List]:
    with open(f'{dir_name}/{file_name}', 'rb') as f:
        return pickle.load(f)


# 03H42M-11062023_100epochs_2048batch_30sparsity_0.01lr_jointspar_cifarext.obj
# 100epochs_2048batch_cifarext.obj

def convert_file_name(file_name: str, compare_key: str, min_float=None) -> str:
    file_name = file_name.replace('.obj', '')
    parts = file_name.split('_')
    print(parts)
    num_epochs = parts[0].replace('epochs', '')
    batch_size = parts[1].replace('batch', '')
    jointspar_str = f' - JointSPAR' if len(parts) > 2 else ''

    for part in parts:
        if compare_key in part:
            val = part.split(compare_key)[0]

            if min_float and float(val) < min_float:
                return None

            return f'{val} {compare_key}'
    return 'Baseline'

    #return f'Batch Size: {batch_size} Epochs: {num_epochs}{jointspar_str}'


if __name__ == '__main__':
    files = [
        file_name for file_name in os.listdir(dir_name) if '.obj' in file_name
    ]
    # files = [
    #     '20epochs_2048batch_30sparsity_cifarext.obj',
    #     '20epochs_2048batch_30sparsity_jointspar_cifarext.obj',
    #     '15H24M-10062023_20epochs_2048batch_35sparsity_0.01lr_jointspar_cifarext.obj',
    #     '15H52M-10062023_20epochs_2048batch_40sparsity_0.01lr_jointspar_cifarext.obj'
    # ]


    loss_list = []
    accuracy_list = []
    times_list = []
    cumul_times_list = []
    file_names = []

    for file in files:
        if '.obj' not in file:
            continue
        res = get_run_info(file)
        print(file, len(res))
        if len(res) == 3:
            losses, accuracies, times = res
        else:
            params: Params
            losses, accuracies, times, params = res

        file_name = convert_file_name(file, compare_key='sparsity', min_float=40)
        if file_name is None:
            continue
        file_names.append(file_name)

        loss_list.append(losses)
        accuracy_list.append(accuracies)
        times_list.append(times)

        t0 = 0
        cumul_list = []
        for t in times:
            t0 += t
            cumul_list.append(t0)
        cumul_times_list.append(cumul_list)

    # Loss plot
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for ind, l in enumerate(loss_list):
        plt.plot(l, label=file_names[ind])
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    for ind, accs in enumerate(accuracy_list):
        plt.plot(accs, label=file_names[ind])

    plt.legend()
    plt.show()

    # Accuracy plot over time in seconds
    plt.title('Training Accuracy Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    for ind in range(len(accuracy_list)):
        plt.plot(cumul_times_list[ind], accuracy_list[ind], label=file_names[ind])
    plt.legend()
    plt.show()

    # Training times plot
    plt.title('Training Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    for ind, times in enumerate(times_list):
        plt.plot(times, label=file_names[ind])
    plt.legend()
    plt.show()


def filename_to_label(file_name: str) -> str:
    parts = file_name.split('_')
    print(parts)


