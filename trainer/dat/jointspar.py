import time
from typing import List

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np 

from utils import seconds_to_string
from torchvision.datasets import CIFAR10
from models import PreActResNet18, PreActBlock, GlobalpoolFC
from torchvision.transforms import ToTensor
from torch.nn import KLDivLoss

timing_dict = {
    'get_active_set': {
        'time': 0.0,
        'calls': 0,
        'time_per_call': 0.0
    },
    'sparsify_gradient': {
        'time': 0.0,
        'calls': 0,
        'time_per_call': 0.0
    },
    'update_p': {
        'time': 0.0,
        'calls': 0,
        'time_per_call': 0.0
    }
}

def time_jointspar(name):
    def decorator_func(func):
        def wrapper(*args, **kwargs):
            start1 = time.perf_counter()
            func(*args)
            delta1 = time.perf_counter() - start1
            timing_dict[name]['calls'] += 1
            timing_dict[name]['time'] += delta1
            timing_dict[name]['time_per_call'] = timing_dict[name]['time'] / timing_dict[name]['calls']
            return func(*args, **kwargs)
        return wrapper
    return decorator_func


class JointSpar:
    def __init__(self, num_layers: int, epochs: int, sparsity_budget: int, p_min: float):
        assert sparsity_budget <= num_layers, 'Sparsity budget must be smaller or equal to number of layers.'
        self.num_layers = num_layers
        self.epochs = epochs
        self.sparsity = sparsity_budget
        self.p_min = torch.tensor(p_min)
        self.p = torch.zeros((self.epochs + 1, self.num_layers))
        p0 = self.sparsity / self.num_layers
        self.p[0, :] = torch.full((self.num_layers,), p0)
        self.Z = torch.zeros((self.epochs, self.num_layers))
        self.S = [[]] * self.epochs
        self.L = torch.zeros(1)

    # Step 2 & 3
    @time_jointspar(name='get_active_set')
    def get_active_set(self, epoch: int) -> List:
        # print(f'Sparsity budget: {self.sparsity}')
        # print(f'p min: {self.p_min}')
        # print(f'Current p: {self.p[epoch, :]}')
        sample = torch.rand(self.num_layers)
        #print(f'Pytorch sample: {sample}')
        self.Z[epoch, :] = torch.where(sample < self.p[epoch, :], torch.tensor(1.0), torch.tensor(0.0))
        #print(f'{epoch}. Z: {self.Z[epoch, :]}')
        self.S[epoch] = torch.nonzero(self.Z[epoch, :] == 1.0).flatten().tolist()
        #print(f'{epoch}. S: {self.S[epoch]}')
        return self.S[epoch]
    
    # Step 5 & 6
    @time_jointspar(name='sparsify_gradient')
    def sparsify_gradient(self, epoch: int, grads: torch.Tensor) -> torch.Tensor:
        # Set L
        self.L = torch.max(self.L, max([torch.norm(g) for g in grads]))
        # 5: normalize
        grads = grads / self.p[epoch]

        # 6: construct sparsified gradient
        sparsified_grads = torch.zeros_like(grads)
        layers_to_compute = torch.tensor(self.S[epoch])
        #print(f'Layers to compute: {layers_to_compute}')
        sparsified_grads[layers_to_compute] = grads[layers_to_compute]
        #print(f'Sparsified grads: {sparsified_grads}')
        return sparsified_grads
    
    # Algo 1
    @time_jointspar(name='update_p')
    def update_p(self, epoch: int, grads, learning_rate: float):
        l = torch.Tensor(self.num_layers, device='cpu')
        w = torch.Tensor(self.num_layers)

        active_set = self.S[epoch]
        for d in range(self.num_layers):
            if d in active_set:
                g = torch.zeros_like(grads[d])
                g = torch.zeros_like(torch.stack([g]*self.Z.shape[1]))
                g[d] = grads[d].detach().cpu()
                l1 = -torch.pow(g.norm(), 2) / torch.pow(self.p[epoch, d], 2)
                l2 = torch.pow(self.L, 2) / torch.pow(self.p_min, 2)
                l[d] = l1.detach().cpu() + l2.detach().cpu()
            else:
                l[d] = 0
            w[d] = self.p[epoch, d] * torch.exp((-learning_rate * l[d]) / self.p[epoch, d])

        # print(f'L: {self.L}')
        # print(f'l: {l}')
        # print(f'w: {w}')

        # line 10: to bring it back to a probability distribution but with a sum of sparsity budget
        w.where(w < self.p_min, torch.tensor(self.p_min))
        # print(f'w: {w}')

        self.p[epoch + 1] = w * (self.sparsity / torch.sum(w))


def set_grad_enabled_for_layers(model, num_layers, active_layers):
    for ind in range(num_layers):
        layer = model.layers[ind]

        requires_grad = ind in active_layers

        if isinstance(layer, PreActBlock):
            layer.bn1.requires_grad_(requires_grad)
            layer.bn2.requires_grad_(requires_grad)
            layer.conv1.requires_grad_(requires_grad)
            layer.conv2.requires_grad_(requires_grad)
        elif isinstance(layer, GlobalpoolFC):
            layer.fc.requires_grad_(requires_grad)
        else:
            layer.requires_grad_(requires_grad)


def get_layer_gradients(layer) -> torch.Tensor:
    if isinstance(layer, PreActBlock):
        return torch.Tensor([get_layer_safe(layer.bn1),
                             get_layer_safe(layer.conv1),
                             get_layer_safe(layer.bn2),
                             get_layer_safe(layer.conv2)])
    elif isinstance(layer, GlobalpoolFC):
        return get_layer_safe(layer.fc)
    return get_layer_safe(layer)

def get_layer_safe(layer) -> torch.Tensor:
    if layer.weight.grad is None:
        return torch.zeros(1)
    return layer.weight.grad

def get_model_gradients(model: PreActResNet18, active_layers) -> torch.Tensor:
    return torch.Tensor([get_layer_gradients(l) if i in active_layers else torch.zeros(1) for i,l in enumerate(model.layers)])

#Just to write it down somewhere, we should maybe change the way we pick which layers are taken and which are not, as atm its truely random leading to maybe zero layers being taken
#we could do smth like "take the #budget layers with probabilites given"?


if __name__ == '__main__':
    #methods for the model to inclue
    # - get_num_layers
    # - freeze_layers   (freeze layers that are not in the active set)
    # - unfreeze_layers

    epochs = 5
    batch_size = 512
    sparsity_budget = 30
    p_min = 0.1
    learning_rate = 0.01
    use_jointspar = True

    print(f'Using JointSPAR: {use_jointspar}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PreActResNet18().to(device)

    num_layers = sum(1 for _ in model.parameters())
    print(f'Number of layers: {num_layers}')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    trainset = CIFAR10(root='./data', train=True, download=False, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = CIFAR10(root='./data', train=False, download=False, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    jointspar = JointSpar(
        num_layers=num_layers,
        epochs=epochs,
        sparsity_budget=sparsity_budget,
        p_min=p_min,
    )

    losses = []
    accuracies = []
    times_per_epoch = []
    active_layers_per_epoch = []

    start = time.perf_counter()

    for epoch in range(jointspar.epochs):
        epoch_start = time.perf_counter()
        if use_jointspar:
            S = jointspar.get_active_set(epoch)
            for i, p in enumerate(model.parameters()):
                p.requires_grad_(i in S)

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if i != 0 or epoch != 0:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            if use_jointspar:
                sparsified_grads=[]
                for i, p in enumerate(model.parameters()):
                    if i in S:
                        curr_p = 0 if p.grad is None else p.grad
                        sparsified_grads.append(curr_p / jointspar.p[epoch, i])
                    else:
                        sparsified_grads.append([])
                jointspar.update_p(epoch, sparsified_grads, learning_rate)

            loss.backward()
            optimizer.step()

            # for i, p in enumerate(model.parameters()):
            #     if p.grad is None:
            #         print('AFTER', i, p.grad, p.requires_grad)

            # if i % 100 == 0:
            #     print(f'Epoch {epoch}, batch {i}, loss {loss.item()}')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_time = time.perf_counter() - epoch_start
        times_per_epoch.append(epoch_time)
        if use_jointspar:
            active_layers_per_epoch.append(len(S))

        accuracy = correct / total
        accuracies.append(accuracy * 100)
        delta = time.perf_counter() - start
        time_per_epoch = delta / (epoch + 1)
        eta = (epochs - (epoch + 1)) * time_per_epoch
        active_layers_str = f'active layers = {len(S)}' if use_jointspar else ''
        print(f'Epoch {epoch + 1} / {epochs}, accuracy {accuracy * 100:.2f}% ETA: {seconds_to_string(eta)} '
              f'Time/Epoch: {time_per_epoch:.2f}s {active_layers_str}')
        if use_jointspar:
            print(timing_dict)

    jointspar_suffix = '_jointspar' if use_jointspar else ''
    file_name = f'./runs/{epochs}epochs_{batch_size}batch{jointspar_suffix}.obj'
    with open(file_name, 'wb') as file:
        pickle.dump([losses, accuracies, times_per_epoch], file)

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

    # plt.plot(times_per_epoch)
    # plt.xlabel('Epoch')
    # plt.ylabel('Time (s)')
    # plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(times_per_epoch, color='blue')
    ax1.set_ylabel('Time (s)', color='blue')

    if use_jointspar:
        ax2 = ax1.twinx()
        ax2.plot(active_layers_per_epoch, color='red')
        ax2.set_ylabel('Active Layers', color='red')

        # Optionally, set the limits and formatting for the second y-axis
        ax2.set_ylim(0, 350)
        ax2.yaxis.set_tick_params(color='red')

    # Show the plot
    plt.show()

    total_time = time.perf_counter() - start
    print(f'Finished Training in {seconds_to_string(total_time)}')
