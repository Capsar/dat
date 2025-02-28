import time
import pickle
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import List
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import KLDivLoss

from trainer.dat.dataset import Cifar_EXT
from trainer.dat.helpers import send_telegram_message
from trainer.dat.params import Params
from trainer.dat.utils import seconds_to_string
from trainer.dat.models import PreActResNet18

timing_dict = {
    'get_active_set': {
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
    def __init__(self, num_layers: int, epochs: int, sparsity_budget: int, p_min: int):
        assert sparsity_budget <= num_layers, 'Sparsity budget must be smaller or equal to number of layers.'
        self.num_layers = num_layers
        self.epochs = epochs
        self.sparsity = sparsity_budget
        self.p_min = torch.tensor(p_min / sparsity_budget, dtype=torch.float32)
        self.p = torch.zeros((self.epochs + 1, self.num_layers))

        p0 = 1 / self.num_layers
        
        self.p[0, :] = torch.full((self.num_layers,), p0)
        self.Z = torch.zeros((self.epochs, self.num_layers))
        self.S = [[]] * self.epochs
        self.L = torch.zeros(1)
        self.KL = KLDivLoss(reduction='sum')

    # Step 2 & 3
    @time_jointspar(name='get_active_set')
    def get_active_set(self, epoch: int) -> List:
        # print(f'Sparsity budget: {self.sparsity}')
        # print(f'p min: {self.p_min}')
        # print(f'Current p: {self.p[epoch, :]}')
        sample = torch.rand(self.num_layers)
        #print(f'Pytorch sample: {sample}')
        self.Z[epoch, :] = torch.where(sample < self.p[epoch, :] * self.sparsity, torch.tensor(1.0), torch.tensor(0.0))
        #print(f'{epoch}. Z: {self.Z[epoch, :]}')
        self.S[epoch] = torch.nonzero(self.Z[epoch, :] == 1.0).flatten().tolist()
        #print(f'{epoch}. S: {self.S[epoch]}')
        return self.S[epoch]
    
    # Algo 1
    @time_jointspar(name='update_p')
    def update_p(self, epoch: int, grads, learning_rate: float):
        l = torch.Tensor(self.num_layers, device='cpu')
        w = torch.Tensor(self.num_layers)
        active_set = self.S[epoch]
                
        self.L = torch.max(self.L.detach().cpu(), torch.max(torch.stack([torch.norm(g).detach().cpu() for i,g in enumerate(grads) if i in active_set])))

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
        # w.where(w < self.p_min, torch.tensor(self.p_min))
        # print(f'w: {w}')

        #self.p[epoch + 1] = w * (self.sparsity / torch.sum(w))
        self.p[epoch + 1] = self.optimize_distribution(w)

    def optimize_distribution(self, p):
        # Convert p to a PyTorch tensor
        p_tensor = p.clone().detach()
        p_tensor= torch.clip(p_tensor, self.p_min, 1.0)

        #print(f'Before update p: {p_tensor}')
        
        # Define the parameters as variables to optimize
        n = len(p)
        q = torch.nn.Parameter(torch.ones(n) / n)

        # Define the optimizer
        optimizer = optim.LBFGS([q])

        # Define the closure function for the optimizer
        def closure():
            optimizer.zero_grad()
            kl_divergence = self.KL(torch.log(q), p_tensor)
            kl_divergence.backward()
            return kl_divergence

        # Perform optimization
        optimizer.step(closure)

        # Normalize q to sum up to 1
        q_normalized = q.detach() / q.sum().detach()

        # Apply the minimum value constraint
        q_final = torch.clip(q_normalized, self.p_min, 1.0)

        #print(f'After update p: {q_final}')

        #print(f'q: {q_final * self.sparsity}')
        return q_final #* self.sparsity

#Just to write it down somewhere, we should maybe change the way we pick which layers are taken and which are not, as atm its truely random leading to maybe zero layers being taken
#we could do smth like "take the #budget layers with probabilites given"?


def main(params: Params, plot=False):
    #send_telegram_message(message=f'Starting run with params: {str(params)}')
    epochs = params.epochs
    batch_size = params.batch_size
    sparsity_budget = params.sparsity_budget
    p_min = params.p_min
    learning_rate = params.learning_rate
    use_cifarext = params.use_cifarext
    use_jointspar = params.use_jointspar

    print(f'Using JointSPAR: {use_jointspar}')
    print(f'Using CifarEXT: {use_cifarext}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PreActResNet18().to(device)

    num_layers = sum(1 for _ in model.parameters())
    print(f'Number of layers: {num_layers}')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if use_cifarext:
        trainloader, testloader, train_sampler = Cifar_EXT.get_local_loader(batch_size, './data')
    else:
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
            print(f'p: {jointspar.p[epoch, :]}')

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if i != 0 or epoch != 0:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        if use_jointspar:
            sparsified_grads = []
            for ind, p in enumerate(model.parameters()):
                if ind in S:
                    curr_p = 0 if p.grad is None else p.grad
                    if p.grad is None and epoch > 0:
                        print(f'{epoch} {i} {ind} p.grad is None!')
                    sparsified_grads.append(curr_p / jointspar.p[epoch, ind])
                else:
                    sparsified_grads.append([])
            jointspar.update_p(epoch, sparsified_grads, learning_rate)

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

    file_name = params.get_filename()
    with open(file_name, 'wb') as file:
        pickle.dump([losses, accuracies, times_per_epoch, params], file)
    print('File saved!')

    if plot:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        plt.plot(accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.show()

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
    time_str = seconds_to_string(total_time)
    print(f'Finished Training in {time_str}')
    #send_telegram_message(message=f'Done with run in {time_str} Final accuracy: {accuracies[-1]:.2f}%')


if __name__ == '__main__':
    #methods for the model to inclue
    # - get_num_layers
    # - freeze_layers   (freeze layers that are not in the active set)
    # - unfreeze_layers


    # If p_min is not in Params object its value is 0.005
    param_list = [
        Params(
            epochs=100,
            batch_size=256,
            sparsity_budget=40,
            p_min=p_min,
            learning_rate=0.01,
            use_cifarext=False,
            use_jointspar=True,
        )
        for p_min in [0.5, 0.05, 0.005, 0.0005]
    ]
    for ind, p in enumerate(param_list):
        main(p)
        #send_telegram_message(f'Remaining runs: {len(param_list) - (ind + 1)}/{len(param_list)}\n\n')


### Cifar Batch 2048 Epochs 100 Sparsity 30: 17M:07S JointSPAR: False
### Cifar Batch 2048 Epochs 100 Sparsity 30: 14M:23S JointSPAR: True