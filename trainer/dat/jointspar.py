import time
from typing import List

import torch


class JointSpar:
    def __init__(self, num_layers: int, epochs: int, sparsity_budget: float, p_min: float):
        assert sparsity_budget <= num_layers, 'Sparsity budget must be smaller or equal to number of layers.'
        self.num_layers = num_layers
        self.epochs = epochs
        self.sparsity = sparsity_budget
        self.p_min = torch.tensor(p_min)
        self.p = torch.zeros((self.epochs, self.num_layers))
        p0 = self.sparsity / self.num_layers
        self.p[0, :] = torch.full((self.num_layers,), p0)
        self.Z = torch.zeros((self.epochs, self.num_layers))
        self.S = [[]] * self.epochs
        self.L = torch.zeros(1)

    # Step 2 & 3
    def get_active_set(self, epoch: int) -> torch.Tensor:
        print(f'Sparsity budget: {self.sparsity}')
        print(f'p min: {self.p_min}')
        print(f'Current p: {self.p[epoch, :]}')
        sample = torch.rand(self.num_layers)
        print(f'Pytorch sample: {sample}')
        self.Z[epoch, :] = torch.where(sample < self.p[epoch, :], torch.tensor(1.0), torch.tensor(0.0))
        print(f'{epoch}. Z: {self.Z[epoch, :]}')
        self.S[epoch] = torch.nonzero(self.Z[epoch, :] == 1.0).flatten().tolist()
        print(f'{epoch}. S: {self.S[epoch]}')
        return self.S[epoch]
    
    # Step 5 & 6 
    def sparsify_gradient(self, epoch: int, grads: torch.Tensor) -> torch.Tensor:
        # Set L
        self.L = torch.max(self.L, torch.max(grads))
        # 5: normalize
        grads = grads / self.p[epoch]
        
        # 6: construct sparsified gradient
        sparsified_grads = torch.zeros_like(grads)
        layers_to_compute = torch.tensor(self.S[epoch])
        sparsified_grads[layers_to_compute] = grads[layers_to_compute]
        print(f'Sparsified grads: {sparsified_grads}')
        return sparsified_grads
    
    # Algo 1
    def update_p(self, epoch: int, grads: torch.Tensor, learning_rate: float):
        l = torch.Tensor(self.num_layers)
        w = torch.Tensor(self.num_layers)

        active_set = self.S[epoch]
        for d in range(self.num_layers):
            if d in active_set:
                g = torch.zeros_like(grads)
                g[d] = grads[d]
                l1 = -torch.pow(g.norm(), 2) / torch.pow(self.p[epoch, d], 2)
                l2 = torch.pow(self.L, 2) / torch.pow(self.p_min, 2)
                l[d] = l1 + l2
            else:
                l[d] = 0
            w[d] = self.p[epoch, d] * torch.exp((-learning_rate * l[d]) / self.p[epoch, d])

        print(f'L: {self.L}')
        print(f'l: {l}')
        print(f'w: {w}')
        # line 10: to bring it back to a probability distribution but with a sum of sparsity budget
        self.p[epoch + 1] = (w / torch.sum(w)) * self.sparsity


if __name__ == '__main__':

    import torch
    from torchvision.datasets import CIFAR10
    from models import PreActResNet18
    from torchvision.transforms import ToTensor

    #methods for the model to inclue
    # - get_num_layers
    # - freeze_layers   (freeze layers that are not in the active set)
    # - unfreeze_layers 

    jointspar = JointSpar(num_layers=3, epochs=15, sparsity_budget=2, p_min=0.1)

    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PreActResNet18().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    trainset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)

    testset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    losses= []
    accuracies= []

    for epoch in range(jointspar.epochs):
        S=jointspar.get_active_set(epoch)
        for i, data in enumerate(trainloader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            #jointspar.update_p(epoch, model[0].weight.grad, 0.01)
            # sparsified_grad= jointspar.sparsify_gradient(epoch, model.layers[0].weight.grad)
            # model.layers[0].weight.grad= sparsified_grad
            # for i in sparsified_grad:
            #     if i == 0:
            #         print(f'Freeze layer {i}')

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}, batch {i}, loss {loss.item()}')

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
        
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch}, accuracy {accuracy}')


    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()

    plt.plot(accuracies)
    plt.show()

    print('Finished Training')


