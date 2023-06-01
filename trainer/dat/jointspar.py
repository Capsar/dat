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
    def get_active_set(self, epoch: int) -> List:
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
        layers_to_sparsify = torch.tensor(self.S[epoch])
        sparsified_grads[layers_to_sparsify] = grads[layers_to_sparsify]
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
    jointspar = JointSpar(num_layers=4, epochs=10, sparsity_budget=3, p_min=0.1)

    print('\nEpoch 0\n')
    jointspar.get_active_set(epoch=0)
    grads0 = torch.tensor([0.2, 0.1, 0.3, 0.4])
    print(f'Grads: {grads0}')
    sparsified0 = jointspar.sparsify_gradient(epoch=0, grads=grads0)
    jointspar.update_p(epoch=0, grads=grads0, learning_rate=0.01)

    print('\nEpoch 1\n')
    jointspar.get_active_set(epoch=1)
    grads1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    print(f'Grads: {grads1}')
    sparsified1 = jointspar.sparsify_gradient(epoch=1, grads=grads1)
    jointspar.update_p(epoch=1, grads=grads1, learning_rate=0.01)

    print('\nEpoch 2\n')
    jointspar.get_active_set(epoch=2)
    grads2 = torch.tensor([0.3, 0.2, 0.2, 0.8])
    print(f'Grads: {grads2}')
    sparsified2 = jointspar.sparsify_gradient(epoch=2, grads=grads2)
    jointspar.update_p(epoch=2, grads=grads2, learning_rate=0.01)
    
    for i in range(3, 9):
        print(f'\nEpoch {i}\n')
        jointspar.get_active_set(epoch=i)
        grads_i = torch.rand(4)
        print(f'Grads: {grads_i}')
        sparsified_i = jointspar.sparsify_gradient(epoch=i, grads=grads_i)
        jointspar.update_p(epoch=i, grads=grads_i, learning_rate=0.01)
