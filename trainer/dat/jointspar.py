from typing import List

import torch


class JointSpar:
    def __init__(self, num_layers: int, epochs: int, sparsity_budget: float, p_min: float):
        assert sparsity_budget <= num_layers, f'Sparsity budget must be smaller or equal to number of layers.'
        self.num_layers = num_layers
        self.epochs = epochs
        self.sparsity = sparsity_budget
        self.p_min = p_min
        self.p = torch.zeros((self.epochs, self.num_layers))
        p0 = self.sparsity / self.num_layers
        self.p[0, :] = torch.full((self.num_layers,), p0)
        self.Z = torch.zeros((self.epochs, self.num_layers))
        self.S = [[]] * self.epochs
        print(f'S: {self.S}')

    def get_active_set(self, epoch: int) -> List:
        print(f'Sparsity budget: {self.sparsity}')
        print(f'p min: {self.p_min}')
        print(f'Current p: {self.p[epoch, :]}')
        sample = torch.rand(self.num_layers)
        print(f'Pytorch sample: {sample}')
        self.Z[epoch, :] = torch.where(sample < self.p[epoch, :], torch.tensor(0.0), torch.tensor(1.0))
        print(f'First Z: {self.Z[epoch, :]}')
        self.S[epoch] = torch.nonzero(self.Z[epoch, :] == 1.0).flatten().tolist()
        print(f'First S: {self.S[epoch]}')
        return self.S[epoch]


if __name__ == '__main__':
    jointspar = JointSpar(num_layers=4, epochs=10, sparsity_budget=3, p_min=0.1)
    jointspar.get_active_set(epoch=0)
