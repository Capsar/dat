from typing import List

import torch


class JointSpar:
    def __init__(self, num_layers: int, epochs: int, sparsity_budget: float, p_min: float):
        assert sparsity_budget <= num_layers, f'Sparsity budget must be smaller or equal to number of layers.'
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
        print(f'S: {self.S}')

    #Step 2 & 3
    def get_active_set(self, epoch: int) -> List:
        print(f'Sparsity budget: {self.sparsity}')
        print(f'p min: {self.p_min}')
        print(f'Current p: {self.p[epoch, :]}')
        sample = torch.rand(self.num_layers)
        print(f'Pytorch sample: {sample}')
        self.Z[epoch, :] = torch.where(sample < self.p[epoch, :], torch.tensor(0.0), torch.tensor(1.0))
        print(f'{epoch}. Z: {self.Z[epoch, :]}')
        self.S[epoch] = torch.nonzero(self.Z[epoch, :] == 1.0).flatten().tolist()
        print(f'{epoch}. S: {self.S[epoch]}')
        return self.S[epoch]
    
    # Step 5 & 6 
    def sparsify_gradient(self, epoch: int, grads: torch.Tensor) -> torch.Tensor:
        # set L
        self.L = torch.max(self.L,torch.max(grads))

        #5: normalize
        grads =  grads / self.p[epoch]
        
        #6: construct sparsified gradient 
        sparsified_grads = torch.zeros_like(grads)
        for i in range(self.num_layers):
            if i in self.S[epoch]:
                sparsified_grads[i] = grads[i]
            else:
                sparsified_grads[i] = torch.zeros_like(grads[i])

        print(f'Sparsified grads: {sparsified_grads}')
        return sparsified_grads
    
    #Algo 1
    def upate_p(self, epoch: int, grads: torch.Tensor, learning_rate: float):
        next=epoch+1

        l=torch.Tensor(self.num_layers)
        w=torch.Tensor(self.num_layers)

        for d in range(self.num_layers):
            if self.Z[epoch,d]!=0:
                g=torch.zeros_like(grads)
                g[d]=grads[d]
                l1=-torch.pow(g.norm(),2)/torch.pow(self.p[epoch,d],2)
                l2= torch.pow(self.L,2)/torch.pow(self.p_min,2)
                l[d]=l1+l2
            else:
                l[d]=0
            
            w[d]=self.p[epoch,d]*torch.exp((-learning_rate*l[d])/self.p[epoch,d])

        # line 10: to bring it back to a probablility distribution but with a sum of sparsity budget 
        self.p[next]=(w*self.sparsity)/torch.sum(w)


if __name__ == '__main__':
    jointspar = JointSpar(num_layers=4, epochs=10, sparsity_budget=3, p_min=0.1)

    print('\nEpoch 0\n')
    jointspar.get_active_set(epoch=0)
    grads = torch.tensor([0.9, 0.1, 0.3, 0.4])
    print(f'Grads: {grads}')
    sparsed=jointspar.sparsify_gradient(epoch=0, grads=grads)
    jointspar.upate_p(epoch=0, grads=grads, learning_rate=0.1)

    print('\nEpoch 1\n')
    jointspar.get_active_set(epoch=1)
    grads = torch.tensor([0.1, 0.2, 0.3, 0.4])
    print(f'Grads: {grads}')
    sparsed=jointspar.sparsify_gradient(epoch=1, grads=grads)
    jointspar.upate_p(epoch=1, grads=grads, learning_rate=0.1)

    print('\nEpoch 2\n')
    jointspar.get_active_set(epoch=2)
    grads = torch.tensor([1.3, 0.2, 0.2, 0.8])
    print(f'Grads: {grads}')
    sparsed=jointspar.sparsify_gradient(epoch=2, grads=grads)
    jointspar.upate_p(epoch=2, grads=grads, learning_rate=0.1)
    
