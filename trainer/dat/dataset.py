import os
import pickle
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

""" Dataset partitioning helper """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []

        random.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Cifar:
    @staticmethod
    def get_loader(batch_size, world_size, rank, root):
        root = root
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, world_size, rank, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                  sampler=train_sampler)

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        return trainloader, testloader, train_sampler


class Cifar_EXT:
    @staticmethod
    def get_loader(batch_size, world_size, rank, root):
        root = root
        root1 = root
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)

        aux_path = os.path.join(root1, 'ti_500K_pseudo_labeled.pickle')
        print("Loading data from ti_500K_pseudo_labeled.pickle")
        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['data']
        aux_targets = aux['extrapolated_targets']
        orig_len = len(trainset.data)

        trainset.data = np.concatenate((trainset.data, aux_data), axis=0)
        trainset.targets.extend(aux_targets)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, world_size, rank, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                  sampler=train_sampler, pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return trainloader, testloader, train_sampler

    @staticmethod
    def get_local_loader(batch_size, root):
        root1 = root
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)

        aux_path = os.path.join(root1, 'ti_500K_pseudo_labeled.pickle')
        print(f'Loading data from ti_500K_pseudo_labeled.pickle from file: {aux_path}')
        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['data']
        aux_targets = aux['extrapolated_targets']

        trainset.data = np.concatenate((trainset.data, aux_data), axis=0)
        trainset.targets.extend(aux_targets)

        train_sampler = torch.utils.data.Sampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader, train_sampler


class ImageNet:
    @staticmethod
    def get_loader(batch_size, world_size, rank, root):
        root = root
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        #trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform_train)
        trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, world_size, rank, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                  sampler=train_sampler, num_workers=32, pin_memory=True)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        #testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform_test)
        testset = torchvision.datasets.ImageFolder(root=root, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return trainloader, testloader, train_sampler
