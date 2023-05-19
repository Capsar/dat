import argparse
from collections import OrderedDict

import torch
from torchvision.models import resnet50
from tqdm import tqdm

from attack import PGD
from dataset import Cifar_EXT, ImageNet
from models import PreActResNet18
from utils import AvgMeter, torch_accuracy

parser = argparse.ArgumentParser(description='distributed adversarial training')
parser.add_argument('--dataset', default='cifar', choices=['cifar', 'imagenet'],
                    help='dataset cifar or imagenet')
parser.add_argument('--dataset-path', type=str,
                    help='dataset folder')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint path')


def eval(net, data_loader, DEVICE=torch.device('cuda:0'), es=(8.0, 20)):
    net.eval()
    pbar = tqdm(data_loader)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    eps, step = es
    at_eval = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item(), acc[0].size(0))

        adv_inp = at_eval.attack(net, data, label)

        with torch.no_grad():
            pred = net(adv_inp)
            acc = torch_accuracy(pred, label, (1,))
            adv_accuracy.update(acc[0].item(), acc[0].size(0))

        pbar_dic = OrderedDict()
        pbar_dic['standard test acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['robust acc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return clean_accuracy.mean, adv_accuracy.mean


def main():
    args = parser.parse_args()
    print(args)
    DEVICE = torch.device('cuda:0')
    if args.dataset == 'cifar':

        net = PreActResNet18()

        batch_size = 2048

        ds_train, ds_val, sp_train = Cifar_EXT.get_loader(batch_size, 1, 0, args.dataset_path)
        es = (8.0, 10)


    elif args.dataset == 'imagenet':

        batch_size = 512

        net = resnet50()

        ds_train, ds_val, sp_train = ImageNet.get_loader(batch_size, 1, 0, args.dataset_path)
        es = (2.0, 4)

    checkpoint_path = args.checkpoint
    torch.backends.cudnn.benchmark = True

    net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    net = torch.nn.DataParallel(net).to(DEVICE)

    eval(net, ds_val, DEVICE, es)


if __name__ == "__main__":
    main()
