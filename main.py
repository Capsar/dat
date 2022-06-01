import torch
import torch.distributed as dist
import time
import numpy as np
import argparse
from tqdm import tqdm
import os
from collections import OrderedDict
from quantization import RandomQuantizer
from attack import PGD
from torch.autograd import Variable
from dataset import Cifar, Cifar_EXT, ImageNet
from models import PreActResNet18, ResNet50
from utils import save_checkpoint, load_checkpoint, torch_accuracy, AvgMeter
from torchvision.models import resnet50
from lamb import Lamb


parser = argparse.ArgumentParser(description='distributed adversarial training')
parser.add_argument('--dataset', default='cifar', choices=['cifar', 'imagenet'],
                    help='dataset cifar or imagenet')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--batch-size', default=2048, type=int,
                    help='batch size')
parser.add_argument('--num-epochs', default=200, type=int,
                    help='total training epoch')
parser.add_argument('--eval-epochs', default=10, type=int,
                    help='eval epoch interval')
parser.add_argument('--fast', action="store_true",
                    help='whether to use fgsm')
parser.add_argument('--wolalr', action="store_true",
                    help='whether to train without layer-wise adptive learning rate')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rates')
parser.add_argument('--dataset-path', type=str,
                    help='dataset folder')
parser.add_argument('--output-dir', default='saved_models', type=str,
                    help='output directory')

qt = RandomQuantizer()
def distributed(param, rank, size):
    quantization = False
    if quantization:
        if size == 1:
            raise ValueError("quantization can't run on single node")
        cpu_grad = param.grad.cpu()
        q, norm = qt.quantize(cpu_grad)
        q_list = [torch.zeros_like(q) for _ in range(size)]
        norm_list = [torch.zeros_like(norm) for _ in range(size)]
        dist.all_gather(q_list, q)
        dist.all_gather(norm_list, norm)
        tmp = torch.zeros_like(cpu_grad)
        for q, norm in zip(q_list, norm_list):
            tmp += qt.dequantize(q, norm)
        param.grad = tmp.cuda()
    else:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad.data /= float(size)
def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)

global global_noise_data
global_noise_data = torch.zeros([512, 3, 32, 32]).cuda()
def train(net, data_loader, optimizer,
          criterion, DEVICE=torch.device('cuda:0'),
          descrip_str='', es = (8.0, 10), fast=False, lr_scheduler=None, warmup=False):


    net.train()
    pbar = tqdm(data_loader, ncols=200)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    size = (dist.get_world_size())
    rank = dist.get_rank()

    eps, step = es
    if not fast:
        at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
        for i, (data, label) in enumerate(pbar):
            if i == 0:
                for param in net.parameters():
                    dist.broadcast(param.data, 0)
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            pbar_dic = OrderedDict()


            adv_inp = at.attack(net, data, label)
            optimizer.zero_grad()
            net.train()
            pred = net(adv_inp)
            loss = criterion(pred, label)

            acc = torch_accuracy(pred, label, (1,))
            advacc = acc[0].item()
            advloss = loss.item()
            (loss * 1.0).backward()


            pred = net(data)

            loss = criterion(pred, label)
            loss.backward()

            for param in net.parameters():
                distributed(param, rank, size)

            optimizer.step()
            if warmup:
                    lr_scheduler.step()
            acc = torch_accuracy(pred, label, (1,))
            cleanacc = acc[0].item()
            cleanloss = loss.item()
            pbar_dic['standard test acc'] = '{:.2f}'.format(cleanacc)
            pbar_dic['standard test loss'] = '{:.2f}'.format(cleanloss)
            pbar_dic['robust acc'] = '{:.2f}'.format(advacc)
            pbar_dic['robust loss'] = '{:.2f}'.format(advloss)
            pbar_dic['lr'] = lr_scheduler.get_lr()[0]
            pbar.set_postfix(pbar_dic)
    else:
        global global_noise_data
        for i, (data, label) in enumerate(pbar):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            global_noise_data.uniform_(-eps/255.0, eps/255.0)
            #sync all parameters at epoch 0
            if i == 0:
                for param in net.parameters():
                    dist.broadcast(param.data, 0)
            for j in range(1):
                noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=True).cuda()
                in1 = data + noise_batch
                in1.clamp_(0, 1.0)
                output = net(in1)
                loss = criterion(output, label)

                acc = torch_accuracy(output, label, topk=(1,))
                cleanacc = acc[0].item()
                cleanloss = loss.item()

                loss.backward()

                # Update the noise for the next iteration
                pert = fgsm(noise_batch.grad, 1.25*eps/255.0)
                global_noise_data[0:data.size(0)] += pert.data
                global_noise_data.clamp_(-eps/255.0, eps/255.0)

                # Dscend on the global noise
                noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=False).cuda()
                in1 = data + noise_batch
                in1.clamp_(0, 1.0)
                output = net(in1)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                for param in net.parameters():
                    distributed(param, rank, size)
                optimizer.step()
                if warmup:
                    lr_scheduler.step()
                # print(lr_scheduler.get_lr()[0])
            pbar_dic = OrderedDict()
            pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
            pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
            pbar_dic['lr'] = lr_scheduler.get_lr()[0]
            pbar.set_postfix(pbar_dic)


def eval(net, data_loader, DEVICE=torch.device('cuda:0'), es=(8.0, 20)):
    net.eval()
    pbar = tqdm(data_loader)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    eps, step = es
    at_eval = PGD(eps=eps/ 255.0, sigma=2/255.0, nb_iter=step)
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
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    eval_epochs = args.eval_epochs
    DEVICE = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if args.dataset == 'cifar':
        net = PreActResNet18()
        net = torch.nn.DataParallel(net).to(DEVICE)

        if args.wolalr:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = Lamb(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), adam=False)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [75, 90, 95], gamma = 0.1)

        ds_train, ds_val, sp_train = Cifar_EXT.get_loader(batch_size, args.world_size, args.rank, args.dataset_path)
        es =(8.0, 10)

        # lr_steps = num_epochs * len(ds_train)
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0, max_lr=args.lr,
        #                                                  step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

    elif args.dataset == 'imagenet':
        net = resnet50()
        net = torch.nn.DataParallel(net).to(DEVICE)

        if args.wolalr:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = Lamb(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), adam=False)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1)

        ds_train, ds_val, sp_train = ImageNet.get_loader(batch_size, args.world_size, args.rank, args.dataset_path)
        es = (2.0, 4)
    
    lr_steps = 5 * len(ds_train)
    print('lr_step:{}'.format(lr_steps))
    lambda1 = lambda step: (step+1) / lr_steps

    warm_up_lr_lchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    #
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0, max_lr=0.15,
    #         step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

    for epoch in range(5):
    #
        descrip_str = 'warmup epoch:{}/{}'.format(epoch, 5)
        train(net, ds_train, optimizer, criterion, DEVICE,
              descrip_str, es, fast=args.fast, lr_scheduler=warm_up_lr_lchedule, warmup=True)

    for epoch in range(num_epochs):

        sp_train.set_epoch(epoch)

        descrip_str = 'Training epoch:{}/{}'.format(epoch, num_epochs)
        train(net, ds_train, optimizer, criterion, DEVICE,
              descrip_str, es, fast=args.fast, lr_scheduler=lr_scheduler)

        lr_scheduler.step()

        if eval_epochs > 0 and (epoch + 1) % eval_epochs == 0:
            eval(net, ds_val, DEVICE, es)

        if args.rank == 0 and (epoch+1) % 10 == 0:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            save_checkpoint(epoch, net, optimizer, lr_scheduler,
                            file_name=os.path.join(args.output_dir, 'epoch-{}.checkpoint'.format(epoch)))


    eval(net, ds_val, DEVICE, es)


if __name__ == "__main__":
    main()
