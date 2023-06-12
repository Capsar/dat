import glob
import json
import time
import logging
import sys
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
import os
from collections import OrderedDict

from trainer.dat.jointspar import JointSpar
from trainer.dat.quantization import RandomQuantizer
from trainer.dat.attack import PGD
from torch.autograd import Variable
from trainer.dat.dataset import Cifar, Cifar_EXT, ImageNet
from trainer.dat.models import PreActResNet18
from trainer.dat.utils import save_checkpoint, torch_accuracy, AvgMeter
from torchvision.models import resnet50
from trainer.dat.lamb import Lamb
from trainer.dat.helpers import send_telegram_message

import wandb

from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser(description='distributed adversarial training')
parser.add_argument('--gcloud', default=False, type=bool,
                    help='whether the code is running on gcloud')
parser.add_argument('--dataset', default='cifar', choices=['cifar', 'cifarext', 'imagenet'],
                    help='dataset cifar or imagenet')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--batch-size', default=2048, type=int,
                    help='batch size')
parser.add_argument('--warmup-epochs', default=5, type=int,
                    help='number of warmup epochs')
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
parser.add_argument('--output-dir', default='saved_models', type=str, help='output directory')
parser.add_argument('--group_surfix', default='timestamp', type=str, help='group name for wandb logging')
parser.add_argument('--machine_type', default='local', type=str, help='machine type for wandb logging')
parser.add_argument('--accelerator_type', default='cpu', type=str, help='accelerator type for wandb logging')

qt = RandomQuantizer()


def distributed(param, rank, size, DEVICE):
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
        param.grad = tmp.to(DEVICE)
    else:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    param.grad.data /= float(size)


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


# global global_noise_data
# global_noise_data = torch.zeros([512, 3, 224, 224]).cuda()
def train(net, data_loader, optimizer, criterion, DEVICE, descrip_str='', es=(8.0, 10), fast=False,
          lr_scheduler=None, warmup=False, epoch=None, S=None, jointspar=None):
    if descrip_str != '':
        descrip = descrip_str.split(':')[0]
        epoch = descrip_str.split(':')[1].split('/')[0]
        total_epoch = descrip_str.split(':')[1].split('/')[1]

    net.train()
    pbar = tqdm(data_loader, ncols=200, file=sys.stdout)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    size = (dist.get_world_size())
    rank = dist.get_rank()

    eps, step = es
    if not fast:
        at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step, DEVICE=DEVICE)
        for i, (data, label) in enumerate(pbar):

            with record_function("broadcast"):
                if i == 0:
                    for param in net.parameters():
                        dist.broadcast(param.data, 0)

            with record_function("data_to_device"):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

            if i != 0 or epoch != 0:
                optimizer.zero_grad()

            with record_function("adv_attack"):
                adv_inp = at.attack(net, data, label)
                optimizer.zero_grad()
                net.train()

            with record_function("adv_forward"):
                pred = net(adv_inp)

            with record_function("adv_loss"):
                loss = criterion(pred, label)
                acc = torch_accuracy(pred, label, (1,))
                advacc = acc[0].item()
                advloss = loss.item()

            with record_function("adv_backward"):
                (loss * 1.0).backward()

            with record_function("clean_forward"):
                pred = net(data)

            with record_function("clean_loss"):
                loss = criterion(pred, label)

            with record_function("clean_backward"):
                loss.backward()

            with record_function("distributed"):
                for param in net.parameters():
                    distributed(param, rank, size, DEVICE)

            with record_function("clean_optimizer_step"):
                optimizer.step()
                if warmup:
                    lr_scheduler.step()

            with record_function("clean_loss"):
                acc = torch_accuracy(pred, label, (1,))
                cleanacc = acc[0].item()
                cleanloss = loss.item()

            metrics = {
                f"{descrip}/clean_acc": cleanacc,
                f"{descrip}/clean_loss": cleanloss,
                f"{descrip}/adv_acc": advacc,
                f"{descrip}/adv_loss": advloss,
                f"{descrip}/lr": lr_scheduler.get_last_lr()[0],
                f"{descrip}/epoch": int(epoch),
                f"{descrip}/total_epoch": int(total_epoch),
                f"{descrip}/rank": rank,
                f"{descrip}/world_size": size,
                f"{descrip}/batch_nr": i,
            }
            wandb.log(metrics)
            pbar_dic = OrderedDict()
            pbar_dic['standard test acc'] = '{:.2f}'.format(cleanacc)
            pbar_dic['standard test loss'] = '{:.2f}'.format(cleanloss)
            pbar_dic['robust acc'] = '{:.2f}'.format(advacc)
            pbar_dic['robust loss'] = '{:.2f}'.format(advloss)
            pbar_dic['lr'] = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix(pbar_dic)

        if jointspar is not None:
            sparsified_grads = []
            for ind, p in enumerate(net.parameters()):
                if ind in S:
                    curr_p = 0 if p.grad is None else p.grad
                    sparsified_grads.append(curr_p / jointspar.p[epoch, ind])
                else:
                    sparsified_grads.append([])
            jointspar.update_p(epoch, sparsified_grads, lr_scheduler.get_last_lr()[0])
    else:
        global global_noise_data
        for i, (data, label) in enumerate(pbar):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            global_noise_data.uniform_(-eps / 255.0, eps / 255.0)
            # sync all parameters at epoch 0
            if i == 0:
                for param in net.parameters():
                    dist.broadcast(param.data, 0)
            for j in range(1):
                noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=True).to(DEVICE)
                in1 = data + noise_batch
                in1.clamp_(0, 1.0)
                output = net(in1)
                loss = criterion(output, label)

                acc = torch_accuracy(output, label, topk=(1,))
                cleanacc = acc[0].item()
                cleanloss = loss.item()

                loss.backward()

                # Update the noise for the next iteration
                pert = fgsm(noise_batch.grad, 1.25 * eps / 255.0)
                global_noise_data[0:data.size(0)] += pert.data
                global_noise_data.clamp_(-eps / 255.0, eps / 255.0)

                # Dscend on the global noise
                noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=False).to(DEVICE)
                in1 = data + noise_batch
                in1.clamp_(0, 1.0)
                output = net(in1)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                for param in net.parameters():
                    distributed(param, rank, size, DEVICE)
                optimizer.step()
                if warmup:
                    lr_scheduler.step()
                # print(lr_scheduler.get_lr()[0])
            pbar_dic = OrderedDict()
            pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
            pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
            pbar_dic['lr'] = lr_scheduler.get_lr()[0]
            pbar.set_postfix(pbar_dic)


def eval(net, data_loader, DEVICE, es=(8.0, 20)):
    net.eval()
    pbar = tqdm(data_loader, file=sys.stdout)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    eps, step = es
    at_eval = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step, DEVICE=DEVICE)
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

        metrics = {
            f"Evaluation/clean_acc": clean_accuracy.mean,
            f"Evaluation/adv_acc": adv_accuracy.mean,
        }
        wandb.log(metrics)
        pbar_dic = OrderedDict()
        pbar_dic['standard test acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['robust acc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return clean_accuracy.mean, adv_accuracy.mean


def main():
    print('start')
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = DEVICE
    group_name = f'T_{args.group_surfix}'
    # send_telegram_message(message=f'Starting main() in task.py with args: {args}')
    args.accelerator_count = 0
    if torch.cuda.is_available():
        args.accelerator_count = torch.cuda.device_count()
    if args.gcloud:
        # Get the rank and world size from the environment variables
        args.dist_url = 'env://'
        print(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], int(os.environ['RANK']),
              int(os.environ['WORLD_SIZE']))

    group_name = f'{args.machine_type}_{int(os.environ["WORLD_SIZE"])}_{args.accelerator_type}_{args.accelerator_count}_{args.dataset}_{args.batch_size}_{args.group_surfix}'
    # Use torch.multiprocessing.spawn to launch distributed processes
    if torch.cuda.is_available():
        torch.multiprocessing.spawn(main_worker, args=(group_name, args), nprocs=args.accelerator_count, join=True)
    else:
        # CPU ONLY
        pass


def main_worker(local_rank, group_name, args):
    DEVICE = f'cuda:{local_rank}'
    args.rank = int(os.environ['RANK']) * args.accelerator_count + local_rank
    args.world_size = int(os.environ['WORLD_SIZE']) * args.accelerator_count
    cluster_spec = json.loads(os.environ['CLUSTER_SPEC'])
    task_type = cluster_spec["task"]["type"]
    task_index = cluster_spec["task"]["index"]
    args.task_name = f"{task_type}-{task_index}-DR{int(os.environ['RANK'])}-LR{local_rank}-R{args.rank}"
    wandb.login(key=os.environ['WANDB_API_KEY'])
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(local_rank)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    eval_epochs = args.eval_epochs
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    global global_noise_data
    if args.dataset == 'cifar' or args.dataset == 'cifarext':
        print('using CIFAR dataset')
        global_noise_data = torch.zeros([batch_size, 3, 32, 32]).to(DEVICE)
        print('cifar check 1')
        net = PreActResNet18().to(DEVICE)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank]).to(DEVICE)
        # net = torch.nn.DataParallel(net).to(DEVICE)

        print('cifar check 2')
        if args.wolalr:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = Lamb(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), adam=False)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 95], gamma=0.1)

        print('cifar check 3')
        if args.dataset == 'cifar':
            ds_train, ds_val, sp_train = Cifar.get_loader(batch_size, args.world_size, args.rank, args.dataset_path)
        else:
            ds_train, ds_val, sp_train = Cifar_EXT.get_loader(batch_size, args.world_size, args.rank, args.dataset_path)
        es = (8.0, 10)

        print('done selecting cifar')

    elif args.dataset == 'imagenet':
        print('using IMAGENET')
        global_noise_data = torch.zeros([batch_size, 3, 224, 224]).to(DEVICE)

        net = resnet50().to(DEVICE)
        net = torch.nn.parallel.DistributedDataParallel(net).to(DEVICE)
        # net = torch.nn.DataParallel(net).to(DEVICE)

        if args.wolalr:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = Lamb(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), adam=False)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 28], gamma=0.1)

        ds_train, ds_val, sp_train = ImageNet.get_loader(batch_size, args.world_size, args.rank, args.dataset_path)
        es = (2.0, 4)

    lr_steps = 5 * len(ds_train)
    print('lr_step:{}'.format(lr_steps))
    lambda1 = lambda step: (step + 1) / lr_steps
    warm_up_lr_lchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0, max_lr=0.15,
    #         step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

    with wandb.init(entity="sdml-dat", project="sdml-dat", group=group_name, name=args.task_name, config=args,
                    sync_tensorboard=True) as wandb_run:
        print('warmup starts')

        for epoch in range(args.warmup_epochs):
            descrip_str = 'Warmup:{}/{}'.format(epoch, args.warmup_epochs)
            train(net, ds_train, optimizer, criterion, DEVICE, descrip_str, es, fast=args.fast,
                  lr_scheduler=warm_up_lr_lchedule, warmup=True)
            # prof.step()

        print('training starts')
        num_layers = sum(1 for _ in net.parameters())
        print(f'Number of layers: {num_layers}')
        using_jointspar = args.jointspar
        print(f'Using jointspar: {using_jointspar}')
        jointspar = None
        if using_jointspar:
            jointspar = JointSpar(
                num_layers=num_layers,
                epochs=num_epochs,
                sparsity_budget=40,
                p_min=5.0
            )

        for epoch in range(num_epochs):
            S = None
            if using_jointspar:
                S = jointspar.get_active_set(epoch)
                for i, p in enumerate(net.parameters()):
                    p.requires_grad_(i in S)
                print(f'p: {jointspar.p[epoch, :]}')

            sp_train.set_epoch(epoch)
            descrip_str = 'Training:{}/{}'.format(epoch, num_epochs)
            train(net, ds_train, optimizer, criterion, DEVICE, descrip_str, es, fast=args.fast,
                  lr_scheduler=lr_scheduler, epoch=epoch, S=S, jointspar=jointspar)
            lr_scheduler.step()
            # prof.step()

            if eval_epochs > 0 and (epoch + 1) % eval_epochs == 0:
                clean_acc, adv_acc = eval(net, ds_val, DEVICE, es)
                message = f'EPOCH {epoch + 1} accuracy: {clean_acc:.3f}% adversarial accuracy: {adv_acc:.3f}%'
                if send_telegram_message(message=message):
                    print('successfully sent Telegram message!')
                else:
                    print('error sending Telegram message!')

            if args.rank == 0 and (epoch + 1) % 10 == 0:
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                save_checkpoint(epoch, net, optimizer, lr_scheduler,
                                file_name=os.path.join(args.output_dir, 'epoch-{}.checkpoint'.format(epoch)))

        # prof.stop()

        print('training done')
        print('eval starts')

        eval(net, ds_val, DEVICE, es)

        wandb.finish()


if __name__ == "__main__":
    main()