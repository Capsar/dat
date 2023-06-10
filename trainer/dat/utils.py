import math
import os

import torch


# Source code from https://github.com/a1600012888/YOPO-You-Only-Propagate-Once

def torch_accuracy(output, target, topk=(1,)):
    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


class AvgMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir {}'.format(path))
        os.mkdir(path)


def save_checkpoint(now_epoch, net, optimizer, lr_scheduler, file_name):
    checkpoint = {'epoch': now_epoch,
                  'state_dict': net.module.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler_state_dict': lr_scheduler.state_dict()}
    if os.path.exists(file_name):
        print('Overwriting {}'.format(file_name))
    torch.save(checkpoint, file_name)


def load_checkpoint(file_name, net=None, optimizer=None, lr_scheduler=None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        check_point = torch.load(file_name)
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['state_dict'])
        if optimizer is not None:
            print('Loading optimizer state dict')
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        if lr_scheduler is not None:
            print('Loading lr_scheduler state dict')
            lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        return check_point['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


def seconds_to_string(seconds: float) -> str:
    days, seconds = divmod(seconds, 60 * 60 * 24)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    day_part = f'{int(days)} DAYS ' if days > 0 else ''
    hour_part = f'{int(hours):02}H:' if hours > 0 else ''
    minute_part = f'{int(minutes):02}M:' if minutes > 0 or hours > 0 else ''
    return f'{day_part}{hour_part}{minute_part}{int(seconds):02}S'
