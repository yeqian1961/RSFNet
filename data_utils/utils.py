import  torch
import  numpy as np
import math
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))
#
# class AvgMeter(object):
#     def __init__(self, num=40):
#         self.num = num
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.sum = 0
#         self.count = 0
#         self.losses = []
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.losses.append(val)
#
#     @property
#     def avg(self):
#         return self.sum / self.count if self.count != 0 else 0
#
#     def show(self):
#         if len(self.losses) == 0:
#             return 0
#         start = max(len(self.losses) - self.num, 0)
#         return sum(self.losses[start:]) / len(self.losses[start:])