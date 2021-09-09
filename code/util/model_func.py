import torch
import numpy as np


def init_func(m, init_gain=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, init_gain)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, init_gain)
        torch.nn.init.constant(m.bias.data, 0.0)

def get_scheduler(optimizer, opt):
    offset_epoch = opt.start_epoch
    decay_epoch = opt.decay_epoch
    num_epoch = opt.num_epoch
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + offset_epoch - decay_epoch) / float(num_epoch-decay_epoch)
        return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def tensor2image(tensor, mean, std):
    image = tensor[0].cpu().detach().float().numpy()
    x = np.zeros(image.shape)
    x[0, :, :] = image[0, :, :] * std[0] + mean[0]
    x[1, :, :] = image[1, :, :] * std[1] + mean[1]
    x[2, :, :] = image[2, :, :] * std[2] + mean[2]
    image = x * 255
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)