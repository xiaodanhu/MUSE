import time
import datetime
import sys
from visdom import Visdom
import numpy as np

def tensor2imageA(tensor):
    image = tensor[0].cpu().float().numpy()
    mean = [0.4934, 0.4291, 0.3876] 
    std = [0.2621, 0.2501, 0.2401]
    x = np.zeros(image.shape)
    x[0, :, :] = image[0, :, :] * std[0] + mean[0]
    x[1, :, :] = image[1, :, :] * std[1] + mean[1]
    x[2, :, :] = image[2, :, :] * std[2] + mean[2]
    
    image = x * 255
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def tensor2imageB(tensor):
    image = tensor[0].cpu().float().numpy()
    mean =[0.4687, 0.4086, 0.3545]
    std = [0.2379, 0.2199, 0.2031]
    x = np.zeros(image.shape)
    x[0, :, :] = image[0, :, :] * std[0] + mean[0]
    x[1, :, :] = image[1, :, :] * std[1] + mean[1]
    x[2, :, :] = image[2, :, :] * std[2] + mean[2]

    image = x * 255
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, num_epochs, batches_epoch):
        # value initialization
        self.viz = Visdom()
        self.num_epochs = num_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        sys.stdout.write('\n')
        sys.stdout.write('\n')
        # output number of epoch and batches
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.num_epochs, self.batch, self.batches_epoch))

        # write out losses 
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.num_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # draw images
        if self.batch % 15 == 0:
            for image_name, tensor in images.items():
                if image_name == 'Input Photo':
                    if image_name not in self.image_windows:
                        self.image_windows[image_name] = self.viz.image(tensor2imageA(tensor.data), opts={'title':image_name})
                    else:
                        self.viz.image(tensor2imageA(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

                elif image_name == 'Golden Portrait' or image_name == 'Generated Portrait':
                    if image_name not in self.image_windows:
                        self.image_windows[image_name] = self.viz.image(tensor2imageB(tensor.data), opts={'title':image_name})
                    else:
                        self.viz.image(tensor2imageB(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # at the end of epoch
        if (self.batch % self.batches_epoch) == 0:
        #if True:
            # losses plotted
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss.cpu().numpy()/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss.cpu().numpy()/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                #print('debug passed')
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
