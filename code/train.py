import os
# import pytorch lib
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchtext
import pickle
torch.manual_seed(0)
from train_options import trainParser

from generator import Generator
from discriminator import Discriminator
from util.model_func import init_func
from util.model_func import get_scheduler

from util.load_train_data import LoadData
from util.get_loss import get_loss
from util.logger import Logger

# setup the options for model training
parser = trainParser()
opt = parser.parse()
if opt.verbose > 0:
    print('Setup the options as below:')
    print(' --', opt)

# setup device (GPU and CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.verbose > 0:
    print('Current device:')
    print(' --', device)

embed_dir = './datasets/photo2portrait/attribute_embeddings.pkl'
embed = pickle.load(open(embed_dir, "rb"))
id_to_word = sorted(list(embed.keys()))
weight = torch.FloatTensor([embed[w] for w in id_to_word])

# setup the network models
if opt.verbose > 0:
    print('Set up networks...')
netG = Generator(opt.num_input, opt.num_output, weight).to(device)
netD = Discriminator(opt.num_output).to(device)
# weight init
# netG.apply(init_func)
netD.apply(init_func)

# setup loss for cycle GAN
criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
# setup of optimizer
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.learn_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.learn_rate, betas=(0.5, 0.999))

# learning rate schedule
scheduler_G = get_scheduler(optimizer_G, opt)
scheduler_D = get_scheduler(optimizer_D, opt)

# setup the data loader
transA = [transforms.Resize(int(opt.data_size * 1.12), Image.BICUBIC),
                  transforms.RandomCrop(opt.data_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.4934, 0.4291, 0.3876], [0.2621, 0.2501, 0.2401])]
transB = [transforms.Resize(int(opt.data_size * 1.12), Image.BICUBIC),
                  transforms.RandomCrop(opt.data_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.4687, 0.4086, 0.3545], [0.2379, 0.2199, 0.2031])]
dataset = LoadData(opt.data_path, transA, transB, unaligned=False)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_cpu)

Tensor = torch.cuda.FloatTensor
# initialize ploter
logger = Logger(opt.num_epoch, len(dataloader))

# training loop
if opt.verbose > 0:
    print('Start the training ...')
    
loss_G_list = []
loss_D_list = []

for epoch in range(opt.start_epoch, opt.num_epoch):
    
    epoch_cnt = 0
    epoch_loss_G, epoch_loss_D = 0.0, 0.0
    for idx, iBatch in enumerate(dataloader):
        real_A = iBatch['A'].to(device)
        real_B = iBatch['B'].to(device)
        label = iBatch['Label']
        real_T = iBatch['T'].to(device)

        target_real = Variable(Tensor(len(real_A)).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(len(real_A)).fill_(0.0), requires_grad=False)

        # generator train
        optimizer_G.zero_grad()
        # generate fake data
        fake_B = netG(real_A, real_T)

        # adversarial loss calculation
        D_uncon, D_con = netD(fake_B)
        loss_gan = criterion(D_uncon, target_real)
        loss_gan_L1 = criterionL1(fake_B, real_B)
        loss_gan_con = get_loss(D_con, label)
        # total loss
        loss_all_G = 0.5*(loss_gan + loss_gan_con) + loss_gan_L1
        # gradient descent
        loss_all_G.backward()
        optimizer_G.step()

        # discriminator B train
        optimizer_D.zero_grad()
        # real loss calculation
        D_uncon, D_con = netD(real_B)
        loss_D_real = criterion(D_uncon, target_real)
        loss_D_real_con = get_loss(D_con, label)
        # fake loss calculation
        D_uncon, D_con = netD(fake_B.detach())
        loss_D_fake = criterion(D_uncon, target_fake)
        loss_D_fake_con = get_loss(D_con, label)
        # total loss
        loss_all_D = 0.25*(loss_D_real + loss_D_fake + loss_D_real_con + loss_D_fake_con)
        # gradient descent
        loss_all_D.backward()
        optimizer_D.step()

        # progress for logger
        logger.log({'loss_G': loss_all_G, 'loss_D': loss_all_D}, images={'Input Photo': real_A, 'Golden Portrait': real_B, 'Generated Portrait': fake_B})
        epoch_loss_G += loss_all_G
        epoch_loss_D += loss_all_D
        epoch_cnt += 1
    
    epoch_loss_G /= epoch_cnt
    epoch_loss_D /= epoch_cnt

    loss_G_list.append(epoch_loss_G.detach().cpu().numpy())
    loss_D_list.append(epoch_loss_D.detach().cpu().numpy())

    print('before scheduler, loss list updated.')
    # learning rate schedule
    scheduler_G.step()
    scheduler_D.step()

    # save model
    if opt.verbose > 0:
        print('Save model ...')

    if not os.path.exists('./output'):
        os.makedirs('./output')

    if epoch % 40 == 0:
        torch.save(netG.state_dict(), './output/models/'+opt.model_type+'/netG_epoch_'+str(epoch)+'.pth')
        torch.save(netD.state_dict(), './output/models/'+opt.model_type+'/netD_epoch_'+str(epoch)+'.pth')

    # save_train = pd.DataFrame({"loss_G":np.array(loss_G_list), "loss_D":np.array(loss_D_list)})
    # save_train.to_csv('./output/train_loss.csv', index=False)
