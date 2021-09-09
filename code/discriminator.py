import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    '''
    Parameters:
            in_features (int)       -- the number of input features/channels
    '''

    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.n_features = [64, 128, 256, 512]  # number of features/channels in each internal layers
        self.n_stride = [2, 2, 2, 1]  # number of stride applied in each internal layers

        # Input convolution blocks
        disc_model = [nn.Conv2d(in_features, self.n_features[0], 4, stride=2, padding=1),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout(0.5, inplace=False)]

        # Internal convolution blocks
        for n_blocks in range(3):
            disc_model += [
                nn.Conv2d(self.n_features[n_blocks], self.n_features[n_blocks + 1], 4, stride=self.n_stride[n_blocks],
                          padding=1),
                nn.InstanceNorm2d(self.n_features[n_blocks + 1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False)]

        self.disc_model = nn.Sequential(*disc_model)

        # Channel # reduction
        self.down = nn.Conv2d(self.n_features[-1], 1, 4, padding=1)
        self.fc =nn.Linear(512*16*16, 1)
        self.out = nn.Conv2d(self.n_features[-1], 8, 1)
        # create separate classifiers for our outputs
        last_channel = 2048
        self.age = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=5))
        self.mood = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=7))
        self.style = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=8))
        self.time = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=3))
        self.pose = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=10))
        self.setting = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=10))
        self.weather = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=10))
        self.face = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=15))
        self.gender = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=3))
        self.clothing = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=15))
        self.hair = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=6))


    def forward(self, x):
        x = self.disc_model(x)

        x1 = self.down(x)
        # Average pooling and flatten
        uncon = torch.sigmoid(F.avg_pool2d(x1, x1.size()[2:]).view(x1.size(0), -1)).view(-1)

        # reduce channels and classificaton
        x2 = self.out(x).view(x.size(0), -1)
        # con = {
        #     'hair': self.hair(x2)
        # }
        con = {
            'age': self.age(x2), 'mood': self.mood(x2), 'style': self.style(x2),
            'time': self.time(x2), 'pose': self.pose(x2), 'setting': self.setting(x2),
            'weather': self.weather(x2), 'face': self.face(x2), 'gender': self.gender(x2),
            'clothing': self.clothing(x2), 'hair': self.hair(x2)
        }
        return uncon, con

