import glob
import random
import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import numpy as np
import pandas as pd
import torch
import torchtext
# set up data loader
class LoadData(Dataset):
    def __init__(self, root, transA, transB, opt, unaligned=False, mode='train'):
        # combine the transform
        self.transformA = transforms.Compose(transA)
        self.transformB = transforms.Compose(transB)
        # set if we want the data to be aligned
        self.unaligned = unaligned

        # setup training/testing file [mode: train/test (A&B are the coupled images)]
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

        # path for annotation directories
        img_list = [img.split('/')[-1] for img in self.files_B]
        with open(root + 'annotation/Attributes_test.json') as json_file:
            truth_label_list = json.load(json_file)

        # seq and labels lists
        self.word_list = {}
        self.label_list = {}
        embed_dir = './datasets/photo2portrait/attribute_embeddings.pkl'
        embed = pickle.load(open(embed_dir, "rb"))
        word_to_id = {token: idx for idx, token in enumerate(embed.keys())}

        file_count = 0
        attr_value = [int(item) for item in opt.att_value_ind.split(',')]
        # iterate over file lists & read the data
        for attribute in truth_label_list.keys():
            truth_label = truth_label_list[attribute]
            labels_per_item = truth_label[attr_value[file_count]]

            for img_id in img_list:
                if file_count == 0:
                    self.word_list.setdefault(img_id, [])
                    self.label_list.setdefault(img_id, [])

                vector = [word_to_id[labels_per_item.lower()]]
                self.word_list[img_id].append(vector)

                label = [0] * len(truth_label)
                label[attr_value[file_count]] = 1
                self.label_list[img_id].append(label)

            file_count += 1

    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        item_A = self.transformA(item_A)
        if self.unaligned:
            item_B = self.transformB(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transformB(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        item_T = self.word_list[self.files_B[index % len(self.files_B)].split('/')[-1]]
        item_T = torch.LongTensor(item_T)

        item_L = self.label_list[self.files_B[index % len(self.files_B)].split('/')[-1]]
        item_L = {
            'hair': item_L[0]
        }
        # item_L = {
        #     'age': item_L[0], 'mood': item_L[1], 'style': item_L[2],
        #     'time': item_L[3], 'pose': item_L[4], 'setting': item_L[5],
        #     'weather': item_L[6], 'face': item_L[7], 'gender': item_L[8],
        #     'clothing': item_L[9], 'hair': item_L[10]
        # }

        return {'A': item_A, 'B': item_B, 'T': item_T, 'Label': item_L}

    def __len__(self):
        # return maxium length in folder A and folder B
        return max(len(self.files_A), len(self.files_B))


