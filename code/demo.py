import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
from test_options import testParser
from util.load_test_data import LoadData
from generator import Generator
from util.model_func import tensor2image
from PIL import Image
import pickle

# setup the options for model testing
parser = testParser()
opt = parser.parse()
# setup device (GPU and CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_dir = opt.data_path + 'attribute_embeddings.pkl'  # glove_embeddings
embed = pickle.load(open(embed_dir, "rb"))
id_to_word = sorted(list(embed.keys()))
weight = torch.FloatTensor([embed[w] for w in id_to_word])

# setup the network models
if opt.verbose > 0:
    print('Set up networks...')

# setup the data loader
transA = [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.4934, 0.4291, 0.3876], [0.2621, 0.2501, 0.2401])]
transB = [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.4687, 0.4086, 0.3545], [0.2379, 0.2199, 0.2031])]
dataloader = DataLoader(LoadData(opt.data_path, transA, transB, opt, unaligned=False, mode='test'),
                        batch_size=1, shuffle=False, num_workers=opt.num_cpu)

# setup output folder
attribute_type = opt.attribute_type
attribute_value = opt.attribute_value
output_path = './output/generated_portrait_'+attribute_type+'_'+attribute_value+'2'
if not os.path.exists(output_path):
    os.makedirs(output_path)

meanA = [0.4934, 0.4291, 0.3876]
stdA = [0.2621, 0.2501, 0.2401]

meanB = [0.4687, 0.4086, 0.3545]
stdB = [0.2379, 0.2199, 0.2031]
with open(opt.data_path + 'annotation/Attributes_test.json') as json_file:
    truth_label_list = json.load(json_file)

# testing loop
if opt.verbose > 0:
    print('Start the testing ...')
for idx, iBatch in enumerate(dataloader):
    # setup input
    real_A = iBatch['A'].to(device)
    real_T = iBatch['T'].to(device)
    label = iBatch['Label']

    show_text = []
    for k, v in truth_label_list.items():
        cur_label = torch.stack(label[k]).permute(-1, 0)[0][:-1]
        show_text.extend([v[i] for i in range(len(cur_label)) if (cur_label[i] == 1)])

    real_A_img = tensor2image(real_A, meanA, stdA)

    # save images
    counter = 2
    for epoch in range(80, opt.num_epoch):
        if epoch == 80 or epoch == 240 or epoch == 400 or epoch == 600:
            # load model
            netG = Generator(opt.num_input, opt.num_output, weight).to(device)
            netG.load_state_dict(torch.load('./output/models/' + attribute_type + '/netG_epoch_' + str(epoch) + '.pth'))
            netG.eval()

            fake_B = netG(real_A, real_T)
            fake_B_img = tensor2image(fake_B, meanB, stdB)
            Image.fromarray(fake_B_img.transpose(1, 2, 0)).save(output_path+'/%05d_epoch%03d.png' % (idx+1, epoch))
            counter += 1

    # print information
    if opt.verbose > 0:
        print('Generated images %05d of %05d' % (idx+1, len(dataloader)))
