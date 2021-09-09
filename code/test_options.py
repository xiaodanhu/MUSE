import argparse


class testParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--attribute_type', type=str, default='hair', help='attribute type')
        parser.add_argument('--attribute_value', type=str, default='short', help='attribute value')
        parser.add_argument('--att_value_ind', type=str, default='4', help='attribute value index in json')
        parser.add_argument('--num_epoch', type=int, default=201, help='number of epochs')
        parser.add_argument('--data_path', type=str, default='./dataset/photo2portrait/', help='root path for data')
        parser.add_argument('--data_size', type=int, default=256, help='size of the data')
        parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
        parser.add_argument('--num_input', type=int, default=3, help='number of channels of input data')
        parser.add_argument('--num_output', type=int, default=3, help='number of channels of output data')
        parser.add_argument('--num_resblock', type=int, default=9, help='number of ResNet block')
        parser.add_argument('--num_cpu', type=int, default=0, help='number of cpu used in data preparation ')
        parser.add_argument('--generator', type=str, default='./output/netG_UNet.pth', help='generator saved file')
        parser.add_argument('--verbose', type=int, default=1, help='level of printing out information ')
        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        return opt
