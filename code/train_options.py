import argparse


class trainParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', type=str, default='combined', help='attribute type')
        parser.add_argument('--num_epoch', type=int, default=401, help='number of total epoch')
        parser.add_argument('--data_path', type=str, default='./dataset/photo2portrait/', help='root path for data')
        parser.add_argument('--data_size', type=int, default=256, help='size of the data')
        parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')  # 32
        parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch index')
        parser.add_argument('--learn_rate', type=float, default=0.0002, help='initial learning rate')  # 0.0002
        parser.add_argument('--num_input', type=int, default=3, help='number of channels of input data')
        parser.add_argument('--num_output', type=int, default=3, help='number of channels of output data')
        parser.add_argument('--num_resblock', type=int, default=9, help='number of ResNet block')
        parser.add_argument('--num_cpu', type=int, default=8, help='number of cpu used in data preparation ')
        parser.add_argument('--verbose', type=int, default=1, help='level of printing out information ')
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='initial lambda for identity loss')
        parser.add_argument('--lambda_cyc', type=float, default=10.0, help='initial lambda for cycle loss')
        parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        return opt
