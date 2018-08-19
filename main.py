import torch
from torch.backends import cudnn
import os
import argparse
from models import *
from datasets import *
from utils import makedir
from train import train

def str2bool(s):
    return s.lower() == 'true'

# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=128)

    # phase
    parser.add_argument('--phase', type=str, default='train')

    # model
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--gen_v', type=int, default=-1)
    parser.add_argument('--ext', type=str, default='vgg19')

    # train
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--num_steps_decay', type=int, default=80000)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pretrained_model', type=int, default=-1)

    parser.add_argument('--lambda_content', type=float, default=1)
    parser.add_argument('--lambda_style', type=float, default=1)

    # log
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--print_log', type=str2bool, default=True)

    parser.add_argument('--root', type=str, default='expr')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--model_path', type=str, default='model')

    config = parser.parse_args()

    # device
    config.device = torch.device('cuda:0')
    #config.select_attrs = config.select_attrs.split(',')

    # log
    try:
        config.log_path = os.path.join(config.root, config.log_path)
    except FileExistsError:
        pass
    config.model_path = os.path.join(config.root, config.model_path)
    makedir(config.log_path)
    makedir(config.model_path)

    return config

def main():
    config = get_parameter()
    cudnn.benchmark = True

    ##### build model #####
    model = {}
    model['G'] = Generator[config.gen_v](config.conv_dim, config.image_size)
    model['ext'] = Extractor()

    ##### create dataset #####
    data = FaceData(config.data_root, config.image_size)

    ##### train/test #####
    train(model, data, config)


if __name__ == '__main__':
    main()
