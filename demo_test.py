import datetime
import logging
import lpips
import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.nets import my_model
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args
import logging
from PIL import Image
from PIL import ImageFile
import os


def demo_test(args, TestImgLoader, model, save_path, device):
    tbar = tqdm(TestImgLoader)
    for batch_idx, data in enumerate(tbar):
        model.eval()
        test_model_fn(args, data, model, save_path, device)
        desc = 'Test demo'
        tbar.set_description(desc)
        tbar.update()


def init():
    # Make dirs
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return device


def load_checkpoint(model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        save_path = args.TEST_RESULT_DIR + '/customer'
        log_path = args.TEST_RESULT_DIR + '/customer_result.log'
    else:
        print('Please specify a checkpoint path in the config file!!!')
        raise NotImplementedError
    mkdir(save_path)
    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    model.load_state_dict(model_state_dict)

    return load_path, save_path, log_path


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def test_model_fn(args, data, model, save_path, device):
    # prepare input and forward
    in_img = data['in_img'].to(device)
    number = data['number']
    b, c, h, w = in_img.size()

    # pad image such that the resolution is a multiple of 32
    w_pad = (math.ceil(w/32)*32 - w) // 2
    h_pad = (math.ceil(h/32)*32 - h) // 2
    w_odd_pad = w_pad
    h_odd_pad = h_pad
    if w % 2 == 1:
        w_odd_pad += 1
    if h % 2 == 1:
        h_odd_pad += 1

    in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

    with torch.no_grad():
        out_1, out_2, out_3 = model(in_img)
        if h_pad != 0:
            out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

    # save images
    if args.SAVE_IMG:
        out_save = out_1.detach().cpu()
        torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)

def create_demo_dataset(
    args,
    data_path,
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                ext = filename.split(".")[-1]
                if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list


    data_files = _list_image_files_recursively(data_dir=data_path)
    dataset = demo_data_loader(data_files)

    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKER, drop_last=True
    )

    return data_loader

class demo_data_loader(data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_src = self.image_list[index]
        number = os.path.split(path_src)[-1]
        number = number.split('.')[0]

        img = Image.open(path_src).convert('RGB')
        img = default_toTensor(img)

        data['in_img'] = img
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)

def main():
    device = init()
    # load model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,
                     ).to(device)

    # load checkpoint
    load_path, save_path, log_path = load_checkpoint(model)

    # set logging for recording information or metrics
    set_logging(log_path)
    logging.warning(datetime.now())

    logging.warning('load model from %s' % load_path)
    logging.warning('save image results to %s' % save_path)
    logging.warning('save logger to %s' % log_path)

    # Create dataset
    test_path = args.DEMO_DATASET
    # Set test batch size to 1 for avoiding OOM
    args.BATCH_SIZE = 1
    DemoImgLoader = create_demo_dataset(args, data_path=test_path)

    # test demo
    demo_test(args, DemoImgLoader, model, save_path, device)


if __name__ == '__main__':
    main()


