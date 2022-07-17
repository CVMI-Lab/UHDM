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
import lpips
from config.config import args

def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    """
    Training Loop for each epoch
    """
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss = model_fn(args, data, model, iters)
        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx+1)
        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    # the learning rate is adjusted after each epoch
    lr_scheduler.step()

    return lr, avg_train_loss, iters

def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

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

    # summary writer
    logger = SummaryWriter(args.LOGS_DIR)
    
    return logger, device

def load_checkpoint(model, optimizer, load_epoch):
    load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    model.load_state_dict(model_state_dict)
    optimizer_dict = torch.load(load_dir)['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters

def main():
    logger, device = init()
    # create model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,
                     ).to(device)
    model._initialize_weights()

    # create optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    # resume training
    if args.LOAD_EPOCH:
        learning_rate, iters = load_checkpoint(model, optimizer, args.LOAD_EPOCH)
    # create loss function
    loss_fn = multi_VGGPerceptualLoss(lam=args.LAM, lam_p=args.LAM_P).to(device)
    # create learning rate scheduler
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)
    # create training function
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device)
    # create dataset
    train_path = args.TRAIN_DATASET
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train')

    # start training
    print("****start traininig!!!****")
    avg_train_loss = 0
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch,
                                                           iters, lr_scheduler)
        logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
        logger.add_scalar('Train/learning_rate', learning_rate, epoch)

        # Save the network per ten epoch
        if epoch % 10 == 0:
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }, savefilename)

        # Save the latest model
        savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }, savefilename)


if __name__ == '__main__':
    main()
