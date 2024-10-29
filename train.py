import argparse
import glob, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sched import scheduler
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import *
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from data.saliency_db import *
from utils.loss import *
import cv2
from model.model import *
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser()
parser.add_argument('--kldiv', default=True, type=bool)
parser.add_argument('--cc', default=True, type=bool)
parser.add_argument('--nss', default=False, type=bool)
parser.add_argument('--sim', default=False, type=bool)
parser.add_argument('--l1', default=False, type=bool)
parser.add_argument('--kldiv_coeff', default=1.0, type=float)
parser.add_argument('--cc_coeff', default=-1.0, type=float)
parser.add_argument('--sim_coeff', default=-1.0, type=float)
parser.add_argument('--nss_coeff', default=-1.0, type=float)
parser.add_argument('--l1_coeff', default=1.0, type=float)

parser.add_argument('--no_epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--log_interval', default=10, type=int)
parser.add_argument('--no_workers', default=4, type=int)
parser.add_argument('--model_val_path', default="./saved_models/", type=str)
parser.add_argument('--clip_size', default=32, type=int)

parser.add_argument('--saliency_path', default="D:\Project\BVSCL\dataset\\", type=str)
# parser.add_argument('--load_path', type=str, default='./swin_small_patch244_window877_kinetics400_1k.pth')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--dataset', default='domain_increase', type=str)
parser.add_argument('--alternate', default=1, type=int)
args = parser.parse_args()

def train(model, optimizer, loader, epoch, device, args, writer):
    model.train()

    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for idx,(sample, target, vail) in enumerate(loader):
        img_clips = sample['rgb']
        audio = sample['audio']
        gt_sal = target['salmap'][:,0]
        gt_bin = target['binmap'][:,0]
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0, 2, 1, 3, 4))
        audio = audio.to(device)
        gt_sal = gt_sal.to(device)
        gt_bin = gt_bin.to(device)

        optimizer.zero_grad()
        z0 = model(img_clips, audio)

        assert z0.size() == gt_sal.size()
        loss = loss_func(z0, gt_sal,gt_bin, args)

        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        cur_loss.update(loss.item())

        if idx % args.log_interval == 0:
            print('epoch: {:2d}, idx: {:5d}, avg_loss: {:.3f}'.format(epoch, idx, cur_loss.avg))
            writer.add_scalar('Loss1', cur_loss.avg, global_step=epoch)
            cur_loss.reset()
            sys.stdout.flush()

    print('epoch: {:2d}, Avg_loss: {:.3f}'.format(epoch, total_loss.avg))
    writer.add_scalar('Loss2', total_loss.avg, global_step=epoch)
    sys.stdout.flush()

    return total_loss.avg


def validate(model, loader, epoch, device, args, writer):
    model.eval()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    tic = time.time()
    for idx,(sample, target, vail) in enumerate(loader):
        img_clips = sample['rgb']
        audio = sample['audio']
        gt_sal = target['salmap']
        gt_bin = target['binmap']
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0, 2, 1, 3, 4))

        pred_sal = model(img_clips, audio)

        gt_sal = gt_sal.squeeze(0).numpy()
        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()
        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

        assert pred_sal.size() == gt_sal.size()
        loss = loss_func(pred_sal, gt_sal,gt_bin, args)
        cc_loss = cc(pred_sal, gt_sal)
        sim_loss = similarity(pred_sal, gt_sal)

        total_loss.update(loss.item())
        total_cc_loss.update(cc_loss.item())
        total_sim_loss.update(sim_loss.item())

    writer.add_scalar('CC', total_cc_loss.avg, global_step=epoch)
    writer.add_scalar('SIM', total_sim_loss.avg, global_step=epoch)
    writer.add_scalar('Loss', total_loss.avg, global_step=epoch)
    print('epoch:{:2d}, avg_loss: {:.3f}, cc_loss: {:.3f}, sim_loss: {:.3f}, time: {:2f}h'.format
          (epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, (time.time() - tic) / 3600))
    sys.stdout.flush()

    return total_cc_loss.avg


if __name__ == '__main__':
    train_dataloader = get_dataloader(root=args.saliency_path, mode='train', task=args.dataset)
    val_dataloader = get_dataloader(root=args.saliency_path, mode='val', task=args.dataset)

    model = VideoSaliencyModel(pretrain=args.load_path)

    if not os.path.exists(args.model_val_path):
        os.makedirs(args.model_val_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    writer = SummaryWriter('logs')
    best_loss = 0
    for epoch in range(0, args.no_epochs):
        loss = train(model, optimizer, train_dataloader, epoch, device, args, writer)
        if epoch % 3 == 0:
            with torch.no_grad():
                cc_loss = validate(model, val_dataloader, epoch, device, args, writer)
                if epoch == 0:
                    best_loss = cc_loss
                if best_loss < cc_loss:
                    best_loss = cc_loss
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.model_val_path + 'best_Net.pth'.format(epoch))
                    else:
                        torch.save(model.state_dict(), args.model_val_path + 'best_Net.pth'.format(epoch))
    writer.close()