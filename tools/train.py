from model import build_ssd
from data import *
from config import crack,voc, laji
from utils import MultiBoxLoss
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
import numpy as np
import datetime
import argparse
from tqdm import tqdm



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
'''
from eval import test_net
'''

torch.cuda.set_device(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

parser = argparse.ArgumentParser(description=
    'Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='LAJI', choices=['LAJI', 'VOC', 'COCO', 'CRACK', 'TRAFIC'],
                    type=str, help='VOC or COCO')
parser.add_argument('--basenet', default=None,#'vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=48, type=int,
                    help='Batch size for training')
parser.add_argument('--max_epoch', default=500, type=int,
                    help='Max Epoch for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default='',type=str,
                    help='Use visdom')
parser.add_argument('--work_dir', default='work_dir/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--weight', default=5, type=int)

parser.add_argument('--log_path', default='logs/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--net_name', default='ssd300',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_path', default='weights/train_weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--config', default='correct_transform_full_data_ciou_sgd_1gpu_6_4',
                    help='Directory for saving checkpoint models')

args = parser.parse_args()

weight = args.weight

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
import os
if not os.path.exists(args.work_dir):
    os.mkdir(args.work_dir)


def data_eval(dataset, net):
    return test_net('eval/', net, True, dataset,
             BaseTransform(trafic['min_dim'], MEANS), 5, 300,
             thresh=0.05)


def train():
    '''
    get the dataset and dataloader
    '''
    print(args.dataset)
    if args.dataset == 'COCO':
        if not os.path.exists(COCO_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')

        cfg = coco
        dataset = COCODetection(root=COCO_ROOT,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS),filename = 'train.txt')
    elif args.dataset == 'VOC':
        if not os.path.exists(VOC_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')

        cfg = voc
        dataset = VOCDetection(root=VOC_ROOT,
                               transform = SSDAugmentation(cfg['min_dim'],
                                mean = cfg['mean'],std = cfg['std']))
        print(len(dataset))

    elif args.dataset == 'LAJI':
        if not os.path.exists(LAJI_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')

        cfg = laji
        dataset = LAJIDetection(root=LAJI_ROOT, transform = SSDAugmentation(cfg['min_dim'], mean=cfg['mean'],
                                                                            std = cfg['std']))
        print(len(dataset))

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # build, load, the net
    ssd_net = build_ssd('train',size = cfg['min_dim'],cfg = cfg)
    '''
    for name,param in ssd_net.named_parameters():
        if param.requires_grad:
            print(name)
    '''
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_state_dict(torch.load(args.resume))

    if args.cuda:
        net = ssd_net.cuda()
    net.train()

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # loss:SmoothL1\Iou\Giou\Diou\Ciou
    print(cfg['losstype'])
    criterion = MultiBoxLoss(cfg = cfg,overlap_thresh = 0.5,
                            prior_for_matching = True,bkg_label = 0,
                            neg_mining = True, neg_pos = 3,neg_overlap = 0.5,
                            encode_target = False, use_gpu = args.cuda,loss_name = cfg['losstype'])

    project_name = "_".join([args.net_name, args.config])
    pth_path = os.path.join(args.save_path, project_name)
    log_path = os.path.join(pth_path, 'tensorboard')
    os.makedirs(pth_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name,epoch_size)
    iteration = args.start_iter
    step_index = 0
    loc_loss = 0
    conf_loss = 0
    step = 0
    num_iter_per_epoch = len(data_loader)
    for epoch in range(args.max_epoch):
        progress_bar = tqdm(data_loader)
        for ii, batch_iterator in enumerate(progress_bar):
            iteration += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, num_iter_per_epoch)
                # adjust_learning_rate(epoch, optimizer, args.gamma, step_index)

            # load train data
            images, targets = batch_iterator
            # print(images,targets)
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            t0 = time.time()
            out = net(images,'train')
            optimizer.zero_grad()

            loss_l, loss_c = criterion(out, targets)
            loss = weight * loss_l + loss_c
            # loss = weight * loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                    step, epoch, args.max_epoch, ii + 1, num_iter_per_epoch, loss_c.item(),
                    loss_l.item(), loss.item()))
            writer.add_scalars('Loss', {'train': loss}, step)
            writer.add_scalars('Regression_loss', {'train': loss_l.item()}, step)
            writer.add_scalars('Classfication_loss', {'train': loss_c.item()}, step)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, step)

            # print(iteration)
            # if iteration % 10 == 0:
            #     print('timer: %.4f sec.' % (t1 - t0))
            #     print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            step += 1
        # if epoch % 10 == 0 and epoch >60:
        #     # epoch>1000 and epoch % 50 == 0:
        #     print('Saving state, iter:', iteration)
        #     #print('loss_l:'+weight * loss_l+', loss_c:'+'loss_c')
        #     save_folder = args.work_dir+cfg['work_name']
        #     if not os.path.exists(save_folder):
        #         os.mkdir(save_folder)
        #     torch.save(net.state_dict(),args.work_dir+cfg['work_name']+'/ssd'+
        #                repr(epoch)+'_.pth')
            if step != 0 and step % 5000 == 0:
                torch.save(net.state_dict(), os.path.join(pth_path, f'{args.net_name}_{epoch}_{step}.pth'))

        loc_loss = 0
        conf_loss = 0
    torch.save(net.state_dict(), os.path.join(pth_path, f'{args.net_name}_{epoch}_{step}.pth'))
    # torch.save(net.state_dict(),args.work_dir+cfg['work_name']+'/ssd'+repr(epoch)+ str(args.weight) +'_.pth')


def save_checkpoint(model, pth_path, name):
    # compound_coef = name.split("_")[0].split("-")[-1]
    # file_name = "_".join([net_name, compound_coef, opt.config])
    # if not os.path.exists(os.path.join(opt.saved_path, file_name)):
    #     os.mkdir(os.path.join(opt.saved_path, file_name))

    torch.save(model.state_dict(), os.path.join(pth_path, name))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(viz,_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def create_acc_plot(viz,_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz,iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


def update_acc_plot(viz,iteration,acc, window1,update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu()*iteration,
        Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    '''
    if iteration == 0:
        print(loc, conf, loc + conf)
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )
    '''
if __name__ == '__main__':
    train()
