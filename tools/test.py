from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())
import PIL
from PIL import Image,ImageDraw,ImageFont 
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import LAJI_CLASSES as labelmap
# from data import VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from model import build_ssd
from data import *
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plot
import numpy as np
import matplotlib.pyplot as plt
from config import voc, laji
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/fang/project/contest/laji/DIoU-SSD-Pytorch/tools/weights/train_weights/ssd300_ciou_sgd_1gpu_6_1/ssd300_231_92336.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='result/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--laji_root', default=LAJI_ROOT, help='Location of VOC root directory')
parser.add_argument('--visbox', default=False, type=bool, help="vis the boxes")
args = parser.parse_args()



def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    #print(img.shape)
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    #img = np.transpose(img,(1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    #label_names = ['neg','bg']
    label_names = list(labelmap) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]

        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))
        #plt.text(bb[0],bb[1],score,family='fantasy',fontsize=36,style='italic',color='mediumvioletred')
        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x,'test')      # forward pass
        detections = y.data #[batch_size,num_class,top_k,conf+locloc][1,21,200,5]
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        if args.visbox == True:
            boxs = detections[0,1:,:,:]
            #print(boxs.shape)
            boxs = boxs[boxs[:,:,0]>args.visual_threshold]
            #print(boxs.shape)
            for t in range(21):
                boxes = detections[0,t,:,:]
                for gg in range(200):
                    if boxes[gg,0]>=args.visual_threshold:
                        tt= boxes[gg,:]
                        print(tt)
                        with open(r'/mnt/home/test_ciou.txt','a') as f1:
                            f1.write(str(i))
                            f1.write(' 		')
                            f1.write(str(t))
                            f1.write(' 		')
                            f1.write(str(tt))
                            f1.write('\n')                    
                        continue
            #print(boxs)
            if boxs.shape[0] != 0:
                boxs= boxs[:,1:] 
                vis_bbox(np.array(img),boxs*scale)
                #x1=boxs[:,0]
                #y2=boxs[:,1]
                #x2=boxs[:,2]
                #y2=boxs[:,3]
                #print(y2)
                #r=boxs.shape
                #print(r[0])
 #plt.text(bb[0],bb[1],score,family='fantasy',fontsize=36,style='italic',color='mediumvioletred')
                plt.axis('off')
                plt.savefig('/mnt/home/ciou/%d.png'%(i))
                plot.show()
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                
                
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test_voc():
    # load net
    torch.cuda.set_device(1)
    net = build_ssd('test', 300, laji) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location="cuda:1"))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = LAJIDetection(args.laji_root, [('2007', 'test')],
                           BaseTransform(300, laji['mean'],laji['std']))
    if args.cuda:
        net = net.cuda()
        #torch.backends.cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(300, laji['mean'],laji['std']),
             thresh=args.visual_threshold)



if __name__ == '__main__':
    
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    test_voc()
