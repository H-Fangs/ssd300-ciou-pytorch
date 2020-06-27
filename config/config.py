# config.py
import os.path

# gets home dir cross platform
#HOME = os.path.expanduser("~")
#HOME = os.path.abspath(os.path.dirname(__file__)).split("/") this path
HOME = os.path.join(os.getcwd()) #../path
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

#crcak (104, 117, 123)

# SSD300 CONFIGS


voc= {
    'model':"resnet50",
    'losstype':'Giou',
    'num_classes':21,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0,1.0,1.0),#(58.395, 57.12, 57.375),
    'lr_steps': (80000, 100000,120000),
    'max_iter': 120000,
    'max_epoch':80,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_diounms",       #Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1':0.5,
    'name': 'VOC',
    'work_name':"SSD300_VOC_FPN_GIOU",
}

laji= {
    # 'model':"senet154",
    'model':"resnet50",
    'losstype':'Ciou',
    'num_classes':45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0, 1.0, 1.0),
    # 'std':(58.395, 57.12, 57.375),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'max_epoch':500,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_diounms",       # Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI',
    'work_name':"SSD300_LAJI_FPN_GIOU",
}

laji_se_resnext50_32x4d= {
    'model':"se_resnext50_32x4d",
    # 'model':"resnet50",
    'losstype':'Ciou',
    'num_classes':45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0, 1.0, 1.0),
    # 'std':(58.395, 57.12, 57.375),
    'lr_steps': (70, 130, 200),
    'max_iter': 120000,
    'max_epoch':500,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_nms",       # Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI',
    'work_name':"SSD300_LAJI_FPN_GIOU",
}

laji_se_resnext101_32x4d= {
    'model':"se_resnext101_32x4d",
    # 'model':"resnet50",
    'losstype':'Ciou',
    'num_classes':45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0, 1.0, 1.0),
    # 'std':(58.395, 57.12, 57.375),
    'lr_steps': (90000, 130000, 160000),
    'max_iter': 120000,
    'max_epoch':500,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    # 'nms_kind': "greedynms",       # Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'nms_kind': "cluster_weighted_nms",       # Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI',
    'work_name':"SSD300_LAJI_FPN_GIOU",
}

laji512_se_resnext101_32x4d= {
    'model':"se_resnext101_32x4d",
    'losstype':'Ciou',
    'num_classes': 45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0,1.0,1.0), # (58.395, 57.12, 57.375),
    'lr_steps': (152500, 230000, 287500),
    'max_iter': 400000,
    'max_epoch': 500,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'backbone_out':[512,1024,2048,512,256,256,256],
    'neck_out':[256,256,256,256,256,256, 256],
    'steps':[8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 51, 133, 215, 296, 378, 460],
    'max_sizes': [51, 133, 215, 296, 378, 460, 542],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_diounms",       #Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI',
    'work_name':"SSD300_LAJI_FPN_GIOU",
}


laji512_se_resnext50_32x4d= {
    'ssd':"se_resnext50_32x4d",
    'losstype':'Ciou',
    'num_classes': 45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0,1.0,1.0), # (58.395, 57.12, 57.375),
    'lr_steps': (152500, 230000, 287500),
    'max_iter': 400000,
    'max_epoch': 500,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'backbone_out':[512,1024,2048,512,256,256,256],
    'neck_out':[256,256,256,256,256,256, 256],
    'steps':[8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 51, 133, 215, 296, 378, 460],
    'max_sizes': [51, 133, 215, 296, 378, 460, 542],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_diounms",       #Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI',
    'work_name':"SSD300_LAJI_FPN_GIOU",
}

laji512 = {
    # 'model':"senet154",
    'model':"resnet50",
    'losstype':'Ciou',
    'num_classes':45,
    'mean':(123.675, 116.28, 103.53),
    'std':(1.0, 1.0, 1.0),
    # 'std':(58.395, 57.12, 57.375),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'max_epoch':500,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,

    'backbone_out':[256, 512,1024,2048,512,256,256],
    'neck_out':[256, 256,256,256,256,256,256],

    'steps':[8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 51, 133, 215, 296, 378, 460],
    'max_sizes': [51, 133, 215, 296, 378, 460, 542],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "cluster_weighted_diounms",       # Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'
    'beta1': 0.5,
    'name': 'LAJI512',
    'work_name':"SSD512_LAJI_FPN_GIOU",
}

crack = {
    'model':"resnet50",
    'num_classes': 2,
    'mean':(127.5, 127.5, 127.5),
    'std':(1.0, 1.0, 1.0),
    'lr_steps': (25000, 35000, 45000),
    'max_iter': 50000,
    'max_epoch':2000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # [[1/0.49], [1/0.16,1/0.09], [1/0.16,1/0.09], [1/0.16,1/0.09], [1/0.09], [1/0.09]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CRACK',
    'work_name':"SSD300_CRACK_FPN_GIOU",
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'max_epoch':80,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}



trafic = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 12000,
    'max_epoch':2000,
    'feature_maps': [50, 25, 13, 7, 5, 3],
    'min_dim': 800,
    'steps': [16, 32, 64, 100, 300, 600],
    'min_sizes': [16, 32, 64, 128, 256, 512],
    'max_sizes': [32, 64, 128, 256, 512, 630],
    'aspect_ratios': [[1], [1,1/2], [1/2,1], [1/2,1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'TRAFIC',
}
