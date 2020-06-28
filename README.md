## “华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类-分享

## Details
* Dataaug: edit resize and mirror augment method  
* Backbone: se_resnext101_32x4d  
* neck: fpn  
* loss: ciou-loss  

## Test result  
* [weight path](https://pan.baidu.com/s/1myke6blQqFD1dx4SuYRBEg)(提取码：lnow)
* put the weight in **weights/train_weights/ssd300_se_resnext101_32x4d_correct_transform_full_data_ciou_sgd_1gpu_6_6/**, than run the **lajiEval.py**

## Results  
* map 72.66  

## Reference
>[Zzh-tju/DIoU-SSD-pytorch](https://github.com/Zzh-tju/DIoU-SSD-pytorch)
