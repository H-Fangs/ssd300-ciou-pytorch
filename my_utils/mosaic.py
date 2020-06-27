import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from my_utils.augmentations import *
import random
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


LAJI_CLASSES = ['__background__', '一次性快餐盒','书籍纸张','充电宝','剩饭剩菜','包','垃圾桶','塑料器皿','塑料玩具','塑料衣架','大骨头','干电池','快递纸袋','插头电线'
            ,'旧衣服','易拉罐','枕头','果皮果肉','毛绒玩具','污损塑料','污损用纸','洗护用品','烟蒂','牙签','玻璃器皿','砧板','筷子','纸盒纸箱','花盆'
            ,'茶叶渣','菜帮菜叶','蛋壳','调料瓶','软膏','过期药物','酒瓶','金属厨具','金属器皿','金属食品罐','锅','陶瓷器皿','鞋','食用油桶','饮料瓶','鱼骨']


class Mosaic(object):
    def __init__(self, size, mean=(104, 117, 123), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std
        self.size = size
        self.transform = Compose([
            # PhotometricDistort(),
            RandomMirror(),
            Resize(self.size),
        ])

    def aug(self, images, boxes, labels, mode):
        """
        according mode, return the mosaic pic
        Args:
            images:
            boxes:
            labels: array style, [xmin, ymin, xmax, ymax, label(int)]
            mode: must in [2, 4]
        Returns:
        """
        assert mode in [2, 4], "mode must in [2, 4]"
        assert len(images) == len(labels), "images amount and labels amount must be equal!"
        assert len(images) == mode, "images amount must equal to the mode"
        ans_img, ans_boxes, ans_labels = [], [], []
        src_images, src_boxes, src_labels = [], [], []
        """
        transform object one by one
        """
        for i in range(len(images)):
            image, box, label = self.transform(images[i], boxes[i], labels[i])
            label = label[:, np.newaxis]
            src_images += [image]
            src_boxes += [box]
            src_labels += [label]
        if mode == 2:
            two_mode = random.randint(1, 2)
            ans_img, ans_boxes, ans_labels = self.imagePinJie(src_images, src_boxes, src_labels, two_mode)
        elif mode == 4:
            four_mode = 3
            ans_img, ans_boxes, ans_labels = self.imagePinJie(src_images, src_boxes, src_labels, four_mode)
        return ans_img, ans_boxes, ans_labels

    def imagePinJie(self, src_images, src_boxes, src_labels, mode=1):
        ans_boxes = np.zeros((0, 4))
        ans_labels = np.zeros((0, 1))
        ans_images = np.zeros((0, self.size, self.size))
        if mode == 1:
            "shuzhi pinjie"
            assert len(src_images) == 2, "src_images size not equal 2"
            image_a, image_b = src_images[0], src_images[1]
            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            boxes_a, boxes_b = src_boxes[0], src_boxes[1]
            labels_a, labels_b = src_labels[0], src_labels[1]

            xscale = a_w / b_w
            image_b = cv2.resize(image_b, None, fx=xscale, fy=1, interpolation=cv2.INTER_CUBIC)
            ans_images = np.vstack((image_a, image_b))

            ans_boxes = np.concatenate([ans_boxes, boxes_a], axis=0)
            ans_labels = np.concatenate([ans_labels, labels_a], axis=0)

            # boxes_b = np.array([boxes_b[i]+a_h if i%2 else boxes_b[i]*xscale for i in range(4)])
            boxes_b[:, [0, 2]] *= xscale
            boxes_b[:, [1, 3]] = boxes_b[:, [1, 3]] + a_h
            ans_boxes = np.append(ans_boxes, boxes_b, axis=0)
            ans_labels = np.append(ans_labels, labels_b, axis=0)

        elif mode == 2:
            "shuiping pinjie"
            assert len(src_images) == 2, "src_images size not equal 2"
            image_a, image_b = src_images[0], src_images[1]
            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            boxes_a, boxes_b = src_boxes[0], src_boxes[1]
            labels_a, labels_b = src_labels[0], src_labels[1]

            yscale = a_h / b_h
            image_b = cv2.resize(image_b, None, fx=1, fy=yscale, interpolation=cv2.INTER_CUBIC)
            ans_images = np.hstack((image_a, image_b))

            ans_boxes = np.concatenate([ans_boxes, boxes_a], axis=0)
            ans_labels = np.concatenate([ans_labels, labels_a], axis=0)

            # boxes_b = np.array([boxes_b[i]*yscale if i%2 else boxes_b[i]+a_w for i in range(4)])
            boxes_b[:, [0, 2]] = boxes_b[:, [0, 2]] + a_w
            boxes_b[:, [1, 3]] = boxes_b[:, [1, 3]]*yscale

            ans_boxes = np.append(ans_boxes, boxes_b, axis=0)
            ans_labels = np.append(ans_labels, labels_b, axis=0)

        elif mode == 3:
            """
            a -- b
            c -- d
            """
            "four image pinjie"
            assert len(src_images) == 4, "src_images size not equal 4"
            image_a, image_b, image_c, image_d = src_images[0], src_images[1], src_images[2], src_images[3]
            boxes_a, boxes_b, boxes_c, boxes_d = src_boxes[0], src_boxes[1], src_boxes[2], src_boxes[3]
            labels_a, labels_b, labels_c, labels_d = src_labels[0], src_labels[1], src_labels[2], src_labels[3]

            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            c_h, c_w, _ = image_c.shape
            d_h, d_w, _ = image_d.shape

            byscale = b_h/a_h
            cxscale = c_w/a_w
            dxscale = d_w/b_w
            dyscale = d_h/c_h

            image_b = cv2.resize(image_b, None, fx=1, fy=byscale, interpolation=cv2.INTER_CUBIC)
            image_c = cv2.resize(image_c, None, fx=cxscale, fy=1, interpolation=cv2.INTER_CUBIC)
            image_d = cv2.resize(image_d, None, fx=dxscale, fy=dyscale, interpolation=cv2.INTER_CUBIC)

            htich = np.hstack((image_a, image_b))
            htich_2 = np.hstack((image_c, image_d))
            ans_images = np.vstack((htich, htich_2))

            # for a connected
            ans_boxes = np.append(ans_boxes, boxes_a, axis=0)
            ans_labels = np.append(ans_labels, labels_a, axis=0)

            # for b connected
            # boxes_b = np.array([boxes_b[i]*byscale if i % 2 else boxes_b[i] + a_w for i in range(4)])
            boxes_b[:, [0, 2]] = boxes_b[:, [0, 2]] + a_w
            boxes_b[:, [1, 3]] = boxes_b[:, [1, 3]]*byscale
            ans_boxes = np.append(ans_boxes, boxes_b, axis=0)
            ans_labels = np.append(ans_labels, labels_b, axis=0)

            # for c connected
            # boxes_c = np.array([boxes_c[i] + a_h if i % 2 else boxes_c[i]*cxscale for i in range(4)])
            boxes_c[:, [0, 2]] = boxes_c[:, [0, 2]]*cxscale
            boxes_c[:, [1, 3]] = boxes_c[:, [1, 3]] + a_h
            ans_boxes = np.append(ans_boxes, boxes_c, axis=0)
            ans_labels = np.append(ans_labels, labels_c, axis=0)

            # for d connected
            # boxes_d = np.array([boxes_d[i]*dyscale + a_h if i % 2 else boxes_d[i]*dxscale + a_w for i in range(4)])
            boxes_d[:, [0, 2]] = boxes_d[:, [0, 2]]*dxscale + a_w
            boxes_d[:, [1, 3]] = boxes_d[:, [1, 3]]*dyscale + a_h
            ans_boxes = np.append(ans_boxes, boxes_d, axis=0)
            ans_labels = np.append(ans_labels, labels_d, axis=0)

        return ans_images, ans_boxes, ans_labels


class MosaicAug(object):
    def __init__(self, label_path, dst_path, size):
        self.labels_path = label_path
        self.dst_path = dst_path
        self.data = {}
        self.data_class_count = {}
        self.class_need_aug = {}
        self.images_need_aug = {}
        self.image_label_total = {}
        self.labels_names = {values:idx for idx, values in enumerate(LAJI_CLASSES)}
        self.load_data()
        self.mosaic = Mosaic(size=size)

    def load_data(self):
        src_labels_file = open(self.labels_path, 'r')
        for line in src_labels_file.readlines():
            line_data = line.strip().split(" ")
            if line_data is None:
                continue
            image_name = line_data[0]
            image_label_total = [int(line_data[i]) if i%5 else self.labels_names[line_data[i]] for i in range(1, len(line_data))]

            image_labels = [image_label_total[5*i:5*i+5] for i in range(len(image_label_total)//5)]
            # image label total
            self.image_label_total[image_name] = image_labels

            image_data = {}
            for t in image_labels:
                image_data[t[-1]] = image_data.get(t[-1], 0) + 1
                self.data_class_count[t[-1]] = self.data_class_count.get(t[-1], 0) + 1
            self.data[image_name] = image_data

        max_length_class = max(self.data_class_count.values())
        self.class_need_aug = {i:math.floor(max_length_class//self.data_class_count[i]) - 1 for i in self.data_class_count}

        # calculate every image aug num
        for image_name, image_data in self.data.items():
            image_data = {k:self.class_need_aug[k] for k, v in image_data.items()}
            max_item = max(list(image_data.values()))

            if max_item == 0:
                continue
            self.images_need_aug[image_name] = max_item

        print("load infos done!!!")

    def save2xml(self, save_image_path, save_xml_path, save_xml):
        """
        Args:
            save_image_path:
            save_xml_path:
            save_xml: array style

        Returns:

        """
        pic_name = save_image_path.split("/")[-1]

        node_root = Element('annotation')

        # image filename
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = pic_name

        # image size
        image = cv2.imread(save_image_path)
        height, width, depth = image.shape
        node_size = SubElement(node_root, 'size')
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(depth)

        # image segmented
        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = str(0)

        # 第二层循环遍历有多少个框
        for i in range(len(save_xml)):
            bbox = save_xml[i].tolist()
            cls_name = LAJI_CLASSES[int(bbox[-1])]

            node_object = SubElement(node_root, 'object')
            # bbox name
            node_name = SubElement(node_object, 'name')
            node_name.text = cls_name

            # bbox Unspecified
            node_pose = SubElement(node_object, 'pose')
            node_pose.text = "Unspecified"

            # bbox truncated
            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = "1"

            # bbox difficult
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = "0"

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(int(bbox[0]))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(bbox[1]))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(bbox[2]))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(bbox[3]))

        xml = tostring(node_root)
        dom = parseString(xml)
        # print xml 打印查看结果
        xml_name = os.path.join(save_xml_path, pic_name.replace(".jpg", ".xml"))
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


    def start_aug(self, aug_mode):
        self.trainval_mosaic = os.path.join(self.dst_path, "trainval_mosaic_24")
        self.trainval_mosaic_VOC2007 = os.path.join(self.trainval_mosaic, "VOC2007")
        self.trainval_mosaic_VOC2007_Annotations = os.path.join(self.trainval_mosaic_VOC2007, "Annotations")
        self.trainval_mosaic_VOC2007_ImageSets = os.path.join(self.trainval_mosaic_VOC2007, "ImageSets")
        self.trainval_mosaic_VOC2007_ImageSets_Main = os.path.join(self.trainval_mosaic_VOC2007_ImageSets, "Main")
        self.trainval_mosaic_VOC2007_JPEGImages = os.path.join(self.trainval_mosaic_VOC2007, "JPEGImages")

        os.makedirs(self.trainval_mosaic, exist_ok=True)
        os.makedirs(self.trainval_mosaic_VOC2007, exist_ok=True)
        os.makedirs(self.trainval_mosaic_VOC2007_Annotations, exist_ok=True)
        os.makedirs(self.trainval_mosaic_VOC2007_ImageSets, exist_ok=True)
        os.makedirs(self.trainval_mosaic_VOC2007_ImageSets_Main, exist_ok=True)
        os.makedirs(self.trainval_mosaic_VOC2007_JPEGImages, exist_ok=True)

        # begin aug

        done_idx = 0
        aug_idx = 0
        while len(self.images_need_aug.keys()) >= aug_mode:
            aug_mode = random.choice([2, 4])
            image_name = random.choice(list(self.images_need_aug.keys()))
            cur_image_need_aug = self.images_need_aug[image_name]
            if cur_image_need_aug == 0:
                _ = self.images_need_aug.pop(image_name)
                continue

            # update cur image
            self.images_need_aug[image_name] -= 1
            if self.images_need_aug[image_name] == 0:
                self.images_need_aug.pop(image_name)

            if len(self.images_need_aug.keys()) < aug_mode - 1:
                break

            src_images, src_boxes, src_labels = [], [], []

            # choice images
            image_count = aug_mode
            while image_count:
                choice_image_name = random.choice(list(self.images_need_aug.keys()))
                if choice_image_name == image_name:
                    continue

                choice_image = cv2.imread(choice_image_name).astype(np.float)
                choice_image_datas = np.array(self.image_label_total[choice_image_name])
                choice_image_boxes = choice_image_datas[:, :-1]
                choice_image_labels = choice_image_datas[:, -1]
                choice_image_need_aug = self.images_need_aug[choice_image_name]
                if choice_image_need_aug == 0:
                    _ = self.images_need_aug.pop(choice_image_name)
                    continue

                src_images.append(choice_image)
                src_boxes.append(choice_image_boxes)
                src_labels.append(choice_image_labels)
                image_count -= 1

                # update image_need_aug infos
                self.images_need_aug[choice_image_name] -= 1

                # pop image_name if need_aug num is zero
                if self.images_need_aug[choice_image_name] == 0:
                    self.images_need_aug.pop(choice_image_name)

            img, boxes, labels = self.mosaic.aug(src_images, src_boxes, src_labels, aug_mode)

            # save
            image_name = image_name.split("/")[-1]
            aug_image_name = image_name.split(".")[0] + "_aug_{}.jpg".format(aug_idx)
            aug_idx += 1
            save_aug_image_name = os.path.join(self.trainval_mosaic_VOC2007_JPEGImages, aug_image_name)
            cv2.imwrite(save_aug_image_name, img)

            # save coord
            save_coord = np.concatenate([boxes, labels], axis=1)

            self.save2xml(save_aug_image_name, self.trainval_mosaic_VOC2007_Annotations, save_coord)
            # MosaicAug.displayImage(img, "1", save_coord)
            # cv2.destroyAllWindows()
            done_idx += 1
            print("{}/{} has done!!".format(done_idx, len(self.image_label_total.keys())))

        print(1)

    @staticmethod
    def displayImage(image, image_name, coords):
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(img)
        for i in range(len(coords)):
            (x1, y1, x2, y2) = coords[i][:-1]
            obj = coords[i][-1]

            font = ImageFont.truetype("./simkai.ttf", 30, encoding="utf-8")
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255),      width=3)
            draw.text((x1, y1 + 10), '{}'.format(obj), (255, 0, 0), font=font)

        cv2.namedWindow(image_name, 0)
        cv2.imshow(image_name, np.array(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def read_xml(xml_path):
        xml_tree = ET.parse(xml_path)
        data = xml_tree.findall("object")
        root = xml_tree.getroot()
        image_name = root.find("filename").text
        ans = []
        for obj in data:
            name = obj.find('name').text
            coord = obj.find('bndbox')
            ground_truth = [coord.find("xmin").text, coord.find("ymin").text, coord.find("xmax").text,
                            coord.find("ymax").text]
            ground_truth = list(map(int, ground_truth))
            ground_truth.extend([name])
            ans.append(ground_truth)
        return ans

    @staticmethod
    def calculate_count(xml_path):
        xmls_path = glob(os.path.join(xml_path, "*.xml"))
        total_data = {}
        for xml_path in xmls_path:
            ans = MosaicAug.read_xml(xml_path)
            for tmp_data in ans:
                total_data[tmp_data[-1]] = total_data.get(tmp_data[-1], 0) + 1

        print(total_data)

    @staticmethod
    def showImage(xml_path, image_path):
        xmls_path = glob(os.path.join(xml_path, "*.xml"))
        for xml_path in xmls_path:
            object_name = xml_path.split("/")[-1]
            ans = MosaicAug.read_xml(xml_path)

            image_name = os.path.join(image_path, object_name.replace("xml", "jpg"))
            image = cv2.imread(image_name)

            MosaicAug.displayImage(image, image_name, ans)
            cv2.destroyAllWindows()


    @staticmethod
    def write(src_annotations_path, dst_path, ratio=0.9):
        test_file = open(os.path.join(dst_path, "test.txt"), "w")
        train_file = open(os.path.join(dst_path, "trainval.txt"), "w")

        annotations = glob(os.path.join(src_annotations_path, "*.xml"))
        random.shuffle(annotations)
        for idx, anno in enumerate(annotations):
            anno_infos = anno.split("/")[-1].split(".")[0]

            if idx < int(ratio*len(annotations)):
                train_file.write(anno_infos + "\n")

            else:
                test_file.write(anno_infos + "\n")
            print("{}/{}".format(idx, len(annotations)))


if __name__ == "__main__":
    #label_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_total_xiugai.txt"
    #dst_path = "/home/fang/public_datasets/laji/"
    #a = MosaicAug(label_path, dst_path, 512)
    #a.start_aug(aug_mode=4)

    xml_path = "/home/fang/public_datasets/laji/trainval_mosaic_24/VOC2007/Annotations/"
    # # xml_path = "/home/fang/public_datasets/laji/trainval/VOC2007/Annotations/"
    #
    image_path = "/home/fang/public_datasets/laji/trainval_mosaic_24/VOC2007/JPEGImages/"
    MosaicAug.calculate_count(xml_path)
    MosaicAug.showImage(xml_path, image_path)

    # src_annotations_path = "/home/fang/public_datasets/laji/trainval_mosaic_24/VOC2007/Annotations/"
    # dst_path = "/home/fang/public_datasets/laji/trainval_mosaic_24/VOC2007/ImageSets/Main/"
    # MosaicAug.write(src_annotations_path, dst_path)
