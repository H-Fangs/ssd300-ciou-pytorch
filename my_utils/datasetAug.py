import json
import xml.etree.ElementTree as ET
import os
from glob import glob
import copy
import cv2
import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFont

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.font_manager import FontProperties

from sklearn.cluster import KMeans
from sklearn.externals import *


class DatasetAug:
    '''
    Dataset augmentation for object detection
    '''
    def __init__(self, image_path=None, label_path=None):
        self.image_path = image_path
        self.label_path = label_path
        assert (self.label_path and self.label_path), 'label_path or image_path is None'
        assert (os.path.isdir(self.label_path) and os.path.isdir(self.image_path)), 'label path or image path not a dir'
        self.isDataAndLabelSameFile = (self.image_path == self.label_path)
        self.get_parent_path()
        self.labelTotalName = 'label_total.txt'
        self.labelAugTotalName = 'label_aug_total.txt'
        self.imageAugFileName = "_".join([os.path.dirname(self.image_path), "aug"])
        if not os.path.exists(self.imageAugFileName):
            os.makedirs(self.imageAugFileName)
        self.class_count_info = {}
        self.get_image_label_paths()
        self.getLabelToTxt()

    def get_parent_path(self):
        """
        :return: return common parent directory
        """
        tmp_image_dirname = os.path.dirname(self.image_path)
        tmp_label_dirname = os.path.dirname(self.label_path)
        while tmp_image_dirname != tmp_label_dirname:
            tmp_image_dirname = os.path.dirname(tmp_image_dirname)
            tmp_label_dirname = os.path.dirname(tmp_label_dirname)
        self.parent_path = tmp_label_dirname

    def get_label_type(self):
        """
        :return: 'label type is json or yaml or txt'
        """
        assert self.label_path, 'label_path is None'
        tmp_json = glob(os.path.join(self.label_path, "*.json"))
        tmp_xml = glob(os.path.join(self.label_path, "*.xml"))
        tmp_txt= glob(os.path.join(self.label_path, "*.txt"))
        if tmp_json:
            self.label_type = 'json'
        elif tmp_xml:
            self.label_type = 'xml'
        elif tmp_txt:
            self.label_type = 'txt'
        else:
            assert self.label_type in ('json', 'yaml', 'txt'), 'Unknown label type, Please check label type!!!'

    def get_image_label_paths(self):
        """
        get image and label path
        :return:
        """
        self.get_label_type()
        self.label_files_path = glob(os.path.join(self.label_path, "*.{}".format(self.label_type)))
        self.image_files_path = [os.path.join(self.image_path, i) for i in os.listdir(self.image_path) if os.path.join(self.label_path, i.replace("jpg", "{}".format(self.label_type))) in self.label_files_path]

    def getLabelToTxt(self):
        if self.label_type == 'xml':
            self.xmlWriteTxt()
        elif self.label_path == 'json':
            self.jsonWriteTxt()
        else:
            self.txtWriteTxt()

    def xmlWriteTxt(self):
        """
        :return: save label type image_path, [[xmin, ymin, xmax, ymax, class_name],[...]]
        """
        outfile = open(os.path.join(self.parent_path, self.labelTotalName), 'w')
        for idx, xml_path in enumerate(self.label_files_path):
            xml_tree = ET.parse(xml_path)
            data = xml_tree.findall("object")
            root = xml_tree.getroot()
            image_name = root.find("filename").text
            ans = [xml_path]
            for obj in data:
                name = obj.find('name').text
                coord = obj.find('bndbox')
                ground_truth = [coord.find("xmin").text, coord.find("ymin").text, coord.find("xmax").text, coord.find("ymax").text]
                ground_truth.extend([name])
                ans.extend(ground_truth)
                self.class_count_info[name] = self.class_count_info.get(name, 0) + 1
            outfile.write(" ".join(ans) + "\n")
            # print("{}/{} xml has done!".format(idx, len(self.label_files_path)), end='')

    def jsonWriteTxt(self):
        """
        :return: save label type image_path, [[xmin, ymin, xmax, ymax, class_name],[...]]
        """
        outfile = open(os.path.join(self.parent_path, self.labelTotalName), 'w')
        for idx, json_path in enumerate(self.label_files_path):
            with open(json_path, 'r') as f:
                json_infos = json.loads(f.read())
            ans = [os.path.join(self.image_path, json_infos['name'])]
            for json_info in json_infos:
                x, y, w, h = json_info['x'], json_info['y'], json_info['w'], json_info['h']
                xmin = x - w/2
                ymin = y - h/2
                xmax = x + w/2
                ymax = x + h/2
                tmp = [xmin, ymin, xmax, ymax, json_info['name']]
                self.class_count_info[json_info['name']] = self.class_count_info.get(json_info['name'], 0) + 1
                ans.append(tmp)
            outfile.write(" ".join(ans) + "\n")
            print("{}/{} json file has done!".format(idx, len(self.label_files_path)), end='')

    def txtWriteTxt(self):
        """
        :return: TODO
        """
        print(1)

    def data_aug(self):
        """
        imageLabelInfos = {key(image name):value(label coords)}
        :return:
        """
        max_length_class = max(self.class_count_info.values())
        self.class_aug_count = {i:math.floor(max_length_class//self.class_count_info[i]) - 1 for i in self.class_count_info}
        # hash every bbox in image
        self.imageLabelCountHash = {}
        self.imageLabelInfos = {}
        f = open(os.path.join(self.parent_path, self.labelTotalName), 'r')
        for i in f.readlines():
            data = i.strip().split(" ")
            coord = data[1:]
            coord = [coord[i:i+5] for i in range(0, len(coord), 5)]
            coord = [list(map(int, tmp[:-1])) + tmp[-1:] for tmp in coord]
            image_name = data[0].split("/")[-1].replace("xml", "jpg")
            self.imageLabelInfos[image_name] = coord
            for c in coord:
                c = list(map(str, c))
                hash_name = "_".join([image_name]+c)
                self.imageLabelCountHash[hash_name] = self.class_aug_count[c[-1]]

        # final aug_file.txt
        not_aug_name = []
        out_file = open(os.path.join(self.parent_path, self.labelAugTotalName), 'w')
        for idx, image_path in enumerate(self.image_files_path):
            image_name = image_path.split("/")[-1]
            image_coords = self.imageLabelInfos[image_name]
            image = cv2.imread(image_path)
            for coord in image_coords:
                str_coord = list(map(str, coord))
                hash_name = "_".join([image_name]+str_coord)

                if self.imageLabelCountHash[hash_name] == 0:
                    continue
                """
                random choice image for augmentation
                """
                aug_count = 1
                choice_count = 1
                while self.imageLabelCountHash[hash_name]:
                    choice_count += 1
                    choice_image_path = random.choice(self.image_files_path)
                    choice_image = cv2.imread(choice_image_path)
                    choice_image_name = choice_image_path.split("/")[-1]
                    choice_image_labels = self.imageLabelInfos[choice_image_name]

                    if choice_image_name == image_name or len(choice_image_labels) > 1:
                        continue
                    str_choice_label = list(map(str, choice_image_labels[0]))
                    choice_hash_image_name = "_".join([choice_image_name] + str_choice_label)
                    """
                    two choice, one is full use the background of choiced image, didn't aug the bbox of the choice image,
                    the other is aug the bbox of src image and choice image
                    if aug_count is greater than 100, else filled in image return 2 bbox
                    """
                    if choice_count > 2000:
                        not_aug_name.append(hash_name)
                        print(hash_name, "not aug!!!")
                        break

                    try:
                        dst_image, new_coord = self.image_roi_interpolate(image, coord[:], choice_image, choice_image_labels[0])
                        aug_count += 1
                    except:
                        continue
                    if not new_coord:
                        continue
                    elif len(new_coord) == 5:
                        new_image_name = image_name.split(".")[0] + "_aug_{}".format(self.imageLabelCountHash[hash_name]) + ".jpg"
                        new_image_path = os.path.join(self.imageAugFileName, new_image_name)
                        cv2.imwrite(new_image_path, dst_image)
                        tmp = [new_image_path]
                        tmp.extend(map(str, new_coord))
                        out_file.write(" ".join(tmp) + "\n")
                        self.imageLabelCountHash[hash_name] -= 1
                    else:
                        """
                        judge the choice_image_name needed aug or not
                        """
                        if self.imageLabelCountHash[choice_hash_image_name] == 0 or aug_count < 400:
                            continue
                        new_image_name = image_name.split(".")[0] + "_aug_{}".format(self.imageLabelCountHash[hash_name]) + ".jpg"
                        new_image_path = os.path.join(self.imageAugFileName, new_image_name)
                        cv2.imwrite(new_image_path, dst_image)
                        tmp = [new_image_path]
                        tmp.extend(map(str, new_coord))
                        out_file.write(" ".join(tmp) + "\n")
                        self.imageLabelCountHash[hash_name] -= 1
                        self.imageLabelCountHash[choice_hash_image_name] -= 1

            print("{}/{} has done".format(idx+1, len(self.image_files_path)))
        print("\n".join(not_aug_name))

    def buchong_data_aug(self, file_path):
        buchong_data_file = open(file_path, 'r')
        buchong_data = set()
        for buchong in buchong_data_file.readlines():
            buchong_data.add(buchong)

    def image_roi_interpolate(self, src_image, src_coord, dst_image, dst_coord):
        s_w, s_h = src_coord[2] - src_coord[0], src_coord[3] - src_coord[1]
        d_w, d_h = dst_coord[2] - dst_coord[0], dst_coord[3] - dst_coord[1]
        dst_h, dst_w, _ = dst_image.shape
        area_src, area_dst, area_dst_image = s_w*s_h, d_w*d_h, dst_w*dst_h
        src_roi = src_image[src_coord[1]:src_coord[3], src_coord[0]:src_coord[2]]

        if area_src < area_dst_image*0.7:
            ans_image, ans_coord = DatasetAug.imageFill(src_roi, dst_image, src_coord, dst_coord)
            if ans_coord:
                return ans_image, ans_coord
            else:
                ans_image, ans_coord = DatasetAug.mixUp(src_roi, src_coord, dst_image, dst_coord)
                if ans_coord:
                    return ans_image, ans_coord
                else:
                    return False, False
        else:
            return False, False

    @staticmethod
    def mixUp(src_roi, src_coord, dst_image, dst_coord, beta=0.5):
        """
        border_size = [top, bottom, left, right]
        :param src_roi:
        :param src_coord:
        :param dst_image:
        :param dst_coord:
        :param beta:
        :return:
        """
        s_w, s_h = src_coord[2] - src_coord[0], src_coord[3] - src_coord[1]
        d_w, d_h = dst_coord[2] - dst_coord[0], dst_coord[3] - dst_coord[1]
        dst_h, dst_w, _ = dst_image.shape
        """
        judge the area of src_roi greater than dst_roi, with and height is also greater than dst_roi 
        """
        if s_w > d_w and s_h > d_h:
            """
            if can fill without make border
            """
            if s_w < dst_w and s_h < dst_h:
                """
                if src_roi can full filled in dst_image, then fill
                """
                new_x = random.randint(max(0, dst_coord[0] + d_w - s_w), min(dst_w - s_w, dst_coord[0]))
                new_y = random.randint(max(0, dst_coord[1] + d_h - s_h), min(dst_h - s_h, dst_coord[1]))
                new_coord = [new_x, new_y, new_x+s_w, new_y+s_h, src_coord[-1]]
                dst_image[new_y:new_y+s_h, new_x:new_x+s_w, :] = src_roi

            elif s_w >= dst_w and s_h >= dst_h:
                return False, False
            else:
                """
                if s_w > dst_w, then make border x axis
                """
                if s_w > dst_w and s_h < dst_h and dst_w*2 - d_w - s_w >= 30:
                    border_size = [0, 0, dst_coord[0], dst_w - dst_coord[2]]
                    dst_image = DatasetAug.imageBordermake(dst_image, border_size, mode=random.choice([1, 2, 3]))
                    src_coord = [src_coord[0]*2, src_coord[1], src_coord[0]+src_coord[2], src_coord[3], src_coord[-1]]

                    s_w, s_h = src_coord[2] - src_coord[0], src_coord[3] - src_coord[1]
                    dst_h, dst_w, _ = dst_image.shape

                    new_x = random.randint(max(0, dst_coord[0] + d_w - s_w), min(dst_w - s_w, dst_coord[0]))
                    new_y = random.randint(max(0, dst_coord[1] + d_h - s_h), min(dst_h - s_h, dst_coord[1]))
                    new_coord = [new_x, new_y, new_x + s_w, new_y + s_h, src_coord[-1]]
                    dst_image[new_y:new_y + s_h, new_x:new_x + s_w, :] = src_roi

                elif s_w < dst_w and s_h > dst_h and dst_h*2 - d_h - s_h >= 30:
                    """
                    if s_h > dst_h, then make border y axis
                    """
                    border_size = [dst_coord[1], dst_h - dst_coord[3], 0, 0]
                    dst_image = DatasetAug.imageBordermake(dst_image, border_size, mode=random.choice([1, 2, 3]))
                    src_coord = [src_coord[0], src_coord[1]*2, src_coord[2], src_coord[3]+dst_coord[1], src_coord[-1]]

                    s_w, s_h = src_coord[2] - src_coord[0], src_coord[3] - src_coord[1]
                    dst_h, dst_w, _ = dst_image.shape

                    new_x = random.randint(max(0, dst_coord[0] + d_w - s_w), min(dst_w - s_w, dst_coord[0]))
                    new_y = random.randint(max(0, dst_coord[1] + d_h - s_h), min(dst_h - s_h, dst_coord[1]))
                    new_coord = [new_x, new_y, new_x + s_w, new_y + s_h, src_coord[-1]]
                    dst_image[new_y:new_y + s_h, new_x:new_x + s_w, :] = src_roi
                else:
                    return False, False

        elif s_w < d_w and s_h < d_h:
            """
            border_size = [top, bottom, left, right]
            judge the src_roi area if 0.7 times lower than dst_roi
            """

            # if 0.2*d_w*d_h < s_w*s_h < d_w*d_h*0.5:
            #     """
            #     fill the dst_roi by the center of dst_roi, dst_roi use gaussian_blur
            #     """
            #     new_x, new_y = dst_coord[0] + (d_w - s_w)//2, dst_coord[1] + (d_h - s_h)//2
            #     dst_roi = dst_image[dst_coord[1]:dst_coord[3], dst_coord[0]:dst_coord[2]]
            #     dst_roi = cv2.GaussianBlur(dst_roi, (13, 13), 0)
            #     dst_image[dst_coord[1]:dst_coord[3], dst_coord[0]:dst_coord[2]] = (dst_roi*(1)).astype(np.uint8)
            #     new_coord = [new_x, new_y, new_x + s_w, new_y + s_h, src_coord[-1]]
            #     new_coord.extend(dst_coord)
            #     dst_image[new_y:new_y + s_h, new_x:new_x + s_w, :] = src_roi
            #     # dst_image[new_y:new_y + s_h, new_x:new_x + s_w, :] = (src_roi*(1-beta)).astype(np.uint8)

            if s_w*s_h > d_w*d_h*0.7:
                """
                resize the src_roi to the size of dst_roi, then fill the dst_roi area
                """
                fx, fy = d_w/s_w, d_h/s_h
                src_roi = cv2.resize(src_roi, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                dst_image[dst_coord[1]:dst_coord[3], dst_coord[0]:dst_coord[2]] = src_roi
                new_coord = dst_coord
            else:
                """
                calculate the area of the four direction of the background of dst_image
                fill the max area of the calculated area, which might have a intersection
                four_area = [left_top, right_top, left_bottom, right_bottom]
                """
                border_size = [dst_coord[1], dst_h - dst_coord[3], dst_coord[0], dst_w - dst_coord[2]]
                dst_image = DatasetAug.imageBordermake(dst_image, border_size, mode=random.choice([1, 2, 3]))

                dst_coord = [dst_coord[0] + border_size[2], dst_coord[1] + border_size[0],
                             dst_coord[2] + border_size[2], dst_coord[3] + border_size[0], dst_coord[-1]]
                dst_h, dst_w, _ = dst_image.shape

                four_area = [[dst_coord[0], dst_coord[1]], [dst_w - dst_coord[2], dst_coord[1]],
                             [dst_coord[0], dst_h - dst_coord[3]], [dst_w - dst_coord[2], dst_h - dst_coord[3]]]
                max_area_index = np.argmax(list(map(lambda s:s[0]*s[1], four_area)))

                if max_area_index == 0:
                    """
                    left_top corner
                    """
                    new_init_x = random.randint(0, dst_coord[0])
                    new_init_y = random.randint(0, dst_coord[1])

                elif max_area_index == 1:
                    """
                    right_top corner
                    """
                    new_init_x = random.randint(dst_coord[2] - s_w, dst_w - s_w)
                    new_init_y = random.randint(0, dst_coord[1])

                elif max_area_index == 2:
                    """
                    left_bottom corner
                    """
                    new_init_x = random.randint(0, dst_coord[0])
                    new_init_y = random.randint(dst_coord[3] - s_h, dst_h - s_h)

                else:
                    """
                    right_bottom corner
                    """
                    new_init_x = random.randint(dst_coord[2] - s_w, dst_w - s_w)
                    new_init_y = random.randint(dst_coord[3] - s_h, dst_h - s_h)

                new_coord = [new_init_x, new_init_y, new_init_x + s_w, new_init_y + s_h, src_coord[-1]]
                new_coord.extend(dst_coord)
                dst_image[new_init_y:new_init_y + s_h, new_init_x:new_init_x + s_w] = src_roi
        else:
            return False, False
        return dst_image, new_coord

    @staticmethod
    def imageBordermake(dstImage, border_size, mode=1):
        """
        mode is border make type
        :param dstImage:
        :param size:
        :param mode:
        :return:
        """
        top_size, bottom_size, left_size, right_size = border_size
        assert mode in [1, 2, 3], "mode id error, please input again"
        make_type = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT101]
        ans_image = cv2.copyMakeBorder(dstImage, top_size, bottom_size, left_size, right_size, make_type[mode-1])
        return ans_image

    @staticmethod
    def imageFill(src_roi, dst_image, src_coord, dst_coord):
        """
        first fill the border of image.
        then, judge the background of dst_image in four direction can fill the src_roi
        if can't fill, return False
        else return filled coord
        :param dstImage:
        :param src_roi:
        :param dst_roi:
        :return:
        """
        dst_h, dst_w, _ = dst_image.shape
        border_size = [dst_coord[1], dst_h - dst_coord[3], dst_coord[0], dst_w - dst_coord[2]]
        dst_image = DatasetAug.imageBordermake(dst_image, border_size, mode=random.choice([1, 2, 3]))

        dst_coord = [dst_coord[0] + border_size[2], dst_coord[1] + border_size[0],
                     dst_coord[2] + border_size[2], dst_coord[3] + border_size[0], dst_coord[-1]]
        dst_h, dst_w, _ = dst_image.shape

        src_w, src_h = src_coord[2] - src_coord[0], src_coord[3] - src_coord[1]
        if src_w < dst_coord[0] and src_h < dst_h:
            """
            left background can fill
            """
            src_min_x, src_min_y = random.randint(0, dst_coord[0] - src_w), random.randint(0, dst_h - src_h)
        elif src_w < dst_w - dst_coord[2] and src_h < dst_h:
            """
            right background can fill
            """
            src_min_x, src_min_y = random.randint(dst_coord[2], dst_w - src_w), random.randint(0, dst_h - src_h)
        elif src_h < dst_coord[1] and src_w < dst_w:
            """
            top background can fill
            """
            src_min_x, src_min_y = random.randint(0, dst_w - src_w), random.randint(0, dst_coord[1] - src_h)
        elif src_h < dst_h - dst_coord[3] and src_w < dst_w:
            """
            bottom background can fill
            """
            src_min_x, src_min_y = random.randint(0, dst_w - src_w), random.randint(dst_coord[3], dst_h - src_h)
        else:
            return False, False
        src_max_x, src_max_y = src_min_x + src_w, src_min_y + src_h
        dst_coord.extend([src_min_x, src_min_y, src_max_x, src_max_y, src_coord[-1]])
        dst_image[src_min_y:src_max_y, src_min_x:src_max_x] = src_roi
        return dst_image, dst_coord

    @staticmethod
    def imagePinJie(src_images, src_coords, mode=1):
        ans_coord = []
        if mode == 1:
            "shuzhi pinjie"
            assert len(src_images) == 2, "src_images size not equal 2"
            image_a, image_b = src_images[0], src_images[1]
            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            coord_a, coord_b = src_coords[0], src_coords[1]

            xscale = a_w / b_w
            image_b = cv2.resize(image_b, None, fx=xscale, fy=1, interpolation=cv2.INTER_CUBIC)

            ans_image = np.vstack((image_a, image_b))
            ans_coord.append(coord_a)
            coord_b = [coord_b[i]+a_h if i%2 else coord_b[i]*xscale for i in range(4)] + [coord_b[-1]]
            ans_coord.append(coord_b)
        elif mode == 2:
            "shuiping pinjie"
            assert len(src_images) == 2, "src_images size not equal 2"
            image_a, image_b = src_images[0], src_images[1]
            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            coord_a, coord_b = src_coords[0], src_coords[1]

            yscale = a_h / b_h
            image_b = cv2.resize(image_b, None, fx=1, fy=yscale, interpolation=cv2.INTER_CUBIC)

            ans_image = np.hstack((image_a, image_b))
            ans_coord.append(coord_a)
            coord_b = [coord_b[i]*yscale if i%2 else coord_b[i]+a_w for i in range(4)] + [coord_b[-1]]
            ans_coord.append(coord_b)
        elif mode == 3:
            """
            a -- b
            c -- d
            """
            "four image pinjie"
            assert len(src_images) == 4, "src_images size not equal 4"
            image_a, image_b, image_c, image_d = src_images[0], src_images[1], src_images[2], src_images[3]
            coord_a, coord_b, coord_c, coord_d = src_coords[0], src_coords[1], src_coords[2], src_coords[3]
            a_h, a_w, _ = image_a.shape
            b_h, b_w, _ = image_b.shape
            c_h, c_w, _ = image_c.shape
            d_h, d_w, _ = image_d.shape

            byscale = b_h/a_h
            cxscale = c_w/a_w
            dxscale = d_w/a_w
            dyscale = d_h/a_h

            image_b = cv2.resize(image_b, None, fx=1, fy=byscale, interpolation=cv2.INTER_CUBIC)
            image_c = cv2.resize(image_b, None, fx=cxscale, fy=1, interpolation=cv2.INTER_CUBIC)
            image_b = cv2.resize(image_b, None, fx=dxscale, fy=dyscale, interpolation=cv2.INTER_CUBIC)

            htich = np.hstack((image_a, image_b))
            htich_2 = np.hstack((image_c, image_d))
            ans_image = np.vstack((htich, htich_2))
            ans_coord.append(coord_a)
            coord_b = [coord_b[i]*byscale if i % 2 else coord_b[i] + a_w for i in range(4)] + [coord_b[-1]]
            coord_c = [coord_c[i] + a_h if i % 2 else coord_c[i]*cxscale for i in range(4)] + [coord_c[-1]]
            coord_d = [coord_d[i]*dyscale + a_h if i % 2 else coord_d[i]*dxscale + a_w for i in range(4)] + [coord_d[-1]]
            ans_coord.extend([coord_b, coord_c, coord_d])
        else:
            return False, False
        return ans_image, ans_coord

    @staticmethod
    def imgShow(image, image_name, coord):
        for i in range(len(coord)):
            data = coord[i]
            a, b = (int(data[0]), int(data[1])), (int(data[2]), int(data[3]))
            cv2.rectangle(image, a, b, (0, 0, 255), 2)
        cv2.namedWindow(image_name, 0)
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def imageScale(image, size):
        h_s, w_s = size
        image = cv2.resize(image, None, fx=h_s, fy=w_s, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("test", image)
        cv2.waitKey(0)

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
    def showAllimage(labels_path):
        label_file = open(labels_path, 'r')
        label_data = {}
        for i in label_file.readlines():
            data = i.strip().split(" ")
            coord = data[1:]
            coord = [coord[i:i+5] for i in range(0, len(coord), 5)]
            coord = [list(map(int, tmp[:-1])) + tmp[-1:] for tmp in coord]
            image_name = data[0]
            label_data[image_name] = coord
        for image_path in label_data.keys():
            image_name = image_path.split("/")[-1]
            image = cv2.imread(image_path)
            DatasetAug.displayImage(image, image_name, label_data[image_path])
            # DatasetAug.imgShow(image, image_name, label_data[image_path])
            cv2.destroyAllWindows()

    @staticmethod
    def show(image_path, xml_path):
        image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1]
        xml_tree = ET.parse(xml_path)
        data = xml_tree.findall("object")
        root = xml_tree.getroot()
        image_name = root.find("filename").text
        ans = []
        for obj in data:
            name = obj.find('name').text
            tmp = [name]
            coord = obj.find('bndbox')
            ground_truth = [coord.find("xmin").text, coord.find("ymin").text, coord.find("xmax").text, coord.find("ymax").text]
            ground_truth = list(map(int, ground_truth))
            # ground_truth = [ground_truth[1], ground_truth[0], ground_truth[3], ground_truth[2]]
            # ground_truth = [ground_truth[1], ground_truth[0], ground_truth[3], ground_truth[2]]
            ans.append(ground_truth)
        # DatasetAug.imgShow(image, image_name, ans)
        return ans

    @staticmethod
    def modify_xml_annotation_image_name(xml_file_path):
        xml_file = os.listdir(xml_file_path)
        xml_name = []
        count = 0
        for i, file in enumerate(xml_file):
            file_path = os.path.join(xml_file_path, file)
            xml_tree = ET.parse(file_path)
            root = xml_tree.getroot()
            sub1 = root.find("filename")
            image_name = sub1.text
            file_name = file.replace('.xml', '.jpg')

            if image_name != file_name:
                count += 1
                xml_name.append(file)
                sub1.text = file_name
                xml_tree.write(file_path)
        print(count)

    @staticmethod
    def modify_xy_axis(xml_paths):
        count = 0
        for i, file in enumerate(xml_paths):
            xml_tree = ET.parse(file)
            data = xml_tree.findall("object")
            for obj in data:
                name = obj.find('name').text
                coord = obj.find('bndbox')
                ground_truth = [coord.find("xmin").text, coord.find("ymin").text, coord.find("xmax").text, coord.find("ymax").text]
                coord.find("xmin").text = ground_truth[1]
                coord.find("ymin").text = ground_truth[0]
                coord.find("xmax").text = ground_truth[3]
                coord.find("ymax").text = ground_truth[2]
            xml_tree.write(file)
            count += 1
        print(count)


    @staticmethod
    def findShowErrorImage(images_path, xmls_path):
        xml_files = os.listdir(xmls_path)
        image_files = os.listdir(images_path)
        count = 0
        for idx, xml in enumerate(xml_files):
            image_name = xml.split("/")[-1].replace("xml", "jpg")
            image_path = os.path.join(images_path, image_name)
            xml_path = os.path.join(xmls_path, xml)
            image = cv2.imread(image_path)
            image_h, image_w, _ = image.shape
            ans = DatasetAug.show(image_path, xml_path)

            for tmp in ans:
                if tmp[2] > image_w or tmp[3] > image_h or tmp[0] < 0 or tmp[1] <0:
                    print("error show image name is {}".format(image_name))
                    count += 1
            #print("{}/{} has detect".format(idx, len(xml_files)))
        print("Nums: {} image name is error".format(count))

    @staticmethod
    def findtxtclassCount(path):
        file = open(path, 'r')
        classCount = {}
        for line in file.readlines():
            data = line.strip().split(" ")
            data_name = data[0]
            data_coord = data[1:]
            data_coord = [data_coord[i:i+5] for i in range(0, len(data_coord), 5)]
            for coord in data_coord:
                classCount[coord[-1]] = classCount.get(coord[-1], 0) + 1

        print(len(classCount.keys()))
        print(classCount)

    @staticmethod
    def randomShuffleData(path, new_path):
        src_file = open(path, 'r')
        src_data = []
        for data in src_file:
            src_data.append(data)
        random.shuffle(src_data)
        dst_file = open(new_path, 'w')
        for i in src_data:
            dst_file.write(i)

    @staticmethod
    def editImageName(path, new_path):
        src_file = open(path, 'r')
        ans_data = []
        for data in src_file.readlines():
            src_data = data.strip().split(" ")
            image_name = src_data[0]
            if "xml" in image_name:
                image_name = image_name.replace("xml", "jpg")
                image_name = image_name.replace("Annotations", "JPEGImages")
                src_data[0] = image_name
            ans_data.append(src_data)
        random.shuffle(ans_data)
        dst_file = open(new_path, 'w')
        for i in ans_data:
            write_data = " ".join(i)
            dst_file.write(write_data + "\n")

    @staticmethod
    def drawWidthHight(label_path):
        file = open(label_path, 'r')
        width_height_ratio = {}
        width_height_point = {}
        all = []
        for i in file.readlines():
            data = i.strip().split(" ")[1:]
            coord = [data[i:i+5] for i in range(0, len(data), 5)]
            for c in coord:
                width = int(c[2]) - int(c[0])
                height = int(c[3]) - int(c[1])
                class_idx = c[4]
                if class_idx not in width_height_point:
                    width_height_point[class_idx] = []
                if class_idx not in width_height_ratio:
                    width_height_ratio[class_idx] = []

                width_height_ratio[class_idx].append(width/height)
                width_height_point[class_idx].append((width, height))

                all.append((width, height))


        for class_idx in width_height_point:
            array_ratio = np.array(width_height_point[class_idx])
            centers = np.array(array_ratio)

            ax1 = plt.subplot(121)
            x = array_ratio[:, 0]
            y = array_ratio[:, 1]
            ax1.scatter(x, y)

            ax2 = plt.subplot(122)
            cls = KMeans(n_clusters=6)
            s = cls.fit(centers)
            centers = cls.cluster_centers_
            color1 = '#DC143C'
            c_x = centers[:, 0]
            c_y = centers[:, 1]
            ax2.scatter(c_x, c_y, c=color1)

            font = FontProperties(fname='simkai.ttf', size=16)
            plt.title(class_idx, fontproperties=font)
            plt.legend()
            plt.savefig("save_class_count_6/{}.jpg".format(class_idx))
            plt.show()

        # data = np.array(all)
        # cls = KMeans(n_clusters=6)
        # s = cls.fit(data)
        #
        # centers = cls.cluster_centers_
        #
        # centers = np.array(centers)
        # color1 = '#DC143C'
        # c_x = centers[:, 0]
        # c_y = centers[:, 1]
        # plt.scatter(c_x, c_y, c=color1)
        # for c in centers:
        #     print(" ".join([str(i[0]/i[1]) for i in centers]))
        # # x = data[:, 0]
        # # y = data[:, 1]
        # # plt.scatter(x, y)
        #
        # font = FontProperties(fname='simkai.ttf', size=16)
        # plt.title("all", fontproperties=font)
        # plt.legend()
        # plt.savefig("save_class_count_3/{}.jpg".format("all"))
        # plt.show()

if __name__ == "__main__":
    # image_path = "img_10288.jpg"
    # image = cv2.imread(image_path)
    # cv2.imshow("image", image)
    # # # DatasetAug.imgShow(image, [0, 100, 180, 120])
    # ans = DatasetAug.imageBordermake(image, [0, 0, 0, 80], 2)
    # cv2.imshow("ans", ans)
    # cv2.waitKey(0)
    # DatasetAug.imageScale(image, [0.5, 1])

    # image_path = "/home/fangs/data/laji/image/"
    # label_path = "/home/fangs/data/laji/label/"
    # aug = DatasetAug(image_path, label_path)
    # aug.data_aug()


    image_path = "/home/fang/public_datasets/laji/trainval/VOC2007/JPEGImages/"
    label_path = "/home/fang/public_datasets/laji/trainval/VOC2007/Annotations/"
    aug = DatasetAug(image_path, label_path)
    # DatasetAug.findShowErrorImage(image_path, label_path)
    # aug.data_aug()
    aug.showAllimage("/home/fang/public_datasets/laji/trainval/VOC2007/label_total_xiugai.txt")
    # aug.showAllimage("/home/fangs/data/laji/label_aug_total.txt")

    #test = ["20190816_095522.jpg", "20190816_095426.jpg", "20190816_095644.jpg", "20190816_095457.jpg",
    #        "20190816_095748.jpg", "20190816_095633.jpg", "20190816_095553.jpg",
    #        "20190816_095611.jpg", "20190816_095538.jpg"]

    # test_0 = ["20190816_095522.jpg", "20190816_095426.jpg", "20190816_095644.jpg", "20190816_095457.jpg",
    #         "20190816_095748.jpg", "img_2205.jpg", "20190816_095633.jpg", "20190816_095553.jpg",
    #         "20190816_095611.jpg", "20190816_095538.jpg"]

    #test_image_path = "/home/fang/public_datasets/laji/trainval/VOC2007/JPEGImages/"
    #test_xml_path = "/home/fang/public_datasets/laji/trainval/VOC2007/Annotations/"
    #for name in test:
    #    image_path = os.path.join(test_image_path, name)
    #    xml_path = os.path.join(test_xml_path, name.replace("jpg", "xml"))
    #    DatasetAug.modify_xy_axis([xml_path])
    #     DatasetAug.show(image_path, xml_path)
    #     cv2.destroyAllWindows()

    # file_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_total.txt"
    # new_xiugai_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_total_xiugai.txt"
    #
    # shuffle_src_file_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_aug_huizong_total.txt"
    # shuffle_dst_file_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_aug_huizong_shuffle_total.txt"
    # #
    # DatasetAug.findtxtclassCount(shuffle_dst_file_path)


    # DatasetAug.showAllimage(file_path)
    # DatasetAug.randomShuffleData(shuffle_src_file_path, shuffle_dst_file_path)
    # DatasetAug.editImageName(file_path, new_xiugai_path)

    # label_path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_aug_total.txt"
    # DatasetAug.drawWidthHight(label_path)