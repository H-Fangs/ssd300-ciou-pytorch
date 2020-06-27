from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import cv2


def make_xml(f, save_xml_path, save_image_path, save_image_name_path):
    # 第一层循环遍历所有的照片
    train_file = open(os.path.join(save_image_name_path, "trainval.txt"), 'w')
    test_file = open(os.path.join(save_image_name_path, "test.txt"), 'w')

    total_data = {}
    for idx, line in enumerate(f):
        lines = str(line).strip().split(" ")
        bbox_num = (len(lines)-1)//5
        if len(lines) != bbox_num*5 + 1:
            print(lines[0], "label not correct!!")
            continue
        tmp_pic_path = str(lines[0])
        tmp_pic_data = lines[1:]
        tmp_pic_name = tmp_pic_path
        total_data[tmp_pic_name] = total_data.get(tmp_pic_name, []) + tmp_pic_data

    for idx, name in enumerate(total_data.keys()):
        box_num = len(total_data[name])//5

        if len(total_data[name]) != box_num*5:
            print(lines[0], "label not correct!!")
            continue
        pic_path = str(name)
        pic_data = total_data[name]
        pic_name = name.split("/")[-1]

        node_root = Element('annotation')

        # image filename
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = pic_name

        # image size
        image = cv2.imread(pic_path)
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
        for i in range(box_num):
            bbox = pic_data[i*5:5*i+5]
            cls_name = bbox[-1]

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
        xml_name = pic_name.replace(".jpg", "")
        xml_name = os.path.join(save_xml_path, xml_name + '.xml')
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

        # save image
        cv2.imwrite(os.path.join(save_image_path, pic_name), image)

        # save image_name
        if idx < int(len(total_data)*0.85):
            train_file.write(pic_name.split(".")[0] + "\n")
        else:
            test_file.write(pic_name.split(".")[0] + "\n")

        print("{}/{} has done!".format(idx, len(total_data)))


txt_path = '/home/fang/public_datasets/laji/trainval/VOC2007/label_aug_huizong_shuffle_total.txt'
save_xml_path = '/home/fang/public_datasets/laji/aug_trainval/VOC2007/Annotations'
save_image_path = "/home/fang/public_datasets/laji/aug_trainval/VOC2007/JPEGImages"
save_image_name_path = "/home/fang/public_datasets/laji/aug_trainval/VOC2007/ImageSets/Main"

f = open(txt_path, 'r')
f = f.readlines()
make_xml(f, save_xml_path, save_image_path, save_image_name_path)
