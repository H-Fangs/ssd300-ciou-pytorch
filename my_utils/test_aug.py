import cv2
import numpy as np
import copy
from PIL import ImageFont, Image, ImageDraw
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    '''
    calcute the intersect of box
    args:
        box_a = [boxs_num,4]
        box_b = [4]

    return iou_area = [boxs_num,1]
    '''
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """
    Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        augmentations.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    '''
    Convert the image to ints
    '''

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    '''
    Sub the image means
    '''

    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Standform(object):
    '''
    make the image to standorm
    '''

    def __init__(self, mean, std):
        self.means = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        return (image - self.means) / self.std, boxes, labels


class ToAbsoluteCoords(object):
    '''
    make the boxes to Absolute Coords
    '''

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    '''
    make the boxes to Percent Coords
    '''

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes = boxes.astype(np.float)
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    '''
    resize the image
    args:
        size = (size,size)
    '''

    def __init__(self, size=300):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise Exception("The size is int or tuple")

    def __call__(self, image, boxes=None, labels=None):
        height, width, channel = image.shape
        image = cv2.resize(image, self.size)
        if boxes is not None:
            fy = height / self.size[0]
            fx = width / self.size[0]
            boxes = boxes.astype(np.float)
            boxes[:, [0, 2]] /= fx
            boxes[:, [1, 3]] /= fy
        return image, boxes.astype(np.int), labels


class RandomSaturation(object):
    '''
    Random to change the Saturation(HSV):0.0~1.0
    assert: this image is HSV
    args:
        lower,upper is the parameter to random the saturation
    '''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    '''
    Random to change the Hue(HSV):0~360
    assert: this image is HSV
    args:
        delta is the parameters to random change the hue.

    '''

    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    '''
    change the image from HSV to BGR or from BGR to HSV color
    args:
        current
        transform
    '''

    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    '''
    Random to improve the image contrast:g(i,j) = alpha*f(i,j)
    '''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    '''
    Random to improve the image bright:g(i,j) = f(i,j) + beta
    '''

    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    '''
    change the iamge shape c,h,w to h,w,c
    '''

    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    '''
    chage the image shape h,w,c to c,h,w
    '''

    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                # calcute the center in the boxes
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                # select the valid box that center in the rect
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])

                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])

                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    '''
    expand:ratio = 0.5
    '''

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        # random to make the left and top
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        # put the image to the expand image
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        # match the box left and top
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


'''
horizontal flip: ration = 0.5
'''


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123), std=(104, 117, 123)):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            # ToPercentCoords(),
            Resize(self.size),
            # Standform(self.mean, self.std)
            # SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


def base_transform(image, size, mean):
    x = Standform(self.mean, self.std)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean, std):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            Resize(self.size),
            Standform(self.mean, self.std)

        ])

    def __call__(self, image, boxes=None, labels=None):
        return self.augment(image, boxes, labels)


class Mytransforms(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['annot']
        class_name = [i[-1] for i in label]
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=i[0], y1=i[1], x2=i[2], y2=i[3]) for i in label], shape=img.shape)
        seq = iaa.SomeOf((1, None),
                         [
                             iaa.Flipud(0.5),  # vertically flip 20% of all images
                             iaa.Fliplr(0.5),
                             iaa.Salt(0.05     ),
                             # iaa.Invert(0.25, per_channel=0.5),
                             iaa.ContrastNormalization((0.5, 1.5)),
                             iaa.GaussianBlur(sigma=1),
                             iaa.Emboss(alpha=0, strength=1, name=None, deterministic=False, random_state=None),
                             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                             iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                             iaa.Multiply((0.6, 1.5)),  # change brightness, doesn't affect BBs
                             iaa.Affine(
                                 translate_px={"x": np.random.randint(1, 30), "y": np.random.randint(1, 30)},
                                 scale=(0.8, 0.95),
                                 rotate=np.random.randint(-20, 20),
                                 shear=np.random.randint(-20, 20),
                             )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                         ])

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        image_aug_h, image_aug_w, _ = image_aug.shape
        aug_data = [[max(i.x1_int, 0), max(i.y1_int, 0), min(i.x2_int, image_aug_w), min(image_aug_h, i.y2_int), class_name[idx]]
               for idx, i in enumerate(bbs_aug.bounding_boxes)]
        sample = {'img': image_aug, 'annot':aug_data}
        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots = [(np.array(i[:-1])*scale).astype(np.int).tolist() + [i[-1]] for i in annots]
        # annots[:, :4] *= scale
        # annots[:, 4] /= np.array([self.img_size, self.img_size, self.img_size, self.img_size])
        return {'img': new_image, 'annot': annots}


class RandomRotate90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        import random
        img, label = sample['img'], sample['annot']
        class_name = [i for i in label]
        label = np.array([i[:-1] for i in label])
        H, W, C = img.shape
        if random.random() < 0.5:
            img = np.rot90(img, 1, (0, 1))
            x = copy.deepcopy(label[:, 0])
            y = copy.deepcopy(label[:, 1])
            w = copy.deepcopy(label[:, 2] - label[:, 0])
            h = copy.deepcopy(label[:, 3] - label[:, 1])
            cx = copy.deepcopy((label[:, 2] + label[:, 0]) // 2)
            cy = copy.deepcopy((label[:, 3] + label[:, 1]) // 2)

            n_cx = cy
            n_cy = W - cx

            label[:, 0] = n_cx - h // 2
            label[:, 1] = n_cy - w // 2
            label[:, 2] = n_cx + h // 2
            label[:, 3] = n_cy + w // 2

            img = np.ascontiguousarray(img)
            annots = [(np.array(label[i])).astype(np.int).tolist() + [class_name[i]] for i in range(len(label))]
            sample = {'img': img, 'annot': annots}
        return sample


class MyTest(object):
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.get_data()
        # self.transform = RandomRotate90()
        # self.transform = Resizer()
        # self.transform = Mytransforms()
        self.transform = SSDAugmentation()

    def get_transform(self, sample):
        img, label = sample['img'], sample['annot']
        boxes = np.array([i[:-1] for i in label])
        labels = np.array([i[-1] for i in label])
        img, boxes, labels = self.transform(img, boxes, labels)
        boxes = boxes.tolist()
        labels = labels.tolist()
        target = [boxes[i] + [labels[i]] for i in range(len(boxes))]
        aug = {'img': img, 'annot': target}
        return aug

    def get_data(self):
        file = open(self.path, 'r')
        for i in file.readlines():
            data = i.strip().split(" ")
            data_coords = data[1:]
            data_path = data[0]
            data_infos = [data_coords[5*i:5*i+5] for i in range(len(data_coords)//5)]

            data_infos = [list(map(int, i[:-1]))+[i[-1]] for i in data_infos]
            self.data[data_path] = data_infos

    def showImage(self, img, labels, config):
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(img)

        for j in range(len(labels)):
            (x1, y1, x2, y2) = labels[j][:-1]
            obj = labels[j][-1]

            font = ImageFont.truetype("./simkai.ttf", 20, encoding="utf-8")
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
            draw.text((x1, y1 + 10), '{}'.format(obj), (255, 0, 0), font=font)
        cv2.namedWindow(config, 0)
        cv2.imshow(config, np.array(img))

    def showAll(self):
        for img in self.data.keys():
            img_path = img
            img_name = img_path.split("/")[-1]
            img_aug_name = img_name.split(".")[0] + "_aug."+img_name.split(".")[-1]
            img_infos = self.data[img]
            img = cv2.imread(img_path)
            self.showImage(img, img_infos, img_name)

            # start augmentation
            sample = {}
            sample['img'] = copy.deepcopy(img)
            sample['annot'] = img_infos
            # aug = self.transform(sample)
            aug = self.get_transform(sample)
            aug_img = aug['img']
            aug_labels = aug['annot']
            self.showImage(aug_img, aug_labels, img_aug_name)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    a = MyTest(path = "/home/fang/public_datasets/laji/trainval/VOC2007/label_total_xiugai.txt")
    a.showAll()