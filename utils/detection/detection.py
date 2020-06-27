import torch
from torch.autograd import Function
# from ..box import decode, diounms
from ..box import decode, nms, diounms

def intersect(box_a, box_b):

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]

def jaccard(box_a, box_b, iscrowd:bool=False):

    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union 

    return out if use_batch else out.squeeze(0)

def box_diou(boxes1, boxes2, beta):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    clt=torch.min(boxes1[:, None, :2], boxes2[:, :2])
    crb=torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    x1=(boxes1[:, None, 0] + boxes1[:, None, 2])/2
    y1=(boxes1[:, None, 1] + boxes1[:, None, 3])/2
    x2=(boxes2[:, None, 0] + boxes2[:, None, 2])/2
    y2=(boxes2[:, None, 1] + boxes2[:, None, 3])/2
    d=(x1-x2.t())**2 + (y1-y2.t())**2
    c=((crb-clt)**2).sum(dim=2)
    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter) - (d / c) ** beta  # iou = inter / (area1 + area2 - inter)

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh,variance, nms_kind, beta1):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_kind = nms_kind
        self.beta1 = beta1

    def forward_0(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            nms_kind: greedynms or diounms
        """
        num = loc_data.size(0) 
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            sort_scores, idx = conf_scores.sort(1, descending=True)
            c_mask = (sort_scores>=self.conf_thresh)[:,:self.top_k]

            s1,s2 = decoded_boxes.size()
            z = decoded_boxes[idx]

            h = (torch.arange(0,self.num_classes).cuda()).float().unsqueeze(1).unsqueeze(1)
            one = torch.ones(self.num_classes,s1,s2).cuda().mul(h)
            boxes = z[:,:self.top_k][c_mask]                 #[N,4] box
            z = one*2 + z

            boxes_batch = z[:,:self.top_k][c_mask]              #[N,4] box with offset

            scores = sort_scores[:,:self.top_k][c_mask]      #[N,1]
            classes = one[:,:self.top_k][c_mask][:,0]        #[N,1]
			
			# Do not support Fast NMS, due to it damages the performance.
			
            if self.nms_kind == "cluster_nms" or self.nms_kind == "cluster_weighted_nms" :
                iou = jaccard(boxes_batch, boxes_batch).triu_(diagonal=1)
            else: 
                if self.nms_kind == "cluster_diounms" or self.nms_kind == "cluster_weighted_diounms":
                    iou = box_diou(boxes_batch, boxes_batch, self.beta1).triu_(diagonal=1)
                else:
                    assert Exception("Currently, NMS only surports 'cluster_nms', 'cluster_diounms', 'cluster_weighted_nms', 'cluster_weighted_diounms'.")
            B = iou
            for j in range(999):
                A=B
                maxA=A.max(dim=0)[0]
                E = (maxA<self.nms_thresh).float().unsqueeze(1).expand_as(A)
                B=iou.mul(E)
                if A.equal(B)==True:
                    break
            keep = maxA < self.nms_thresh
            if self.nms_kind == "cluster_weighted_nms" or self.nms_kind == "cluster_weighted_diounms":
                n = len(scores)
                weights = (B*(B>0.8).float() + torch.eye(n).cuda()) * (scores.reshape((1,n)))
                xx1 = boxes[:,0].expand(n,n)
                yy1 = boxes[:,1].expand(n,n)
                xx2 = boxes[:,2].expand(n,n)
                yy2 = boxes[:,3].expand(n,n)

                weightsum=weights.sum(dim=1)
                xx1 = (xx1*weights).sum(dim=1)/(weightsum)
                yy1 = (yy1*weights).sum(dim=1)/(weightsum)
                xx2 = (xx2*weights).sum(dim=1)/(weightsum)
                yy2 = (yy2*weights).sum(dim=1)/(weightsum)
                boxes = torch.stack([xx1, yy1, xx2, yy2], 1)

            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            score_box = torch.cat((scores.unsqueeze(1),boxes), 1)

            for cl in range(1, self.num_classes):
                mask = (classes == cl)
                output[i, cl, :]=torch.cat((score_box[mask],output[i, cl, :]),0)[:self.top_k]
        return output

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            nms_kind: greedynms or diounms
        """
		
	# This funtion is no longer supported. Due to extremely time-consuming.
		
        num = loc_data.size(0) 
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class

                if self.nms_kind == "greedynms":
                    print("use greedynms")
                    ids, count = diounms(boxes, scores, self.nms_thresh, self.top_k)
                else:
                    if self.nms_kind == "diounms":
                        print("use diounms")
                        ids, count = diounms(boxes, scores, self.nms_thresh, self.top_k, self.beta1)
                    else:
                        print("use default greedy-NMS")
                        ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


    def forward_09(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            nms_kind: greedynms or diounms
        """

        # This funtion is no longer supported. Due to extremely time-consuming.

        threshold = 0.01
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        # conf_data = conf_data[:, :, 1:]
        output = torch.zeros(num, self.num_classes, num_priors, 5)
        scores = torch.max(conf_data, dim=2, keepdim=True)[0]
        tmp = scores.squeeze(0).squeeze(1)
        # print(tmp.cpu().numpy().tolist())
        scores_over_thresh = (scores > threshold)[:, :, 0]

        # Decode predictions into bboxes.
        decoded_boxes = decode(loc_data[0], prior_data, self.variance).unsqueeze(0)

        for i in range(num):
            classification_per = conf_data[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = decoded_boxes[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)

            # print(classes_.cpu().numpy().tolist())
            c_mask = [classes_ != 0][0]
            # print(c_mask.cpu().numpy().tolist())
            transformed_anchors_per = transformed_anchors_per[c_mask, :]
            scores_per = scores_per[c_mask, :]
            classes_ = classes_[c_mask]

            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=self.nms_thresh)

            if anchors_nms_idx.shape[0] != 0:
                # classes_ = classes_[c_mask]
                # scores_ = scores_[c_mask]
                # transformed_anchors_per = transformed_anchors_per[c_mask, :]

                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]

                final = torch.cat((scores_.unsqueeze(1), boxes_), 1)

                for cls in classes_.cpu().numpy().tolist():
                    output[i, cls, :anchors_nms_idx.shape[0]] = final

        return output


def batched_nms(boxes, scores, idxs, iou_threshold):

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


# def nms(bboxes, scores, iou_threshold=0.5):
#     x1 = bboxes[:,0]
#     y1 = bboxes[:,1]
#     x2 = bboxes[:,2]
#     y2 = bboxes[:,3]
#     areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
#     _, order = scores.sort(0, descending=True)    # 降序排列
#
#     keep = []
#     while order.numel() > 0:       # torch.numel()返回张量元素个数
#         if order.numel() == 1:     # 保留框只剩一个
#             i = order.item()
#             keep.append(i)
#             break
#         else:
#             i = order[0].item()    # 保留scores最大的那个框box[i]
#             keep.append(i)
#
#         # 计算box[i]与其余各框的IOU(思路很好)
#         xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
#         yy1 = y1[order[1:]].clamp(min=y1[i])
#         xx2 = x2[order[1:]].clamp(max=x2[i])
#         yy2 = y2[order[1:]].clamp(max=y2[i])
#         inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
#
#         iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
#         idx = (iou <= iou_threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
#         if idx.numel() == 0:
#             break
#         order = order[idx+1]  # 修补索引之间的差值
#     return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor