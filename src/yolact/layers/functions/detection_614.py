# import torch
import mindspore
# import torch.nn.functional as F
import mindspore.nn as nn
from mindspore import ops as P
from src.yolact.utils.ms_box_utils import jaccard
from src.config import yolact_plus_resnet50_config as cfg

import numpy as np


class Detectx(nn.Cell):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        # self.use_fast_nms = False
        self.use_fast_nms = True
        self.x_max = P.ArgMaxWithValue(0)
        self.y_max = P.ArgMaxWithValue(1)
        self.less = P.Less()
        self.less_equal = P.LessEqual()
        # self.max = P.ArgMaxWithValue(axis=1)
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.sort = P.TopK(sorted=True)
        # self.range = nn.Range()
        # self.max = P.ReduceMax()
        self.tril = nn.Tril()
        self.range = nn.Range(80)  # num_class
        self.expanddims = P.ExpandDims()
        shape = (80,200)
        self.broadcast_to = P.BroadcastTo(shape)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.concat = P.Concat(1)
        self.exp = P.Exp()

    # def __call__(self, predictions): #, net):
    def construct(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]     #  可能因为对于每张图生成的先验框的位置是相同的，故batch_size的甚至整个数据集所有图片共用先验框即可
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
            这里是pre_outs传进来的参数

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """
        # 输入：loc，conf，mask系数；prior；protomask；
        # 这一部分输出包括detect的四个输出，如下：
        # 输出：经loc转换的bbox coords；class；conf；mask系数；（result的前四个）
        # 加上  把输入的protomask赋给了result['proto'] （= proto_data[batch_idx]）
        # 所以总输出是result五元组；

        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']
        prior_data = self.cast(prior_data, mindspore.float32)
        proto_data = predictions['proto'] # if 'proto' in predictions else None

        out = []

        batch_size = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        perm = (0, 2, 1)
        x_conf_preds = self.reshape(conf_data, (batch_size, num_priors, self.num_classes))
        conf_preds = self.transpose(x_conf_preds, perm)


        for batch_idx in range(batch_size):
            # print(loc_data[batch_idx])
            decoded_boxes = self.decode(loc_data[batch_idx], prior_data)
            # print('输出decode的box')
            # print(decoded_boxes)
            result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data)
            # if result is not None and proto_data is not None:
            #       result['proto'] = proto_data[batch_idx]
            #       # print("输出proto_data[batch_idx]", (result['proto']).shape, result['proto'])  # 138 138 32
            # out.append({'detection': result, 'net': net})

            result['proto'] = proto_data[batch_idx]
            out.append({'detection': result, 'net': net})
            # print('ooooooooooooooooooooooo')
        return out

    def decode(self, loc, priors):

        variances = [0.1, 0.2]
        boxes = self.concat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, 2:] * self.exp(loc[:, 2:] * variances[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        _, conf_scores = self.x_max(cur_scores)
        # keep = (conf_scores > self.conf_thresh)
        keep = self.less(self.conf_thresh, conf_scores)
        keep = self.cast(keep, mindspore.float32)
        scores = self.mul(cur_scores, keep)
        # scores = cur_scores[:, keep]
        # boxes = decoded_boxes[keep, :]
        boxes = (self.mul((decoded_boxes).T, keep)).T
        # masks = mask_data[batch_idx, keep, :]
        masks = mask_data[batch_idx, :, :]
        masks = (self.mul((masks).T, keep)).T


        if scores.shape[1] == 0:
            return None

        # if self.use_fast_nms:  # false
        #     if self.use_cross_class_nms:  # fasle
        #         boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        #     else:
        #         boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        # else:
        #     boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh,
        #                                                          self.conf_thresh)
        #
        #     if self.use_cross_class_nms:
        #         print('Warning: Cross Class Traditional NMS is not implemented.')

        boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def cc_fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200):
        # Collapse all the classes into 1
        scores, classes = scores.max(dim=0)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        boxes_idx = boxes[idx]

        # Compute the pairwise IoU between the boxes
        iou = jaccard(boxes_idx, boxes_idx)

        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        # iou_max, _ = torch.max(iou, dim=0)
        iou_max, _ = P.Maximum(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]

        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = False):
        # scores, idx = scores.sort(1, descending=True)
        # sort = P.Sort(descending=True)
        # scores, idx = sort(scores)
        # sort 排序数量受限，用TopK可以从5w多当中取

        # sort = P.TopK(sorted=True)  # topk是按最后一个维度，就是1维，可以
        scores, idx = self.sort(scores, top_k)

        num_classes, num_dets = idx.shape

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4) # 80 200 4
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1) # 80 200 32

        iou = jaccard(boxes, boxes)  # [80,200,200]
        # 取iou上三角，只保留去掉对角线的部分，用iou-iou的下三角
        x = self.tril(iou)
        iou = iou - x

        # y_max = P.ArgMaxWithValue(axis=1)
        # _, iou_max = y_max(iou)  # [80,200]

        _, iou_max = self.y_max(iou)  # [80,200]

        # Now just filter out the ones higher than the threshold
        # keep = (iou_max <= iou_threshold)  # [80,200]
        # iou_max = self.cast(iou_max, mindspore.float32)
        keep = self.less_equal(iou_max, iou_threshold)
        keep = self.cast(keep, mindspore.int32)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.

        # if second_threshold:
        #     keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        # classes = nn.Range(num_classes, device=boxes.device)[:, None].expand_as(keep)
        # classes = nn.Range(num_classes)[:, None].expand_as(keep)

        # range = nn.Range(num_classes)
        # broadcast_to = P.BroadcastTo(keep.shape)  # expand_as 可以用expand替换  expand = broadcastto
        # classes = broadcast_to(range()[:, None])

        classes = self.broadcast_to(self.expanddims(self.range(),1))

        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        # scores = scores[keep]
        scores = self.mul(scores, keep)
        scores = scores.view(-1)
        # Only keep the top cfg.max_num_detections highest scores across all classes
        # scores, idx = scores.sort(0, descending=True)
        # sort = P.Sort(axis=0, descending=True)
        # scores, idx = sort(scores)   # 通过score比较获取最大的score的索引，用这个索引去选择classes，boxes，masks
        # idx = idx[:cfg.max_num_detections]
        # scores = scores[:cfg.max_num_detections]
        # sort = P.TopK(sorted=True)  # topk是按最后一个维度，就是1维，可以
        scores, idx = self.sort(scores, cfg['max_num_detections'])
        classes = classes.view(-1)  # 选择之前要对classses，boxes，masks做flatten
        boxes = boxes.view(-1, 4)
        masks = masks.view(-1, 32)
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

        from src.yolact.utils.cython_nms import nms as cnms

        num_classes = scores.shape[0]
        # num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            # ms.nn.Range
            # a = nn.Range(cls_scores.size(0), device=boxes.device)
            a = nn.Range(cls_scores.shape[0])
            idx = a()

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.shape[0] == 0:
                continue

            preds = P.Concat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = mindspore.Tensor(keep).long()
            # keep = mindspore.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])

        idx = P.Concat(idx_lst, dim=0)
        classes = P.Concat(cls_lst, dim=0)
        scores = P.Concat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores
