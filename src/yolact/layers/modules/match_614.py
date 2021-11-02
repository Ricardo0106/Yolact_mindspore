import os
import mindspore.ops as P
import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.dataset import context
from mindspore import Tensor
from src.config import yolact_plus_resnet50_config as cfg

class match(nn.Cell):
    def __init__(self):  # ,batch_size,num_priors):
        super(match, self).__init__()
        self.cast = P.Cast()
        self.argmaxwithV0 = P.ArgMaxWithValue(0)
        self.argmaxwithV1 = P.ArgMaxWithValue(1)
        self.scalarcast = P.ScalarCast()
        self.squeeze = P.Squeeze()
        self.concat = P.Concat(1)
        self.log = P.Log()
        self.squeeze0 = P.Squeeze(0)
        self.squeeze2 = P.Squeeze(2)
        self.expand_dims = P.ExpandDims()
        # shape = (1,4,10)
        shape = (1, 128, 57744)  # 57744
        self.broadcast_to1 = P.BroadcastTo(shape)
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.prod = P.ReduceProd()
        self.zeroslike = P.ZerosLike()
        self.select = P.Select()
        shape2 = (1, 128, 57744, 2)  # 57744
        # shape2 = (1,4,10,2)
        self.broadcast_to2 = P.BroadcastTo(shape2)
        self.update = self.expand_dims(mindspore.Tensor(2, mindspore.float16), 0)
        self.range_op_1 = mindspore.nn.Range(0, 128, 1)
        self.range_op_2 = mindspore.nn.Range(0, 57744, 1)  # 19248
        # self.range_op_1 = mindspore.nn.Range(0, 4, 1)
        # self.range_op_2 = mindspore.nn.Range(0, 10, 1)  # 19248

        shape_crowd = (1, 57744, 10)
        # shape_crowd = (1, 10, 10)
        self.broadcast_to1_crowd = P.BroadcastTo(shape_crowd)
        shape2_crowd = (1, 57744, 10, 2)
        # shape2_crowd = (1, 10, 10, 2)
        self.broadcast_to2_crowd = P.BroadcastTo(shape2_crowd)
        self.reducesum = P.ReduceSum()
        self.oneslike = P.OnesLike()
        self.logicalAnd = P.LogicalAnd()
        self.iou = mindspore.ops.IOU()

    def point_form(self, boxes):
        # concat = P.Concat(1)
        return self.concat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                            boxes[:, :2] + boxes[:, 2:] / 2))  # xmax, ymax

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
        inter = self.intersect(box_a, box_b)  # 1 128 4 ，1 19248，4  ---> 1 128 19248
        area_a = (box_a[:, :, 2:3:1] - box_a[:, :, 0:1:1]) * (box_a[:, :, 3:4:1] - box_a[:, :, 1:2:1])
        area_a = self.broadcast_to1(area_a)
        area_b = self.expand_dims(self.squeeze2((box_b[:, :, 2:3:1] - box_b[:, :, 0:1:1]) *
                                                (box_b[:, :, 3:4:1] - box_b[:, :, 1:2:1])),
                                  1)
        area_b = self.broadcast_to1(area_b)
        union = area_a + area_b - inter
        out = inter / area_a if iscrowd else inter / union

        return out if use_batch else self.squeeze(out)

    def intersect(self, box_a, box_b):

        max_xy = self.min(self.broadcast_to2(self.expand_dims(box_a[:, :, 2:], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, 2:], 1)))
        min_xy = self.max(self.broadcast_to2(self.expand_dims(box_a[:, :, :2], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, :2], 1)))
        min_tensor = self.zeroslike(max_xy - min_xy)
        wants = self.select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
        return self.prod(wants, 3)  # inter

    def jaccard_crowd(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
        inter = self.intersect_crowd(box_a, box_b)  # 1 128 4 ，1 19248，4  ---> 1 128 19248
        area_a = (box_a[:, :, 2:3:1] - box_a[:, :, 0:1:1]) * (box_a[:, :, 3:4:1] - box_a[:, :, 1:2:1])
        area_a = self.broadcast_to1_crowd(area_a)
        area_b = self.expand_dims(self.squeeze2((box_b[:, :, 2:3:1] - box_b[:, :, 0:1:1]) *
                                                (box_b[:, :, 3:4:1] - box_b[:, :, 1:2:1])),1)
        area_b = self.broadcast_to1_crowd(area_b)
        union = area_a + area_b - inter
        out = inter / area_a if iscrowd else inter / union

        return out if use_batch else self.squeeze(out)

    def intersect_crowd(self, box_a, box_b):

        n = box_a.shape[0]  # 1
        A = box_a.shape[1]  # 57744
        B = box_b.shape[1]  # 10
        max_xy = self.min(self.broadcast_to2_crowd(self.expand_dims(box_a[:, :, 2:], 2)),
                          self.broadcast_to2_crowd(self.expand_dims(box_b[:, :, 2:], 1)))
        min_xy = self.max(self.broadcast_to2_crowd(self.expand_dims(box_a[:, :, :2], 2)),
                          self.broadcast_to2_crowd(self.expand_dims(box_b[:, :, :2], 1)))
        min_tensor = self.zeroslike(max_xy - min_xy)
        wants = self.select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
        return self.prod(wants, 3)  # inter

    def encode(self, matched, priors):

        # 编码gtbox对应的坐标和先验框对应的坐标
        variances = [0.1, 0.2]
        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = self.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = self.concat((g_cxcy, g_wh))  # [num_priors,4]

        return loc

    def construct(self, pos_thresh, neg_thresh, truths, priors, labels,
                 crowd_boxes):
        # truths 128 4
        # priors 19248 4
        decoded_priors = self.point_form(priors)
        overlaps = self.jaccard(truths, decoded_priors)
        # overlaps = self.iou(decoded_priors, truths)
        overlaps = self.cast(overlaps, mindspore.float32)
        best_truth_idx, best_truth_overlap = self.argmaxwithV0(overlaps)

        # 不能加if，用select重写
        best_truth_idx = self.cast(best_truth_idx, mindspore.float16)
        best_truth_overlap = self.expand_dims(best_truth_overlap, 0)
        best_truth_overlap = self.cast(best_truth_overlap, mindspore.float16)
        best_truth_idx = self.expand_dims(best_truth_idx, 0)

        x_idx = P.Tile()(P.ExpandDims()(self.range_op_1(), 1), (1, 57744))
        z_idx = P.Tile()(self.range_op_2(), (128, 1))
        # x_idx = P.Tile()(P.ExpandDims()(self.range_op_1(), 1), (1, 10))
        # z_idx = P.Tile()(self.range_op_2(), (4, 1))
        minus_one = P.OnesLike()(overlaps) * -1

        for _ in range(overlaps.shape[0]):

            best_prior_idx, best_prior_overlap = self.argmaxwithV1(overlaps)  # int32 float32
            idx_j, out_j = self.argmaxwithV0(best_prior_overlap)  # int32 float32
            i = best_prior_idx[idx_j]
            overlaps = self.select(x_idx == idx_j, minus_one, overlaps)
            overlaps = self.select(z_idx == i, minus_one, overlaps)
            idx_j = self.cast(idx_j, mindspore.float16)
            out_j = self.expand_dims(out_j, 0)
            best_truth_overlap[::, i] = self.select(out_j > 0, self.update, best_truth_overlap[::, i])
            idx_j = self.expand_dims(idx_j, 0)
            best_truth_idx[::, i] = self.select(out_j > 0, idx_j, best_truth_idx[::, i])
        best_truth_idx = self.squeeze(best_truth_idx)
        best_truth_overlap = self.squeeze(best_truth_overlap)
        best_truth_idx = self.cast(best_truth_idx, mindspore.int32)
        matches = truths[best_truth_idx]  # Shape: [num_priors,4]
        conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
        conf[best_truth_overlap < pos_thresh] = -1
        conf[best_truth_overlap < neg_thresh] = 0  # label as background

        c = crowd_boxes[0]>0
        c = self.cast(c, mindspore.float16)
        c = self.reducesum(c)
        if cfg['crowd_iou_threshold'] < 1:
            crowd_overlaps = self.jaccard_crowd(decoded_priors, crowd_boxes, iscrowd=True)  # 10 10
            best_crowd_idx, best_crowd_overlap = self.argmaxwithV1(crowd_overlaps)
            minus_one2 = self.oneslike(conf) * -1
            cond = self.logicalAnd(conf <= 0, best_crowd_overlap>cfg['crowd_iou_threshold'])
            cond = self.logicalAnd(cond, c>0)
            conf = self.select(cond,minus_one2,conf)

        loc = self.encode(matches, priors)
        best_truth_idx = self.cast(best_truth_idx, mindspore.int32)

        return loc, conf, best_truth_idx


# test code
# 用这个测试代码上面的shape1和shape2要更改

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', default=4))
    # GRAPH_MODE PYNATIVE_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    img = Tensor(np.random.randint(0, 255, (1, 3, 550, 550)).astype(np.float32))
    pos = 0.5
    neg = 0.4
    # a = Tensor(np.random.rand(128,4).astype(np.float32))
    a = Tensor(np.array([[1.3, 0.4, 5.2, 3.1], [2.3, 1.4, 6.2, 7.1], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float32))
    b = Tensor(np.random.rand(10, 4).astype(np.float32))
    c = Tensor(np.random.randint(1, 80, (4,)).astype(np.float32))
    e = Tensor(np.random.rand(1, 10, 4).astype(np.float32))
    g = Tensor(np.random.rand(1, 10).astype(np.float32))
    h = Tensor(np.random.rand(1, 10).astype(np.float32))
    idx = 0
    num_crowd = 1
    j = Tensor(np.random.rand(10, 4).astype(np.float32))
    mt = match()
    outs = mt(pos, neg, a, b, c, j)
    print(outs)
