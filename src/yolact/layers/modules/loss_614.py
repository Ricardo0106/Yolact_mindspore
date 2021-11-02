# -*- coding: utf-8 -*-
import mindspore
import mindspore.ops as P
import mindspore.nn as nn
from src.config import mask_type
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from src.yolact.layers.modules.match_614 import match
from src.yolact.utils.ms_box_utils import center_size, crop
from src.config import yolact_plus_resnet50_config as cfg

class MultiBoxLoss(nn.Cell):

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio,batch_size,num_priors):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.sum = P.ReduceSum(True)
        self.sum_f = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.concat = P.Concat()
        self.concat2 = P.Concat(2)
        self.squeeze0 = P.Squeeze(0)
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.max = P.ReduceMax()
        self.transpose = P.Transpose()
        self.softmax = P.Softmax()
        self.exp = P.Exp()
        self.log = P.Log()
        self.topk = P.TopK(sorted=True)
        self.reverse = P.ReverseV2(axis=[1])
        self.oneslike = P.OnesLike()
        self.select = P.Select()
        self.less = P.Less()
        self.matmul = P.MatMul()
        self.cat = P.Concat(2)  # 1  这里出问题了试一试0
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction='sum')
        self.sigmoid = P.Sigmoid()
        self.resize_bilinear = nn.ResizeBilinear()
        self.scalarCast = P.ScalarCast()
        self.smoothL1loss = nn.SmoothL1Loss()
        self.closs = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(reduction='sum')
        # mask
        self.matmulnn = nn.MatMul(False, True)
        self.gather = P.Gather()
        self.binary_cross_entropy_n = P.BinaryCrossEntropy(reduction='none')
        self.maximum = P.Maximum()
        self.zeroslike = P.ZerosLike()
        self.oneslike = P.OnesLike()
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        self.batch_size = batch_size
        self.num_priors = num_priors

        self.min_value = mindspore.Tensor(0, mindspore.float32)
        self.max_value = mindspore.Tensor(1, mindspore.float32)

        self.match_test = match()
        self.gatherD = P.GatherD()

        self.maskiounet = mask_iou_net()
        self.randChoicewithmask = P.RandomChoiceWithMask(cfg['masks_to_train'])

    def construct(self, predictions, gt_bboxes, gt_labels, crowd_boxes, masks, num_crowd):

        loc_data = predictions['loc']  # b 57744 4
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors = predictions['priors']
        proto_data = predictions['proto']     # b 138 138 32
        gt_labels = self.cast(gt_labels, mindspore.int32)
        labels = F.stop_gradient(gt_labels)
        loc_t = ()
        conf_t = ()
        idx_t = ()
        gt_box_t = ()

        for idx in range(self.batch_size):

            truths = gt_bboxes[idx]  # 128,4
            labels_idx = labels[idx]
            truths = F.stop_gradient(truths)
            priors_t = F.stop_gradient(priors)
            cur_crowd_boxes = crowd_boxes[idx]
            loc_idx, conf_idx,idx_idx = self.match_test(
                self.pos_threshold, self.neg_threshold,truths, priors_t, labels_idx,cur_crowd_boxes)
            loc_t += (self.expand_dims(loc_idx, 0),)
            conf_t += (self.expand_dims(conf_idx, 0),)
            idx_t += (self.expand_dims(idx_idx, 0),)
            gt_box_t += (self.expand_dims(truths[idx_idx], 0),)

        loc_t = self.concat(loc_t)
        conf_t = self.concat(conf_t)
        idx_t = self.concat(idx_t)
        gt_box_t = self.concat(gt_box_t)
        pos = conf_t > 0
        pos = self.cast(pos, mindspore.float16)
        num_pos = self.sum_f(pos)
        num_pos = F.stop_gradient(num_pos)
        pos = self.cast(pos, mindspore.bool_)
        pos_idx = self.expand_dims(self.cast(pos, mindspore.int32), 2).expand_as(loc_data)
        pos_idx = self.cast(pos_idx, mindspore.bool_)
        print("loc坐标：")
        print(loc_t)
        print(loc_data)
        # Localization Loss (Smooth L1)
        loss_B = 0
        if cfg['train_boxes']:  # true
            pos_idx = self.cast(pos_idx, mindspore.float16)
            loss_B = self.sum_f((self.smoothL1loss(loc_data, loc_t) * pos_idx)) * cfg['bbox_alpha'] / num_pos
        # print("loc loss:",loss_B)
        # Segm loss
        loss_S = 0
        if cfg['use_semantic_segmentation_loss']:  # true
            loss_S = self.semantic_segmentation_loss(predictions['segm'], masks, labels)
        # print("seg loss:", loss_S)
        # Mask loss
        maskiou_net_input = 0
        maskiou_t = 0
        label_t = 0
        mask_t = 0
        ret = 0

        if cfg['train_masks'] and cfg['mask_type'] == mask_type['lincomb']:  # true
            ret = self.lincomb_mask_loss(pos, idx_t, mask_data, proto_data, masks, gt_box_t, labels)

        # loss_M = 0
        if cfg['use_maskiou']:
            loss_M, maskiou_net_input, maskiou_t, label_t, mask_t = ret
        else:
            loss_M = ret
        # print("mask loss",loss_M)

        # Mask IoU Loss
        loss_I = 0
        if cfg['use_maskiou']:
            loss_I = self.mask_iou_loss(maskiou_net_input, maskiou_t, label_t, mask_t)
        # print("iou loss:", loss_I)

        # Confidence loss
        loss_C = self.ohem_conf_loss(conf_data, conf_t, pos, self.batch_size)

        self.print_loss(loss_B,loss_C,loss_M,loss_S,loss_I)

        return loss_B, loss_C, loss_M, loss_S, loss_I


    def print_loss(self, loss1,loss2,loss3,loss4,loss5):

        print("loss_B", loss1)
        print("loss_C", loss2)
        print("loss_M", loss3)
        print("loss_S", loss4)
        print("loss_I", loss5)
        return 0
    # 分割损失
    def semantic_segmentation_loss(self, segment_data, mask_t, class_t):

        batch_size, num_classes, mask_h, mask_w = segment_data.shape  # 1，80，69，69
        loss_s = 0
        for idx in range(batch_size):
            cur_segment = self.squeeze0(segment_data[idx:idx+1:1,:,:,:])
            cur_class_t = class_t[idx]
            em = mask_t[idx:idx+1:1, :, :, :]
            cm = self.cast(em, mindspore.float16)
            downsampled_masks = self.resize_bilinear(cm, (self.scalarCast(mask_h, mindspore.int32),
                                                          self.scalarCast(mask_w, mindspore.int32)))
            downsampled_masks = self.squeeze0(downsampled_masks)  # 128, 69,69
            downsampled_masks = self.cast(self.less(0.5, downsampled_masks),mindspore.float16)
            segment_t = self.zeroslike(cur_segment)  # 80,69,69

            # 用mask_t构造segment_t
            for obj_idx in range(downsampled_masks.shape[0]):  # pad后是128
                i = self.cast(cur_class_t[obj_idx], mindspore.int32)
                j = self.select(i > 0, i, i+2)
                segment_t[j, :, :] = self.maximum(segment_t[j, :, :], downsampled_masks[obj_idx, :, :])

            loss_s += self.bcewithlogitsloss(cur_segment, segment_t)
        return loss_s / mask_h / mask_w * cfg['semantic_segmentation_alpha'] / batch_size
    # 分类损失
    def ohem_conf_loss(self, conf_data, conf_t, pos, num):

        batch_conf = conf_data.view(-1, self.num_classes)

        x_max = self.max(self.max(batch_conf, 1), 0)  # x_max: 2.003
        loss_c = self.log(self.sum(self.exp(batch_conf - x_max), 1)) + x_max - batch_conf[::, 0:1:1]
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        loss_c[conf_t < 0] = 0
        # 由于先验框数量多，要用topk排序，又不做筛选，所以要用k，手动设置
        # k = 19248
        k = 57744
        loss_idx = self.topk(loss_c, k)
        _, idx_rank = self.topk(self.cast(loss_idx[1], mindspore.float16), k)#_, idx_rank = self.topk(self.cast(loss_idx[1], mindspore.float32), k)
        idx_rank = self.reverse(idx_rank)
        m = self.cast(pos, mindspore.float16)# m = self.cast(pos, mindspore.float32)
        num_pos = self.sum(m, 1)# num_pos = self.sum(m, 1)
        num_neg = self.negpos_ratio * num_pos

        neg_shape = idx_rank.shape
        broadcast_to1 = P.BroadcastTo(neg_shape)
        num_neg = self.cast(num_neg, mindspore.int32)
        new = broadcast_to1(num_neg)
        neg = idx_rank < new  # 指示负样本的mask，里面是T/F

        pos = self.cast(pos, mindspore.bool_)
        neg = self.cast(neg, mindspore.float16)
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos] = 0
        neg[conf_t < 0] = 0  # Filter out neutrals

        pos = self.cast(pos, mindspore.float16)# pos = self.cast(pos, mindspore.float32)
        c_conf_t = self.less(0, (pos + neg)).view(-1)  # 2 19248
        c_conf_t = self.cast(c_conf_t, mindspore.float16)
        all = self.sum_f(c_conf_t)
        cp = conf_data.view(-1, self.num_classes)
        ct = conf_t.view(-1)
        loss_final = self.sum_f(self.closs(cp, ct) * c_conf_t) / all

        return cfg['conf_alpha'] * loss_final

    def _mask_iou(self, mask1, mask2):

        intersection = self.sum_f(mask1 * mask2, (0, 1))   # 138 138 19248
        area1 = self.sum_f(mask1, (0, 1))
        area2 = self.sum_f(mask2, (0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    # iou损失
    def mask_iou_loss(self, maskiou_net_input, maskiou_t, label_t, mask_t):

        maskiou_net_input = self.cast(maskiou_net_input, mindspore.float16)
        perm = (2,0,1)
        mask_t = self.transpose(mask_t,perm)  # b*100, 138 138
        gt_mask_area = self.sum_f(mask_t, (1, 2))
        # select 2 100
        select = gt_mask_area > cfg['discard_mask_area']
        select = self.cast(select, mindspore.float16)
        num = self.sum_f(select)
        # maskiou_p 是N，80对80个类的iou都做预测，下面根据标签取出所属类的iou
        maskiou_p = self.maskiounet(maskiou_net_input)
        label_t = self.expand_dims(label_t, 1)  # b*100，1
        gather = self.gatherD(maskiou_p, 1, label_t)  # b*100，1
        maskiou_p = gather.view(-1)
        loss_i = self.sum_f(self.smoothL1loss(maskiou_p, maskiou_t) * select)
        loss_i /= num

        return loss_i * cfg['maskiou_alpha']

    # mask损失
    def lincomb_mask_loss(self, pos, idx_t, mask_data, proto_data, masks, gt_box_t, labels):

        mask_h = proto_data.shape[1]  # 138
        mask_w = proto_data.shape[2]  # 138
        loss_m = 0

        maskiou_t_list = ()
        maskiou_net_input_list = ()
        label_t_list = ()
        mask_t_list = ()
        for idx in range(mask_data.shape[0]):

            down_mask = self.cast(self.expand_dims(masks[idx], 0), mindspore.float16)
            downsampled_masks = self.resize_bilinear(down_mask, size=(
                self.scalarCast(mask_h, mindspore.int32), self.scalarCast(mask_w, mindspore.int32)),
                                                     align_corners=False)
            downsampled_masks = self.squeeze0(downsampled_masks)
            downsampled_masks = F.stop_gradient(downsampled_masks)

            perm = (1, 2, 0)
            downsampled_masks = self.transpose(downsampled_masks, perm)
            if cfg['mask_proto_binarize_downsampled_gt']:  # True
                downsampled_masks = self.cast(self.less(0.5, downsampled_masks), mindspore.float16)

            pos = self.cast(pos, mindspore.bool_)
            cur_pos = pos[idx]
            # out_idx 是选出的100个T对应的idx，当不够100时，补F对应的idx
            out_idx, out_mask = self.randChoicewithmask(cur_pos)
            out_idx = out_idx.view(-1)
            cur_pos = self.cast(cur_pos, mindspore.float16)
            pos_mask = cur_pos[out_idx]
            num = self.sum_f(pos_mask)
            pos_idx_t = idx_t[idx, out_idx]  # 100个样本对应的gt序号

            proto_masks = self.squeeze0(proto_data[idx:idx+1:1, ::, ::, ::])  # 138 138 32
            proto_coef = self.squeeze0(mask_data[idx:idx+1:1, out_idx, ::])   # 100 32
            pos_gt_box_t = self.squeeze0(gt_box_t[idx:idx+1:1, out_idx, ::])
            # mask_ = self.gather(downsampled_masks, pos_idx_t, 2)              # 138 138 57744
            mask_ = downsampled_masks[:,:,pos_idx_t]                            # 138 138 100

            mask_t_list += (mask_,)
            label_t_ = labels[idx]  # labels是list  label_t_ 是tensor
            label_t = label_t_[pos_idx_t]

            proto_coef = self.squeeze(proto_coef)
            pred_masks = self.matmulnn(proto_masks, proto_coef)
            pred_masks = self.sigmoid(pred_masks)
            if cfg['mask_proto_crop']:  # true
                # 训练的时候，用对应gt框的真是坐标，裁剪mask
                pred_masks = crop(pred_masks, pos_gt_box_t)

            weight_bce = None
            pred_masks = C.clip_by_value(pred_masks, self.min_value,self.max_value)
            # mask_t 138 138 128    pred_masks 138 138 100
            pred_masks = self.cast(pred_masks, mindspore.float16)
            pre_loss = self.binary_cross_entropy_n(pred_masks, mask_, weight_bce)  # 138 138 100
            if cfg['mask_proto_normalize_emulate_roi_pooling']:  # true
                weight = mask_h * mask_w if cfg['mask_proto_crop'] else 1
                pos_gt_csize = center_size(pos_gt_box_t)
                gt_box_width = pos_gt_csize[:, 2] * mask_w
                gt_box_height = pos_gt_csize[:, 3] * mask_h
                pre_loss = self.sum_f(pre_loss, (0, 1))
                pre_loss = pre_loss / gt_box_width / gt_box_height * weight


            pre_loss = pre_loss * pos_mask
            pre_loss /= num
            loss_m += self.sum_f(pre_loss)
            if cfg['use_maskiou']:  # fask rescore branch

                perm = (2, 0, 1)
                maskiou_net_input = self.expand_dims(self.transpose(pred_masks, perm), 1)
                # pred_masks 138 138 19248  --> 19248 1 138 138  作为后面conv2d的输入(batchsize,inchannel,h,w)
                pred_masks = self.cast(self.less(0.5, pred_masks), mindspore.float16)
                maskiou_t = self._mask_iou(pred_masks, mask_)
                maskiou_net_input_list += (maskiou_net_input, )
                maskiou_t_list += (maskiou_t, )
                label_t_list += (label_t, )

        lossm = loss_m * cfg['mask_alpha'] / mask_h / mask_w

        if cfg['use_maskiou']:  # true

            maskiou_t = self.concat(maskiou_t_list)                    # b*19248 ,
            label_t = self.concat(label_t_list)                        # b*19248 ,
            maskiou_net_input = self.concat(maskiou_net_input_list)    # b*19248 , 1, 138,138
            mask_t = self.concat2(mask_t_list)   # 138 138 b*19248
            return lossm, maskiou_net_input, maskiou_t, label_t, mask_t

        return lossm


class mask_iou_net(nn.Cell):
    def __init__(self):
        super(mask_iou_net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(2, 2), pad_mode='pad',
                               has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), pad_mode='pad',
                               has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), pad_mode='pad',
                               has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), pad_mode='pad',
                               has_bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), pad_mode='pad',
                               has_bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=80, kernel_size=(1, 1), stride=(1, 1), pad_mode='pad',
                               has_bias=True)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((3,3), (3,3), 'valid')
        self.squeeze = P.Squeeze(-1)

    def construct(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)  # 19248 80 3 3
        maskiou_p = self.squeeze(self.squeeze(self.max_pool2d(out)))  # 19248 80

        return maskiou_p