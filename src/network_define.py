"""yolact training network wrapper."""

import mindspore.nn as nn

time_stamp_init = False
time_stamp_first = 0

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn

    def construct(self, x, gt_bboxes, gt_label, crowd_boxes, gt_mask, num_crowd):

        # prediction dict: six tensor
        prediction = self._net(x, gt_bboxes, gt_label, crowd_boxes, gt_mask, num_crowd)
        loss_B, loss_C, loss_M, loss_S, loss_I= self._loss(prediction, gt_bboxes, gt_label, crowd_boxes, gt_mask, num_crowd)
        loss = loss_B + loss_C + loss_M + loss_S + loss_I

        return loss


    @property
    def network(self):
        """
        Get the network.

        Returns:
            Cell, return backbone network.
        """
        return self._net