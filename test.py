import mindspore.ops as P
import os
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.dataset import context


class se(nn.Cell):
    def __init__(self): # ,batch_size,num_priors):
        super(se, self).__init__()
        self.cast = P.Cast()
        self.squeeze = P.Squeeze()

    def construct(self, crowd_boxes,cfg):

       # m1 = (crowd_boxes[0, 0] > 0) or (crowd_boxes[0, 1] > 0)
       # m2 = (crowd_boxes[0, 2] > 0) or (crowd_boxes[0, 3] > 0)
       # m = m1 or m2
       # tmp = cfg['crowd_iou_threshold'] < 1
       # tmp_1 = self.cast(m,mindspore.float16)
       # tmp_2 = self.cast(tmp,mindspore.float16)
       # if tmp_1 * tmp_2 == 1:
       #     print("12345")
       # else:
       #     print("23456")
       #
       # return 0


       m1 = (crowd_boxes[0, 0] > 0) or (crowd_boxes[0, 1] > 0)
       m2 = (crowd_boxes[0, 2] > 0) or (crowd_boxes[0, 3] > 0)
       m = m1 or m2
       if m and cfg<1 :
           print("12345:",m)

       return m



if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', default=2))
    # GRAPH_MODE PYNATIVE_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id,
                        save_graphs=True)
    # crowd_boxes = Tensor(np.random.rand(10, 4).astype(np.float16))
    crowd_boxes = Tensor(np.array([[0.4,0.5,1,0.8]]).astype(np.float16))

    s = se()
    cfg = 0.7
    s(crowd_boxes,cfg)