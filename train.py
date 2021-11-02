import os
import argparse
import ast
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import SGD
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore.communication.management import get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, SummaryCollector
# from layers.modules.loss import MultiBoxLoss
from src.yolact.layers.modules.loss_614 import MultiBoxLoss
# from src.yolact.yolactpp import Yolact
from src.yolact.yolactpp import Yolact
from src.config import yolact_plus_resnet50_config as cfg
from src.dataset import data_to_mindrecord_byte_image, create_yolact_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import WithLossCell
from mindspore.common import initializer as init_p

set_seed(1)

parser = argparse.ArgumentParser(description="MaskRcnn training")
parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False, help="If set it true, only create "
                    "Mindrecord, default is false.")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="Do train or not, default is true.")
parser.add_argument("--do_eval", type=ast.literal_eval, default=False, help="Do eval or not, default is false.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--pre_trained", type=str, default="weights/ms_resnet.ckpt", help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=3, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
args_opt = parser.parse_args()


# import os
# device_id = int(os.getenv("DEVICE_ID", args_opt))
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id,max_call_depth=10000)#,save_graphs=True)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id,max_call_depth=10000)

def init_weights(module):

    for name, cell in module.cells_and_names():
        is_conv_layer = isinstance(cell, nn.Conv2d)

        if is_conv_layer and "backbone" not in name:
            cell.weight.set_data(init_p.initializer('XavierUniform', cell.weight.shape))

            if cell.has_bias is True:
                cell.bias.set_data(init_p.initializer('zeros', cell.bias.shape))

if __name__ == '__main__':

    print("Start train for yolact!")
    if not args_opt.do_eval and args_opt.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    prefix = "yolact.mindrecord"
    mindrecord_dir = cfg['mindrecord_dir']
    # mindrecord_file = os.path.join(mindrecord_dir, prefix)
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(cfg['coco_root']):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("coco_root not exits.")
        else:
            if os.path.isdir(cfg['IMAGE_DIR']) and os.path.exists(cfg['ANNO_PATH']):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("IMAGE_DIR or ANNO_PATH not exits.")

    if not args_opt.only_create_dataset:

        dataset = create_yolact_dataset(mindrecord_file, batch_size=cfg['batch_size'],
                                            device_num=device_num, rank_id=rank)
        num_steps = dataset.get_dataset_size()
        print("pre epoch step num: ", num_steps)
        print("Create dataset done!")
        net = Yolact()
        net = net.set_train()

        # 加载backbone预训练参数
        backbone_path = args_opt.pre_trained
        if backbone_path != "":
            param_dict = load_checkpoint(backbone_path)
            if cfg['pretrain_epoch_size'] == 0:
                for item in list(param_dict.keys()):
                    if not (item.startswith('backbone')):
                        param_dict.pop(item)
            load_param_into_net(net, param_dict)

        # 初始化the rest of net参数
        init_weights(net)

        loss = MultiBoxLoss(num_classes=cfg['num_classes'], pos_threshold=cfg['positive_iou_threshold'], neg_threshold=cfg['negative_iou_threshold'], negpos_ratio=cfg['ohem_negpos_ratio'],batch_size=cfg['batch_size'],num_priors=cfg['num_priors'])
        net_with_loss = WithLossCell(net, loss)

        lr = Tensor(dynamic_lr(cfg, start_epoch=0, total_epochs=cfg['epoch_size'], steps_each_epoch=num_steps),
                    mstype.float32)

        opt = SGD(params=net.trainable_params(), learning_rate=cfg['lr'], momentum=cfg['momentum'],
              weight_decay=cfg['decay'])

        # define model
        model = Model(net_with_loss, optimizer=opt, amp_level='O3')

        print("============== Starting Training ==============")
        time_cb = TimeMonitor(data_size=num_steps)
        loss_cb = LossMonitor()

        summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
        cb = [time_cb, loss_cb]
        if cfg['save_checkpoint']:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_epochs'] * num_steps,
                                          keep_checkpoint_max=cfg['keep_checkpoint_max'])
            save_checkpoint_path = os.path.join(cfg['save_checkpoint_path'], 'ckpt_' + str(rank) + '/')
            ckpoint_cb = ModelCheckpoint(prefix='yolact', directory=save_checkpoint_path, config=ckptconfig)
            cb += [ckpoint_cb]


        model.train(cfg['epoch_size'], dataset, callbacks=cb, dataset_sink_mode=False)
        # profiler.analyse()
        print("============== End Training ==============")
