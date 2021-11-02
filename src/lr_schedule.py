"""lr generator for yolact"""

def dynamic_lr(cfg, start_epoch = 0, total_epochs = 0, steps_each_epoch = 0):

    """dynamic learning rate generator"""
    start_steps = start_epoch * steps_each_epoch
    total_steps = (total_epochs - start_epoch) * steps_each_epoch
    lr_init = cfg['lr']
    gamma = cfg['gamma']

    # if args.autoscale and args.batch_size != 8:
    #     factor = args.batch_size / 8
    #     if __name__ == '__main__':
    #         print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))
    #
    #     cfg.lr *= factor
    #     cfg.max_iter //= factor   改epoch
    #     cfg.lr_steps = [x // factor for x in cfg.lr_steps]

    lr_steps = (280000, 600000, 700000, 750000)   # 单卡
    # lr_steps = (70000, 150000, 175000, 187500)
    lr_stages = [lr_init * (gamma ** 0),
                 lr_init * (gamma ** 1),
                 lr_init * (gamma ** 2),
                 lr_init * (gamma ** 3),
                 lr_init * (gamma ** 4)
                 ]

    lr = []
    lr_warmup_until = cfg['lr_warmup_until']
    lr_warmup_init = cfg['lr_warmup_init']

    step_index = 0   # 指示第几次更换学习率
    # <=500 change
    # >500 <=280k lr_init
    # >280k <=600k lr_init * (gamma ** 1)
    # >600k <=700k lr_init * (gamma ** 2)
    # >700k <=750k lr_init * (gamma ** 3)
    # >750k        lr_init * (gamma ** 4)
    for cur_step in range(total_steps):
        # 如果cur_step到达另一个阶段，则改变学习率
        if lr_warmup_until > 0 and cur_step <= lr_warmup_until:
            lr.append((lr_init - lr_warmup_init) * (cur_step / lr_warmup_until) + lr_warmup_init)
        else:
            lr.append(lr_stages[step_index])
        # 750k 之后用 step_index 限制不进入下面条件 即可一直使用lr_init * (gamma ** 4)
        while step_index < len(lr_steps) and cur_step >= lr_steps[step_index]:
            step_index += 1

    learning_rate = lr[start_steps:]
    return learning_rate

# def dynamic_lr(config, rank_size=1, start_steps=0):
#     """dynamic learning rate generator"""
#
#     base_lr = cfg['lr']
#     # 需要在cfg中设置一个start_iter
#     iteration = max(config.start_iter, 0)
#     total_steps = config.max_iter
#     lr = []
#     step_index = 0  # 指示第几次更换学习率，论文中迭代280k，600k，700k，750k 时更换
#     for i in range(total_steps):
#         if cfg['lr_warmup_until'] > 0 and iteration <= cfg['lr_warmup_until']:
#             lr.append((base_lr - cfg['lr_warmup_init']) * (iteration / cfg['lr_warmup_until']) + cfg['lr_warmup_init'])
#         else:
#             lr.append(base_lr)
#         # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
#         while step_index < len(cfg['lr_steps']) and iteration >= cfg['lr_steps'][step_index]:
#             step_index += 1
#             lr.append(cfg['lr'] * (cfg['gamma'] ** step_index))
#     learning_rate = lr[start_steps:]
#     return learning_rate