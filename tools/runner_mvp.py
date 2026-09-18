import torch
import torch.nn as nn
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.AverageMeter_mvp import AverageMeter_test
from utils.metrics_mvp import Metrics
from utils.metrics_mvp import Metrics_val
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_net_mvp(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # print model info
    print_log('Trainable_parameters:', logger=logger)
    print_log('=' * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger=logger)

    print_log('Untrainable_parameters:', logger=logger)
    print_log('=' * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger=logger)


    print_log('Using Data parallel ...', logger=logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch - 1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):

            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == 'MVP':
                partial = data[0].cuda()
                gt = data[1].cuda()

            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            ret = base_model(partial)

            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)

            _loss = sparse_loss + dense_loss
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10),
                                               norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()


        if isinstance(scheduler, list):
            for item in scheduler:
                # 如果是 timm 的 CosineLRScheduler，需要传 epoch
                if type(item).__name__ == 'CosineLRScheduler':
                    item.step(epoch)
                else:
                    item.step()
        else:
            # 处理非 list 的情况
            if type(scheduler).__name__ == 'CosineLRScheduler':
                scheduler.step(epoch)
            else:
                scheduler.step()


        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger=logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config,
                               logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args, logger=logger)
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # [MODIFIED] 这里 len(test_dataloader) 实际上是批次数 (num_batches)，不是样本数
    num_batches = len(test_dataloader)
    interval = max(num_batches // 10, 1)  # 防止 dataloader 较短时 interval 为 0

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):


            partial = data[0]
            gt = data[1]
            current_bs = partial.size(0)

            # 2. 模型前向计算
            ret = base_model(partial)
            dense_points = ret[-1]


            # 4. 计算整体/Batch指标
            _metrics = Metrics_val.get(dense_points, gt, require_emd=False)

            # 确保 _metrics 里的东西是标量数值
            if isinstance(_metrics[0], torch.Tensor):
                _metrics_vals = [m.item() for m in _metrics]
            else:
                _metrics_vals = _metrics


            if 'Overall' not in category_metrics:
                category_metrics['Overall'] = AverageMeter(Metrics.names())
            category_metrics['Overall'].update(_metrics_vals, n=current_bs)

            if (idx + 1) % interval == 0:
                print_log('Test Batch[%d/%d] BatchSize = %d  Metrics = %s' %
                          (idx + 1, num_batches, current_bs, ['%.4f' % m for m in _metrics]), logger=logger)


        for d, v in category_metrics.items():
            test_metrics.update(v.avg())

        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
                  logger=logger)

    return Metrics_val(config.consider_metric, test_metrics.avg())




def test_net_mvp(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader)


def test(base_model, test_dataloader):

    base_model.eval()  # set model to eval mode
    metrics = ['F-Score', 'CDL1', 'CDL2', 'EMDistance']
    test_loss_meters = {m: AverageMeter_test() for m in metrics}
    test_loss_cat = torch.zeros([16, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150 * 26
    novel_cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 50 * 26
    cat_num = torch.cat((cat_num, novel_cat_num), dim=0)
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft',
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    dataset_length = len(test_dataloader)

    with torch.no_grad():
        for i, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):

            label = model_ids
            inputs = data[0].cuda()
            gt = data[1].cuda()

            ret = base_model(inputs)
            dense_points = ret[-1]

            result_dict = Metrics.get(dense_points, gt, require_emd=False)

            batch_size = inputs.size(0)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item(), n=batch_size)

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)].item()


            step_interval_to_print = 5
            if i % step_interval_to_print == 0:
                print_log('test [%d/%d]' % (i, dataset_length))


        print_log('Loss per category:')
        category_log = ''
        for i in range(16):
            category_log += '\ncategory name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i])
        print_log(category_log)


        print_log('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        print_log(overview_log)