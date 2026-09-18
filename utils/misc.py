import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops import pointnet2_utils
import math



def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc

def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config, last_epoch=-1):
    # 如果 YAML 没写 warmingup_e，默认值为 0
    warming_up_t = getattr(config, 'warmingup_e', 0)
    lowest_decay = getattr(config, 'lowest_decay', 0.01)

    # 策略 1：余弦退火衰减 (Cosine)
    if config.get('decay_strategy') == 'cosine':
        max_epoch = getattr(config, 'max_epoch', 300)

        def lr_lbmd(e):
            # 【修改点】：增加 warming_up_t > 0 判断，防除 0
            if warming_up_t > 0 and e < warming_up_t:
                return max(e / warming_up_t, 0.001)
            else:
                progress = (e - warming_up_t) / max(1, (max_epoch - warming_up_t))
                progress = min(1.0, max(0.0, progress))
                decay_factor = lowest_decay + 0.5 * (1.0 - lowest_decay) * (1.0 + math.cos(math.pi * progress))
                return max(decay_factor, lowest_decay)

    # 策略 2：线性匀速衰减 (Linear)
    elif config.get('decay_strategy') == 'linear':
        max_epoch = getattr(config, 'max_epoch', 300)

        def lr_lbmd(e):
            # 【修改点】：增加 warming_up_t > 0 判断，防除 0
            if warming_up_t > 0 and e < warming_up_t:
                return max(e / warming_up_t, 0.001)
            else:
                progress = (e - warming_up_t) / max(1, (max_epoch - warming_up_t))
                progress = min(1.0, max(0.0, progress))
                decay_factor = 1.0 - progress * (1.0 - lowest_decay)
                return max(decay_factor, lowest_decay)

    # 策略 3：原版的指数阶梯衰减
    elif config.get('decay_step') is not None:
        lr_decay = getattr(config, 'lr_decay', 0.9)
        decay_step = getattr(config, 'decay_step', 20)

        def lr_lbmd(e):
            # 【修改点】：增加 warming_up_t > 0 判断，防除 0
            if warming_up_t > 0 and e < warming_up_t:
                return max(e / warming_up_t, 0.001)
            else:
                return max(lr_decay ** ((e - warming_up_t) / decay_step), lowest_decay)

    else:
        raise NotImplementedError("Please specify 'decay_strategy' ('cosine' or 'linear') or 'decay_step' in kwargs.")

    scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    return scheduler


def build_lambda_bnsche(model, config, last_epoch=-1):
    lowest_decay = getattr(config, 'lowest_decay', 0.01)
    bn_momentum = getattr(config, 'bn_momentum', 0.9)

    # 策略 1：余弦衰减 (Cosine) —— 完美随学习率同步平滑下降
    if config.get('decay_strategy') == 'cosine':
        max_epoch = getattr(config, 'max_epoch', 300)

        def bnm_lmbd(e):
            progress = e / max(1, max_epoch)
            progress = min(1.0, max(0.0, progress))
            # 从 bn_momentum (如0.9) 平滑余弦下降到 lowest_decay (如0.01)
            decay_factor = lowest_decay + 0.5 * (bn_momentum - lowest_decay) * (1.0 + math.cos(math.pi * progress))
            return max(decay_factor, lowest_decay)

    # 策略 2：线性匀速衰减 (Linear)
    elif config.get('decay_strategy') == 'linear':
        max_epoch = getattr(config, 'max_epoch', 300)

        def bnm_lmbd(e):
            progress = e / max(1, max_epoch)
            progress = min(1.0, max(0.0, progress))
            # 匀速下坡
            decay_factor = bn_momentum - progress * (bn_momentum - lowest_decay)
            return max(decay_factor, lowest_decay)

    # 策略 3：保留原有的指数阶梯衰减 (兼容老版本)
    elif config.get('decay_step') is not None:
        bn_decay = getattr(config, 'bn_decay', 0.5)
        decay_step = getattr(config, 'decay_step', 20)

        def bnm_lmbd(e):
            return max(bn_momentum * (bn_decay ** (e / decay_step)), lowest_decay)

    else:
        raise NotImplementedError("Please specify 'decay_strategy' or 'decay_step' in bnmscheduler kwargs.")

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler

    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()


        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()


def seprate_point_cloud_train(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    '''
     Vectorized seprate point cloud (GPU Accelerated Version)
    '''
    B, N, C = xyz.shape
    assert N == num_points
    assert C == 3
    if crop == num_points:
        return xyz, None

    # 1. 批量生成虚拟摄像机中心 (Batch Center Generation)
    if fixed_points is None:
        # 生成 B 个不同的随机方向
        centers = F.normalize(torch.randn(B, 1, 3, device=xyz.device), p=2, dim=-1)
    else:
        if isinstance(fixed_points, list):
            # 从固定点列表中随机抽取 (为了兼容原逻辑)
            fixed_list = [random.sample(fixed_points, 1)[0] for _ in range(B)]
            centers = torch.stack(fixed_list).view(B, 1, 3).to(xyz.device)
        else:
            # Test 时固定的一个方向，扩展到整个 Batch
            centers = fixed_points.view(1, 1, 3).expand(B, 1, 3).to(xyz.device)

    # 2. 批量计算距离并排序 (No for-loop!)
    # 利用 PyTorch 广播机制，一次性算完 Batch 内所有点到各自中心的距离
    distance_matrix = torch.norm(xyz - centers, p=2, dim=-1)  # (B, N)
    sort_idx = torch.argsort(distance_matrix, dim=-1, descending=False)  # (B, N)

    # 根据排序结果，把整个 Batch 的点云重新排列
    xyz_sorted = torch.gather(xyz, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)

    # 3. 分支处理：训练期 (随机裁剪+FPS) vs 测试期 (固定裁剪)
    if isinstance(crop, list):
        # 训练/验证：生成整个 Batch 每张点云对应的随机裁剪数
        num_crops = torch.randint(crop[0], crop[1] + 1, (B,), device=xyz.device)

        input_data_padded = xyz_sorted.clone()
        crop_data_padded = xyz_sorted.clone()

        # 虽然有 for 循环，但只涉及索引赋值，不涉及任何 PyTorch 计算图和 CUDA 算子调度，极快！
        for i in range(B):
            nc = num_crops[i].item()
            if padding_zeros:
                input_data_padded[i, :nc] = 0.0
            else:
                # 【神级 Trick】：把要扔掉的点替换为保留部分的第一个有效点
                # 这样 FPS 在采样时因为距离为 0，绝对不会选中这些无效点！
                # 从而完美实现将不同长度的点云打包成相同的形状 (B, N, 3) 进行一次性批处理 FPS
                input_data_padded[i, :nc] = xyz_sorted[i, nc:nc + 1]

            if nc < N:
                crop_data_padded[i, nc:] = xyz_sorted[i, 0:1]

        # 🚀 见证奇迹：一次 CUDA 算子调用处理整个 Batch 的 FPS！(速度提升数十倍)
        input_data = fps(input_data_padded, 2048)
        crop_data = fps(crop_data_padded, 2048)

        return input_data.contiguous(), crop_data.contiguous()

    else:
        # 测试期：裁剪数量是固定的常数
        nc = crop
        if padding_zeros:
            input_data = xyz_sorted.clone()
            input_data[:, :nc, :] = 0.0
        else:
            # 因为 nc 恒定，直接通过切片就能取出整个 Batch 的剩余部分
            input_data = xyz_sorted[:, nc:, :]

        crop_data = xyz_sorted[:, :nc, :]

        return input_data.contiguous(), crop_data.contiguous()







def get_ptcloud_img(ptcloud, count_num):

    # 创建透明背景的图形
    fig = plt.figure(figsize=(8, 8), facecolor=(0, 0, 0, 0))  # 设置背景为完全透明

    # 点云数据分离
    x, z, y = ptcloud.transpose(1, 0)

    ax = fig.add_subplot(projection='3d', adjustable='box')
    ax.set_facecolor((0, 0, 0, 0))  # 设置绘图区域背景为完全透明

    # 隐藏坐标轴
    ax.axis('off')
    ax.view_init(30, 45)

    # 设置点云显示范围
    max_val, min_val = np.max(ptcloud), np.min(ptcloud)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)

    # 生成渐变色（基于 y 值）
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))  # 归一化 y 值到 [0, 1]

    # plt.cm.Blues：蓝色渐变。
    # plt.cm.Greens：绿色渐变。
    # plt.cm.Reds：红色渐变。
    # plt.cm.Oranges：橙色渐变。
    # plt.cm.Purples：紫色渐变。
    # plt.cm.Greys：灰色渐变。
    # plt.cm.plasma：等离子色渐变。
    # plt.cm.viridis：彩虹色渐变（默认）。
    # plt.cm.inferno：火焰色渐变。
    # plt.cm.magma：岩浆色渐变。

###################################################################################################
    # # # ShapeNet-55
    # if (count_num + 76 - 1) // 76 == 1:
    #     colors = plt.cm.Blues(y_normalized)  # 蓝色渐变
    # elif (count_num + 76 - 1) // 76 == 2:
    #     colors = plt.cm.Greens(y_normalized)  # 绿色渐变
    # elif (count_num + 76 - 1) // 76 == 3:
    #     colors = plt.cm.Reds(y_normalized)  # 红色渐变
    # elif (count_num + 76 - 1) // 76 == 4:
    #     colors = plt.cm.Oranges(y_normalized)  # 橙色渐变
    # elif (count_num + 76 - 1) // 76 == 5:
    #     colors = plt.cm.Purples(y_normalized)  # 紫色渐变
    # elif (count_num + 76 - 1) // 76 == 6:
    #     colors = plt.cm.Greys(y_normalized)  # 灰色渐变
    # elif (count_num + 76 - 1) // 76 == 7:
    #     colors = plt.cm.plasma(y_normalized)  # 等离子色渐变

###################################################################################################
    # # PCN
    # if (count_num + 10 - 1) // 10 == 1:
    #     colors = plt.cm.Blues(y_normalized)  # 蓝色渐变
    # elif (count_num + 10 - 1) // 10 == 2:
    #     colors = plt.cm.Greens(y_normalized)  # 绿色渐变
    # elif (count_num + 10 - 1) // 10 == 3:
    #     colors = plt.cm.Reds(y_normalized)  # 红色渐变
    # elif (count_num + 10 - 1) // 10 == 4:
    #     colors = plt.cm.Oranges(y_normalized)  # 橙色渐变
    # elif (count_num + 10 - 1) // 10 == 5:
    #     colors = plt.cm.Purples(y_normalized)  # 紫色渐变
    # elif (count_num + 10 - 1) // 10 == 6:
    #     colors = plt.cm.Greys(y_normalized)  # 灰色渐变

###################################################################################################
    # KITTI
    if (count_num + 20 - 1) // 20 == 1:
        colors = plt.cm.Blues(y_normalized)  # 蓝色渐变
    elif (count_num + 20 - 1) // 20 == 2:
        colors = plt.cm.Greens(y_normalized)  # 绿色渐变
    elif (count_num + 20 - 1) // 20 == 3:
        colors = plt.cm.Reds(y_normalized)  # 红色渐变
    elif (count_num + 20 - 1) // 20 == 4:
        colors = plt.cm.Oranges(y_normalized)  # 橙色渐变
    elif (count_num + 20 - 1) // 20 == 5:
        colors = plt.cm.Purples(y_normalized)  # 紫色渐变
    elif (count_num + 20 - 1) // 20 >= 6:
        colors = plt.cm.Greys(y_normalized)  # 灰色渐变

###################################################################################################

    # 绘制点云
    ax.scatter(
        x, y, z,
        zdir='z',
        c=colors,   # '#00bfff',
        marker='o',
        s=20,
        alpha=1,  # 点的透明度
        edgecolor='black',  # 去掉边缘线
        linewidths=0.5
    )

    # 绘图并提取 RGBA 数据
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # 适应 RGBA 形状

    # 关闭图形
    plt.close(fig)

    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y',
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    # pic_path = path + '.png'
    # fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, gt, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale, gt * scale



from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
