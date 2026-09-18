import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
from .build import DATASETS
import transforms3d
import random
import math


@DATASETS.register_module()
class MVP(data.Dataset):
    def __init__(self, MVPyaml, npoints=2048, novel_input=True, novel_input_only=False):
        if MVPyaml.Train:
            self.input_path = '../../PointCloudCompleteData/MVP/mvp_train_input.h5'
            self.gt_path = '../../PointCloudCompleteData/MVP/mvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = '../../PointCloudCompleteData/MVP/mvp_test_input.h5'
            self.gt_path = '../../PointCloudCompleteData/MVP/mvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.mvp_train = MVPyaml.Train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]


        self.pc_augm_scale = 1.5
        self.pc_augm_rot = 1  # 1
        self.pc_augm_mirror_prob = 0.5  # 0.5
        self.pc_augm_jitter = 0


    def augment_cloud(self, Ps):
        """" Augmentation on XYZ and jittering of everything """
        M = transforms3d.zooms.zfdir2mat(1)
        if self.pc_augm_scale > 1:
            #            print('scale')
            s = random.uniform(1 / self.pc_augm_scale, self.pc_augm_scale)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        if self.pc_augm_rot:
            #            print('rot')
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([1, 0, 0], angle), M)  # y=upright assumption
            if random.random() < self.pc_augm_rot / 2:
                angle = random.uniform(0, 2 * math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # y=upright assumption
        if self.pc_augm_mirror_prob > 0:  # mirroring x&z, not y
            #            print('mirror')
            if random.random() < self.pc_augm_mirror_prob / 2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
            if random.random() < self.pc_augm_mirror_prob / 2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
        result = []
        for P in Ps:
            P[:, :3] = np.dot(P[:, :3], M.T)

            if self.pc_augm_jitter:
                sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
                P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
            result.append(P)
        return result[0], result[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 2. 在 NumPy 格式下进行数据增强（此时内部的 np.dot 将完美兼容运行）
        if self.mvp_train:
            partial = self.input_data[index].copy()
            complete = self.gt_data[index // 26].copy()
            partial, complete = self.augment_cloud([partial, complete])
        else:
            partial = self.input_data[index]
            complete = self.gt_data[index // 26]

        # 3. 数据增强彻底完成后，再统一转换为 PyTorch Tensor
        partial = torch.from_numpy(partial)
        complete = torch.from_numpy(complete)

        label = (self.labels[index])
        return label, label, (partial, complete)