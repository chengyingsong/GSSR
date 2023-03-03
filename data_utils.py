import sys
from tkinter import Variable
import torch
import numpy as np
import torch.utils.data as data

from os import listdir
from os.path import join
import scipy.io as scio


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def shuffle(x, g=3):
    """Band shuffle"""
    C, H, W = x.size()
    # g = 3  # group number
    # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
    if C // g * g < C:
        # 不能整除，最后几个band不shuffle
        split_x = x[: C // g * g, :, :]
        x[: C // g * g, :, :] = (
            split_x.view(g, int(C / g), H, W)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(C // g * g, H, W)
        )
    else:
        x = x.view(g, int(C / g), H, W).permute(1, 0,
                                                2, 3).contiguous().view(C, H, W)
    return x


def choose_x(input, i):
    """
    选择和本帧差异性最大的三帧
    """
    Num = input.shape[1]
    if i == 0:
        x = input[:, 0:3, :, :]
    elif i == Num - 1:
        x = input[:, i-2:i+1, :, :]
    else:
        x = input[:, i-1:i+2, :, :]
    return x


class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, shuffle, shufflemode="origin", interval=3):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        self.shuffle = shuffle
        self.interval = interval
        self.shufflemode = shufflemode

    def __getitem__(self, index):
        mat = scio.loadmat(
            self.image_filenames[index], verify_compressed_data_integrity=False
        )
        # mat = h5py.File(self.image_filenames[index], "r")
        input = mat["lr"].astype(np.float32)
        label = mat["hr"].astype(np.float32)

        # print(input.shape, label.shape)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        # input shape:C,H,W
        # assert(not torch.equal(input, shuffle(input)))
        if self.shuffle and "random" not in self.shufflemode:
            # 正常shuffle
            input = shuffle(input, self.interval)
            label = shuffle(label, self.interval)
        elif self.shuffle and self.shufflemode == "random":
            # random shuffle
            idx = torch.randperm(input.shape[0])
            input = input[idx, :, :].view(input.size())
            label = label[idx, :, :].view(label.size())
        return input, label

    def __len__(self):
        return len(self.image_filenames)


class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, shuffle, shufflemode="origin", interval=3):
        super(ValsetFromFolder, self).__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        self.shuffle = shuffle
        self.interval = interval
        self.shufflemode = shufflemode

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        # mat = h5py.File(self.image_filenames[index], "r")
        input = mat["LR"].astype(np.float32).transpose(2, 0, 1)
        label = mat["HR"].astype(np.float32).transpose(2, 0, 1)
        # print(input.shape)
        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label).float()
        if self.shuffle and "random" not in self.shufflemode:
            # print("shuffle")
            input = shuffle(input, self.interval)
            label = shuffle(label, self.interval)
        elif self.shuffle and self.shufflemode == "random":
            idx = torch.randperm(input.shape[0])
            input = input[idx, :, :].view(input.size())
            label = label[idx, :, :].view(label.size())
        return input, label

    def __len__(self):
        return len(self.image_filenames)


def chop_forward(x, model, scale, shave=16):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]
    outputlist = []
    for i in range(4):
        input_batch = inputlist[i]
        output_batch = model(input_batch)
        # print("patch shape:",output_batch.shape)
        outputlist.append(output_batch)

    output = np.zeros((c, h*scale, w*scale)).astype(np.float32)
    # print("output shape: ",output.shape)
    output[:, 0:h_half*scale, 0:w_half*scale] = outputlist[0][0,
                                                              :, 0:h_half*scale, 0:w_half*scale].cpu().numpy()
    output[:, 0:h_half*scale, w_half*scale:w*scale] = outputlist[1][0, :,
                                                                    0:h_half*scale, (w_size - w + w_half)*scale:w_size*scale].cpu().numpy()
    output[:, h_half*scale:h*scale, 0:w_half*scale] = outputlist[2][0, :,
                                                                    (h_size - h + h_half)*scale:h_size*scale, 0:w_half*scale].cpu().numpy()
    output[:, h_half*scale:h*scale, w_half*scale:w*scale] = outputlist[3][0, :,
                                                                          (h_size - h + h_half)*scale:h_size*scale, (w_size - w + w_half)*scale:w_size*scale].cpu().numpy()

    return output
