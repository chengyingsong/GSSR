import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from option import opt
from data_utils import ValsetFromFolder, is_image_file, shuffle
import scipy.io as scio
from eval import PSNR, SSIM, SAM
from model import Model
import time

input_path = (
    "/data2/cys/data/"
    + opt.datasetName
    + "/process_test/"
    + str(opt.upscale_factor)
    + "/"
)

out_path = ("result/"
            + opt.datasetName+"/"
            + str(opt.upscale_factor)
            + "/"
            # + opt.method
            + "Interactformer"
            + "/"
            )

val_set = ValsetFromFolder(
    input_path, opt.shuffle, opt.shuffleMode, opt.window_size)
val_loader = DataLoader(
    dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False
)

if not os.path.exists(out_path):
    os.makedirs(out_path)

PSNRs = []
SSIMs = []
SAMs = []

if opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception(
            "No GPU found or Wrong gpu id, please run without --cuda")

model = Model(opt)

if opt.cuda and opt.dist:
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda()

if opt.model_name:
    checkpoint = torch.load(opt.model_name)
    Branch_dict = checkpoint["model"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,
                       v in Branch_dict.items() if k in model_dict}
    miss_param = {k for k in Branch_dict.keys() if k not in model_dict}
    if len(miss_param) != 0:
        print("miss_param:", len(miss_param))
        print(miss_param)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(checkpoint)
        model.eval()

images_name = [x for x in listdir(input_path) if is_image_file(x)]
T = 0
for index, batch in enumerate(val_loader):
    with torch.no_grad():
        input, HR = Variable(batch[0]), Variable(batch[1])
        SR = np.zeros((HR.shape[1], HR.shape[2],
                       HR.shape[3])).astype(np.float32)

        HR = HR.data[0].numpy()
        B, C, h, w = input.shape
        g = (C + opt.window_size - 1) // opt.window_size

        if opt.cuda:
            input = input.cuda()

        start_time = time.time()

        if opt.method == "SGSR":
            h1 = []
            channel_count = torch.zeros(C)
            for i in range(g):
                start = i * opt.window_size
                end = (i + 1) * opt.window_size
                if end > C:
                    end = C
                    start = end - opt.window_size
                x = input[:, start:end, :, :]
                y, h1 = model(x, h1, i)

                SR[start:end, :, :] += y.cpu().data[0].numpy()
                channel_count[start:end] += 1
            SR = SR / channel_count.reshape(-1, 1, 1).numpy()

            end_time = time.time()
            T = T + (end_time - start_time)
            SR[SR < 0] = 0
            SR[SR > 1.0] = 1.0

            psnr = PSNR(SR, HR)
            ssim = SSIM(SR, HR)
            sam = SAM(SR, HR)

            PSNRs.append(psnr)
            SSIMs.append(ssim)
            SAMs.append(sam)

            if opt.method == "SGSR":
                SR = shuffle(torch.from_numpy(SR), opt.band //
                             opt.window_size).numpy()
                HR = shuffle(torch.from_numpy(HR), opt.band //
                             opt.window_size).numpy()

            SR = SR.transpose(1, 2, 0)
            HR = HR.transpose(1, 2, 0)
            scio.savemat(out_path + images_name[index], {'HR': HR, 'SR': SR})

    print("=====FPS:{:.3f}=====T:{:.3f}".format(
        len(images_name)/T*opt.band, T))
    print(
        "=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(
            np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs)
        )
    )
    print(
        "=====stdPSNR:{:.3f}=====stdSSIM:{:.4f}=====stdSAM:{:.3f}".format(
            np.std(PSNRs), np.std(SSIMs), np.std(SAMs)
        )
    )
    print(T / len(images_name))
