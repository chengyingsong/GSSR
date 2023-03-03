import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import Model
from loss import Loss

from option import opt
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR, SAM, SSIM
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import time

psnr = []
out_path = "result/" + opt.datasetName + "/"


def main():
    print(opt)
    best_psnr = 0
    best_sam = 1e6

    if opt.show:
        global writer
        writer = SummaryWriter(
            log_dir="logs/"
            + opt.datasetName
            + "/"
            + str(opt.upscale_factor)
            + "/"
            + opt.exgroup
            + "/"
            + opt.ex
        )

    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception(
                "No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    g = (opt.band + opt.window_size - 1) // opt.window_size

    # Loading datasets
    train_set = TrainsetFromFolder(
        "/data2/cys/data/"
        + opt.datasetName
        + "/process_train/"
        + str(opt.upscale_factor)
        + "/",
        opt.shuffle,
        opt.shuffleMode,
        opt.window_size,
    )
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
    )
    val_set = ValsetFromFolder(
        "/data2/cys/data/"
        + opt.datasetName
        + "/process_test/"
        + str(opt.upscale_factor)
        + "/",
        opt.shuffle,
        opt.shuffleMode,
        opt.window_size,
    )
    val_loader = DataLoader(
        dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False
    )

    # Buliding model
    model = Model(opt)
    # print(model)

    # choose Loss
    criterion = Loss(opt)

    if opt.cuda and opt.dist:
        model = nn.DataParallel(model).cuda()
    elif opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(0.9, 0.999), eps=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            Branch_dict = checkpoint["model"]
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k,
                               v in Branch_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setting learning rate
    scheduler = MultiStepLR(
        optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch=-1
    )

    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(
            epoch, optimizer.param_groups[0]["lr"]))
        start = time.time()
        train(train_loader, optimizer, model, criterion, epoch)
        end = time.time()
        print("epoch Cost:", (end - start) / 60, "min")
        scheduler.step()
        best_psnr, best_sam = val(
            val_loader, model, epoch, optimizer, best_psnr, best_sam
        )


def train(train_loader, optimizer, model, criterion, epoch):

    model.train()
    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(
            batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        B, N, h, w = input.shape

        # CutMix
        # if np.random.rand(1) < 0.5:
        #     input,label = CutMix(input,label)

        h1 = []
        c = opt.window_size
        g = (N + c - 1) // c
        for i in range(g):
            start = i * c
            end = (i + 1) * c
            if end > input.shape[1]:
                end = input.shape[1]
                start = end - c
            x = input[:, start:end, :, :]
            new_label = label[:, start:end, :, :]
            SR, h1 = model(x, h1, i)
            h1 = Variable(h1.detach().data, requires_grad=False)
            loss = criterion.loss(SR, new_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iteration % 100 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(train_loader), loss.item()
                )
            )

        if opt.show:
            niter = epoch * len(train_loader) + iteration
            if niter % 500 == 0:
                writer.add_scalar("Train/Loss", loss.item(), niter)


def val(val_loader, model, epoch, optimizer, best_psnr, best_sam):

    model.eval()
    val_psnr = 0
    val_sam = 0
    val_SSIM = 0

    for iteration, batch in enumerate(val_loader, 1):
        with torch.no_grad():
            input, label = Variable(batch[0]), Variable(batch[1])

            if opt.cuda:
                input = input.cuda()

            B, C, h, w = input.shape
            g = (C + opt.window_size - 1) // opt.window_size
            SR = np.zeros((label.shape[1], label.shape[2], label.shape[3])).astype(
                np.float32
            )

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
            SR[SR < 0] = 0
            SR[SR > 1.0] = 1.0
            val_psnr += PSNR(SR, label.data[0].numpy())
            val_sam += SAM(SR, label.data[0].numpy())
            val_SSIM += SSIM(SR, label.data[0].numpy())
    val_psnr = val_psnr / len(val_loader)
    val_sam = val_sam / len(val_loader)
    val_SSIM = val_SSIM / len(val_loader)
    if val_psnr > best_psnr:
        save_model(model, epoch, optimizer, "psnr_best")
        best_psnr = val_psnr
    if val_sam < best_sam:
        save_model(model, epoch, optimizer, "sam_best")
        best_sam = val_sam

    print(
        "PSNR = {:.3f},best_PSNR = {:.3f},SSIM = {:.4f},SAM = {:.3f},best_sam={:.3f}".format(
            val_psnr, best_psnr, val_SSIM, val_sam, best_sam
        )
    )

    if opt.show:
        writer.add_scalar("Val/PSNR", val_psnr, epoch)
        writer.add_scalar("Val/SAM", val_sam, epoch)

    return best_psnr, best_sam


def save_model(model, epoch, optimizer, name):
    model_out_dir = (
        "checkpoint/"
        + opt.datasetName
        + "/"
        + str(opt.upscale_factor)
        + "/"
        + opt.ex
        + "/"
    )
    model_out_path = model_out_dir + name + ".pth"
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
