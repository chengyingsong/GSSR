import torch.nn as nn


class Loss():
    def __init__(self, opt):
        loss_list = opt.loss
        self.criterions = {}
        for loss_name in loss_list:
            if loss_name == "L1":
                self.criterions["L1"] = nn.L1Loss()
        self.Num = len(loss_list)

        if opt.cuda:
            for key, value in self.criterions.items():
                self.criterions[key] = self.criterions[key].cuda()

    def loss(self, SR, GT):
        """
        多 Loss 集成
        """
        losses = []
        for _, value in self.criterions.items():
            # if key != "SAM" or epoch >= self.SAMepoch:
            loss_i = value(SR, GT)
            # print(loss_i)
            losses.append(loss_i)
        final_Loss = losses[0]
        for i in range(1, self.Num):
            final_Loss += losses[i]
        return final_Loss
