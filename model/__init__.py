from model.SFCSR import SFCSR
# from model.Branch import BranchUnit
from model.SGSR import SGSR
from model.THreeDFCNN import ThreeDFCNN
from model.Bicubic import Bicubic
from model.ERCSR import ERCSR
from model.MCNet import MCNet
from model.GDRRN import GDRRN
from model.EDSR import EDSR
from model.RFSR import Net
from model.SSPSR import SSPSR
from model.baseline import Baseline
from model.Interactformer import Interactformer

def Model(opt):
    if opt.method == "SFCSR":
        model = SFCSR(opt)
    elif opt.method == "baseline":
        model = Baseline(opt)
    elif opt.method == "SGSR":
        model = SGSR(opt)
    elif opt.method == "Bicubic":
        model = Bicubic(opt)
    elif opt.method == "3DFCNN":
        model = ThreeDFCNN(opt)
    elif opt.method == "ERCSR":
        model = ERCSR(opt)
    elif opt.method == "MCNet":
        model = MCNet(opt)
    elif opt.method == "GDRRN":
        model = GDRRN(opt)
    elif opt.method == "EDSR":
        model = EDSR(opt)
    elif opt.method == "RFSR":
        model = Net(opt)
    elif opt.method == "SSPSR":
        model = SSPSR(opt)
    elif opt.method == "Interactformer":
        model = Interactformer(opt)
    return model
