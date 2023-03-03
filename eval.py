import math
import cv2
import numpy as np
from scipy.signal import convolve2d
import torch


def PSNR(pred, gt):
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))

    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr


def SSIM(pred, gt):
    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[i, :, :], gt[i, :, :])
    return ssim / gt.shape[0]


def SAM(pred, gt):
    # Shape  N,H,W
    eps = 2.2204e-16
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    nom = sum(pred*gt)
    denom1 = sum(pred*pred)**0.5
    denom2 = sum(gt*gt)**0.5
    sam = np.real(np.arccos(nom.astype(np.float32)/(denom1*denom2+eps)))
    sam[np.isnan(sam)] = 0
    sam_sum = np.mean(sam)*180/np.pi

    return sam_sum


def cal_sam(Itrue, Ifake):
    if len(Itrue.shape) == 3:
        Itrue = Itrue.unsqueeze(0)
        Ifake = Ifake.unsqueeze(0)
    # print(Itrue.shape)  B,N,H,W
    esp = 2.2204e-16
    InnerPro = torch.sum(Itrue*Ifake, 1, keepdim=True)
    len1 = torch.norm(Itrue, p=2, dim=1, keepdim=True)
    len2 = torch.norm(Ifake, p=2, dim=1, keepdim=True)
    divisor = len1*len2
    mask = torch.eq(divisor, 0)
    divisor = divisor + (mask.float())*esp
    cosA = torch.sum(InnerPro/divisor, 1).clamp(-1+esp, 1-esp)
    sam = torch.acos(cosA)
    sam = torch.mean(sam)*180 / np.pi
    return sam


def matlab_style_gauss2D(shape=np.array([11, 11]), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape-np.array([1, 1]))/2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1]+1, 1)
    y = np.arange(-siz[0], siz[1]+1, 1)
    m, n = np.meshgrid(x, y)

    h = np.exp(-(m*m + n*n).astype(np.float32) / (2.*sigma*sigma))
    h[h < eps*h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(
        shape=np.array([win_size, win_size]), sigma=1.5)
    window = window.astype(np.float32)/np.sum(np.sum(window))

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)).astype(np.float32) / \
        ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def constrast(img):
    # 传入灰度图
    m,n = img.shape
    b = np.sum(np.power(img[:,1:] - img[:,:n-1],2)) + np.sum(np.power(img[1:,:]-img[:m-1,:],2))
    return b/((m-1)*n+(n-1)*m)

def Bconstrast(img):
    # img: B,N,H,W -- > B,N,1
    B,N,H,W = img.size()
    # print(img.shape)
    b = torch.sum((img[:,:,:,1:]-img[:,:,:,:W-1])*(img[:,:,:,1:]-img[:,:,:,:W-1]),dim=(2,3),keepdim=True) \
        + torch.sum((img[:,:,1:,:]-img[:,:,:H-1,:])*(img[:,:,1:,:]-img[:,:,:H-1,:]),dim=(2,3),keepdim=True)
    b = b / ((H-1)*W + (W-1)*H)
    # print(b.shape)
    return b








    # img_ext = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)/1.0
    # rows_ext,cols_ext = img_ext.shape
    # b = 0.0
    # for i in range(1,rows_ext- 1):
    #     for j in range(1,cols_ext-1):
    #         b += ((img_ext[i,j]-img_ext[i,j+1])**2 + (img_ext[i,j]-img_ext[i,j-1])**2+
    #         (img_ext[i,j]-img_ext[i+1,j])**2 + (img_ext[i,j]-img_ext[i-1,j])**2)

    # cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
    # return cg
    # return b

