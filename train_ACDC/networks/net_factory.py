from networks.unet import UNet, UNet_2d
import torch.nn as nn
import torch

def get_device():
    """Get the appropriate device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    device = get_device()
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)
    return net

def BCP_net(in_chns=1, class_num=4, ema=False):
    device = get_device()
    net = UNet_2d(in_chns=in_chns, class_num=class_num).to(device)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

