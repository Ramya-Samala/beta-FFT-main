from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT


def get_device():
    """Get the best available device (CUDA GPU or CPU)"""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def net_factory(net_type="unet", in_chns=1, class_num=3, model_dict=None):
    device = get_device()
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).to(device)
    else:
        net = None
    return net
