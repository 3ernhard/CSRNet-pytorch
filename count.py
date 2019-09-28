#!/usr/bin/env python3

import torch
import sys
import numpy as np
# from torchvision.transforms.functional import to_tensor
from torchvision import transforms
# from matplotlib import pyplot as plt, cm
from PIL import Image

from config import C
from model import CSRNet


def count(img, plot=False):
    model = CSRNet()
    if C["cuda"]:
        model = model.cuda()
        pth = torch.load(C["pth"])
    else:
        model = model.cpu()
        pth = torch.load(C["pth"], map_location="cpu")
    model.load_state_dict(pth["state_dict"])
    # img = 255.0 * to_tensor(Image.open(img_path).convert("RGB"))
    # for i in range(3):
    #     img[i,:,:] = img[i,:,:] + C["img_corr"][i]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=C["mean"], std=C["std"])
    ])
    img = transform(img.convert("RGB"))
    if C["cuda"]:
        img = img.cuda()
    else:
        img = img.cpu()
    output = model(img.unsqueeze(0)).detach().cpu()
    dmap = np.asarray(output.reshape(output.shape[2], output.shape[3]))
    count = int(np.sum(dmap))
    return count


def count_frag(img, t_width=1500, name="None", plot=False):
    o_width, o_height = img.size
    factor = int(round(o_width/t_width, 0))
    n_width = int(o_width/factor)
    n_height = int(o_height/factor)
    n = 0
    for i in range(0, o_height, n_height):
        for j in range(0, o_width, n_width):
            box = (j, i, j+n_width, i+n_height)
            c_image = img.crop(box)
            c = count(c_image, plot=plot)
            n += c
            print(f"\t{n:6d} {c:+6d}")
    print(f"\t{n:6d}")
    with open("./count.log", "a") as f:
        f.write(f"{n:>8}    {name}\n")
    return n


if __name__ == "__main__":
    count_frag(Image.open(sys.argv[2]), name=sys.argv[2])
