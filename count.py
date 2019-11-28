#!/usr/bin/env python3

import torch
import sys
import numpy as np
# from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from PIL import Image

from config import C, LOG
from model import CSRNet


def count(img):
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


def count_frag(img, t_width=1000, name="None"):
    o_width, o_height = img.size
    factor = int(round(o_width/t_width, 0))
    n_width = int(o_width/factor)
    n_height = int(o_height/factor)
    n = 0
    for i in range(0, o_height, n_height):
        for j in range(0, o_width, n_width):
            box = (j, i, j+n_width, i+n_height)
            c_image = img.crop(box)
            c = count(c_image)
            n += c
            print(f"\t{n:6d} {c:+6d}")
    print(f"\t{n:6d}")
    with open(LOG, "a") as f:
        f.write(f"{name}[{t_width}]:{n}\n")
    return n


def main(img_name, t_width=1200):
    count_frag(Image.open(img_name), t_width, img_name.split("/")[-1])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        main(sys.argv[1])
