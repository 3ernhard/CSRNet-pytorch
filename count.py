#!/usr/bin/env python3

import torch
import sys
import numpy as np
# from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt, cm


from config import C
from model import CSRNet

img_path = sys.argv[1]

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
img = transform(Image.open(img_path).convert("RGB"))
if C["cuda"]:
    img = img.cuda()
else:
    img = img.cpu()

output = model(img.unsqueeze(0)).detach().cpu()
dmap = np.asarray(output.reshape(output.shape[2], output.shape[3]))
count = int(np.sum(dmap))

print("Predicted count:", count)
plt.imsave(f"{img_path}_{count}_dmap.jpg", dmap, cmap=cm.jet)
