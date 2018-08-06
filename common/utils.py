import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from PIL import Image

# Input: Numpy Array
def show_image(img, transform = True):
    if transform == True:
        # npimg = img.numpy()
        plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')


def var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def save_image(pic, path=None):
    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if path == None:
        show_image(ndarr, transform = False)
    else:
        im = Image.fromarray(ndarr)
        im.save(path)
