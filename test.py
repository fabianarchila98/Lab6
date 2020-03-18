# import torch libraries
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

# import the utility functions
from model import HED
from dataproc import TestDataset

rng = np.random.RandomState(37148)

# create instance of HED model
net = HED()
net.cuda()

# load the weights for the model
net.load_state_dict(torch.load('train/HED.pth'))

# batch size
nBatch = 1

# load the images dataset
dataRoot = './data/HED-BSDS/'

# create data loaders from dataset
testPath = dataRoot + 'test.lst'
testDataset = TestDataset(testPath, dataRoot)
testDataloader = DataLoader(testDataset, batch_size=nBatch)


def grayTrans(img):
    img = img.numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    return img


for i, sample in enumerate(testDataloader):
    # get input sample image
    inp, fname = sample
    print(fname)
    inp = inp.permute(0, 3, 1, 2).float()
    inp = Variable(inp.cuda())

    # perform forward computation
    s1, s2, s3, s4, s5, s6 = net.forward(inp)

    # convert back to numpy arrays
    out = []
    out.append(grayTrans(s6.data.cpu()))
    out.append(grayTrans(s1.data.cpu()))
    out.append(grayTrans(s2.data.cpu()))
    out.append(grayTrans(s3.data.cpu()))
    out.append(grayTrans(s4.data.cpu()))

    print(fname)
    img = Image.fromarray(out[0], 'L')
    img.save(fname[0].split('.', 1)[0]+'.png')
