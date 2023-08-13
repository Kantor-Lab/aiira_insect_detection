import datetime
import os.path
from torch.backends import cudnn
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import pandas as pd
from itertools import compress
from collections import OrderedDict
import csv


def transforms_validation(image):
    crop_size=224
    resize_size=256
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    interpolation=InterpolationMode.BILINEAR
    transforms_val = transforms.Compose(
                    [
                    transforms.Resize(resize_size, interpolation=interpolation),
                    transforms.CenterCrop(crop_size),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std)])
    image = Image.fromarray(np.uint8(image))
    image=transforms_val(image).reshape((1,3,224,224))
    return image



def evaluate(model, image, cmnDf, classes_file='classes.txt'):
    model.eval()
    device = torch.device('cpu')
    image = transforms_validation(image)
    file = open(classes_file, 'r')
    classes = []
    content = file.readlines()
    for i in content:
        spl = i.split('\n')[0]
        classes.append(spl)
    with torch.inference_mode():
        image = image.to(device, non_blocking=True)
        output = model(image)
        logps = output
        T = 1
        energy = -(T * torch.logsumexp(logps / T, dim=1))
        energy = torch.reshape(energy, (-1, 1))
        energy = energy[0][0].numpy()
        confirmed = False
        other_classes = {}
        if energy < 11.49:
            confirmed = True
            smx = torch.nn.functional.softmax(output)
            smx = smx.numpy()[0]
            op = smx
            qhat = 0.935176  # alpha = 0.025
            op = op >= 1 - qhat
            softmax = list(compress(smx, op))
            names = list(compress(classes, op))
            sortedNames = [x for _, x in sorted(zip(softmax, names))]
            sortedSoftmax = [y for y, _ in sorted(zip(softmax, names))]
            sortedNames = sortedNames[::-1]
            sortedSoftmax = sortedSoftmax[::-1]
            op_ix = torch.argmax(torch.tensor(smx))
            sumSoftmax = sum(sortedSoftmax)
            if len(sortedSoftmax) > 1:
                sortedSoftmax = sortedSoftmax / sumSoftmax * 100;
                softmax_class_dict = OrderedDict(zip(sortedNames, sortedSoftmax))
                for k in list(softmax_class_dict.keys())[1:]:
                    sc_name = k
                    cmn_name = cmnDf.loc[cmnDf['Scientific Name'] == sc_name, 'Common Name'].iloc[0]
                    other_classes[sc_name] = cmn_name
        op = torch.nn.functional.softmax(output)
        op_ix = torch.argmax(op)
        sciPred = classes[op_ix]
        cmnPred = cmnDf.loc[cmnDf['Scientific Name'] == sciPred, 'Common Name'].iloc[0]
        order = cmnDf.loc[cmnDf['Scientific Name'] == sciPred, 'Order'].iloc[0]
    return sciPred, cmnPred, confirmed, order, other_classes