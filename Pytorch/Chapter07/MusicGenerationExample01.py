import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import random
import skimage.io as io
from struct import pack, unpack
from io import StringIO, BytesIO
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import MIDI


def main():
    pass


if __name__ == "__main__":
    print("> Music Generation Example 01")
    main()
