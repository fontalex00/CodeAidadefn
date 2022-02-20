
import copy
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import stats
#from autoattack import AutoAttack

# Directories
DATAPATH  = 'D:\\Alexdir_heil\\Pyprox_Datadir'
CNETPATH  = 'D:\\Droploc\\CodeAidadefn\\Cnets_\\'
WHTSFILE  = 'D:\\Droploc\\CodeAidadefn\\dweights.c'
WHT1FILE  = 'D:\\Droploc\\CodeAidadefn\\dweight1.c'
CONSFILE  = 'D:\\Dropbox\\CodeAidadefn\\Source\\console.c'
TNETFILE  = 'D:\\Dropbox\\CodeAidadefn\\Source\\lastnet.pth'
BOARDFN   = 'D:\\Dropbox\\CodeAidadefn\\Source\\cznordo.c'

# Globals parameters
BUTCHSZA  = 200
BUTCHEND  = (60000/BUTCHSZA)-1
BLRATE    = 0.001
SBOL      =-0.0
BUDL      = 10.0
EPOCHS    = 00
HD        = [0,-0.10,-0.10,-0.10,-0.10]
UBIAS     = [0, 1.00, 2.00, 2.00, 1.00]
LBIAS     = [0, 0.00, 0.00, 0.00, 0.00]

closed = ''' '''